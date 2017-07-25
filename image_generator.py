from ops import *
import math
class wgenerator(object):
    def __init__(self, word_to_idx, batch_size, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024,
                 n_time_step=16, learning_rate=0.001,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.batch_size = batch_size
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']
        self.p = 3
        self.reuse = False
        self.learning_rate = learning_rate
        self.grad_clip = 5.0

        tf.set_random_seed(1234)

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features_t = tf.placeholder(tf.float32, [self.batch_size, 14, 14, self.D])
        self.features = tf.reshape(self.features_t, (self.batch_size, self.L, self.D))
        self.captions = tf.placeholder(tf.int32, [self.batch_size, self.T + 1])
        self.z = tf.placeholder(tf.float32, [self.batch_size, 16])
        self.captions_onehot = tf.placeholder(tf.float32, [self.batch_size, self.T + 1, self.V])

        #        self.learning_rate = tf.train.exponential_decay(
        #              self.learning_rate,                # Base learning rate.
        #              5 * (self.n_iters_per_epoch),  # Current index into the dataset.
        #              self.n_iters_per_epoch,          # Decay step.
        #              0.9,                # Decay rate.
        #              staircase=True)

    def _get_initial_lstm(self, features, z):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)
            features_mean = tf.concat([features_mean, z], axis=1)
            w_h = tf.get_variable('w_h', [self.D+16, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D+16, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return (c, h)
    def _get_initial_lstm_d(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)
            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return (c, h)

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse or self.reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)
            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits  # (N,V)

    def _decode_lstm_d(self, x, h, context, result, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)
            r_logits = tf.nn.tanh(fully_connected(result, self.M, 'r_emb'))
            out_logits = tf.reduce_sum(r_logits*h_logits, 1)
            return out_logits  # (N,V)
    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def caption_generator(self, features, features_proj, state, caption, lstm_cell, t):
        with tf.variable_scope('generator', reuse=(t!=0)):
            context, alpha = self._attention_layer(features, features_proj, state[1], reuse=(t != 0))
            xt = self._word_embedding(inputs=caption)
            if self.selector:
                context, beta = self._selector(context, state[1], reuse=(t != 0))
            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, state = lstm_cell(inputs=tf.concat([xt, context],1), state=state)

            result_logits = self._decode_lstm(xt, state[1], context, dropout=self.dropout,
                                       reuse=(t != 0))  # (N,V)
            result = tf.nn.softmax(result_logits)
            return result, result_logits, alpha, state, xt, context

    def caption_discrimiator(self, features, features_proj, state_d, xt, result,lstm_cell_d,  reuse):
        with tf.variable_scope('discrimiator', reuse=reuse):
            context, alpha = self._attention_layer(features, features_proj, state_d[1], reuse=reuse)
            if self.selector:
                context, beta = self._selector(context, state_d[1], reuse=reuse)
            with tf.variable_scope('lstm', reuse=reuse):
                _, state_d = lstm_cell_d(inputs=tf.concat([xt, context], 1), state=state_d)

            out_logits = self._decode_lstm_d(xt, state_d[1], context, result, dropout=self.dropout,
                                              reuse=reuse)  # (N,V)
            out = tf.nn.sigmoid(out_logits)
            return out, out_logits, state_d


    def caption_discrimiator_full(self, state, state_d, lstm_cell, lstm_cell_d, action, features, features_proj, features_proj_d,  k, reuse=False):
        out_list = []
        for i in range(min(self.T-k, self.p)):
            result, result_logits, alpha, state, xt, context = self.caption_generator(features, features_proj, state,
                                                                                  action, lstm_cell, 1)
            out, out_logits, state_d = self.caption_discrimiator(features, features_proj_d, state_d, xt, result, lstm_cell_d, reuse)
            action = tf.cast(tf.argmax(result, 1), tf.int32)
            out_list.append([out, out_logits])
        return out_list

    def mcs(self,word_list, state, features, features_proj, lstm_cell, k):
        sample_list = []
        next_token = word_list[:,k]
        for t in range(self.T-k):
            result, result_logits, alpha, state, xt, context = self.caption_generator(features, features_proj,
                                                                                        state, next_token, lstm_cell, t+1)
            next_token = tf.cast(tf.reshape(tf.argmax(result_logits, 1), [self.batch_size]), tf.int32)
            sample_list.append(tf.expand_dims(next_token, 1))
        sample_list = tf.concat(sample_list, 1)
        sample_list = tf.concat([word_list, sample_list], 1)
        return sample_list

    def build_model(self, mode, reuse=False):
        self.reuse = reuse
        features = self.features
        captions = self.captions
        z = self.z
        # batch normalize feature vectors
        features = self._batch_norm(features, mode=mode, name='conv_features_generator_discrimiator')
        with tf.variable_scope('generator_f'):
            state_t = self._get_initial_lstm(features=features, z=z)
            features_proj = self._project_features(features=features)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H, reuse=self.reuse)
        with tf.variable_scope('discrimiator_f'):
            state_t_d = tf.stop_gradient(state_t)
            features_proj_d = self._project_features(features=features)
            lstm_cell_d = tf.contrib.rnn.BasicLSTMCell(num_units=self.H, reuse=self.reuse)

        state = tf.contrib.rnn.LSTMStateTuple(state_t[0], state_t[1])
        state_d = tf.contrib.rnn.LSTMStateTuple(state_t_d[0], state_t_d[1])
        initial_stae = state
        initial_stae_d = state_d
        d_loss_total = 0.0
        g_loss_total = 0.0
        for t in range(self.T):
            old_state = state
            old_state_d = state_d
            with tf.device("/cpu:0"):
                ct = tf.one_hot(captions[:, t + 1], self.V)
            result, result_logits, alpha, state, xt, context = self.caption_generator(features, features_proj, state,
                                                                                      captions[:, t], lstm_cell, t)

            D, D_logits, state_d = self.caption_discrimiator(features, features_proj_d, state_d, xt, ct, lstm_cell_d,
                                                             reuse=(t != 0))
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
            d_loss_total += d_loss_real / 16
            if t % self.p == 0:
                out_list = self.caption_discrimiator_full(old_state, old_state_d, lstm_cell, lstm_cell_d,
                                                          captions[:, t], features, features_proj, features_proj_d, t,
                                                          True)
                d_loss_fake = 0.0
                g_loss = 0.0
                for i in range(min(self.p, self.T - t)):
                    d_loss_fake += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=out_list[i][1],
                                                                labels=tf.zeros_like(out_list[i][0])))
                    g_loss += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=out_list[i][1],
                                                                labels=tf.ones_like(out_list[i][0])))
                d_loss_total += d_loss_fake / min(self.p, self.T - t) / math.ceil(16 / self.p)
                g_loss_total += g_loss / min(self.p, self.T - t) / math.ceil(16 / self.p)

        word_list = tf.expand_dims(captions[:, 0], 1)
        state = initial_stae
        sample_list = self.mcs(word_list, state, features, features_proj, lstm_cell, 0)
        return d_loss_total, g_loss_total, self.features_t, self.captions, self.idx_to_word, sample_list, self.z

    def build_model_noraml(self, mode):
        features = self.features
        captions = self.captions
        z = self.z
        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        # batch normalize feature vectors
        features = self._batch_norm(features, mode=mode, name='conv_features_generator_discrimiator')
        with tf.variable_scope('generator_f'):
            state_t = self._get_initial_lstm(features=features, z=z)
            x = self._word_embedding(inputs=captions_in)
            features_proj = self._project_features(features=features)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        with tf.variable_scope('discrimiator_f'):
            state_t_d = self._get_initial_lstm_d(features=features)
            lstm_cell_d = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        state_d = tf.contrib.rnn.LSTMStateTuple(state_t_d[0], state_t_d[1])
        state = tf.contrib.rnn.LSTMStateTuple(state_t[0], state_t[1])
        loss = 0.0
        alpha_list, loss_list = [], []
        result_list = tf.expand_dims(captions[:, 0], 1)
        for t in range(self.T):
            result, result_logits, alpha ,state, xt, context = self.caption_generator(features, features_proj, state, captions[:,t], lstm_cell, t)
            alpha_list.append(alpha)
            word = tf.cast(tf.reshape(tf.argmax(result, 1), [self.batch_size]), tf.int32)
            result_list = tf.concat([result_list, tf.expand_dims(word, 1)], 1)
            loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_logits, labels=captions_out[:, t])  # (N) log opr after softmax
            loss += tf.reduce_sum(loss_vec)

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16. / 196 - alphas_all) ** 2)
            loss += alpha_reg
        self.pretrain_loss = loss / tf.to_float(self.batch_size)

        return self.pretrain_loss, self.features_t, self.captions, self.idx_to_word, result_list, self.z
    def build_model_test(self, mode):
        features = self.features
        z = self.z
        # batch normalize feature vectors
        features = self._batch_norm(features, mode=mode, name='conv_features_generator_discrimiator')
        with tf.variable_scope('generator_f'):
            state_t = self._get_initial_lstm(features=features, z=z)
            features_proj = self._project_features(features=features)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        state = tf.contrib.rnn.LSTMStateTuple(state_t[0], state_t[1])
        result_list = []
        next_token = tf.fill([tf.shape(features)[0]], self._start)
        for t in range(self.T):
            result, result_logits, alpha ,state, xt, context = self.caption_generator(features, features_proj, state, next_token, lstm_cell, t)
            next_token = tf.cast(tf.reshape(tf.argmax(result, 1), [self.batch_size]), tf.int32)
            result_list.append(next_token)
        return self.features_t, self.idx_to_word, result_list, self.z