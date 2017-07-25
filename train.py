from image_generator import wgenerator
from core.utils import *
import tensorflow as tf
import json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def train():
    data = load_coco_data()
    batch_size = 64
    captions_file = data['captions']
    filenames = data['file_names']
    word2idx = data['word_to_idx']
    img_id = data['image_idxs']
    model = wgenerator(batch_size=batch_size, word_to_idx=data['word_to_idx'], dropout=False, prev2out=False, ctx2out=False)
    epoches = 20001
    d_loss_total, g_loss_total, features, captions, idx2word, result_list, z = model.build_model(mode='train')
    idx2word[8787]='U'
    t_vars = tf.global_variables()
    saver = tf.train.Saver(t_vars)
    train_vars = tf.trainable_variables()
    d_vars = [var for var in train_vars if 'discrimiator' in var.name]
    g_vars = [var for var in train_vars if 'generator' in var.name]
    d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.25).minimize(d_loss_total, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.25).minimize(g_loss_total, var_list=g_vars)
    #b_vars = [var for var in t_vars if 'discrimiator' not in var.name]
    #d_one = [var for var in t_vars if 'discrimiator' in var.name]
    #saver_re1 = tf.train.Saver(b_vars)
    #saver_re2 = tf.train.Saver(d_one)
    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)
    init = tf.initialize_all_variables()
    sess.run(init)
    #saver_re1.restore(sess, './gmodel_normal.ckpt')
    #saver_re2.restore(sess, './gmodel_discrimiator.ckpt')
    saver.restore(sess, './gmodel_2.ckpt')
    image_features = np.zeros((batch_size, 14, 14, 512), dtype=np.float32)
    for i in range(epoches):
        for ii in range(1):
            zs = np.random.uniform(-1, 1, [batch_size, 16])
            seed = np.random.randint(0, captions_file.shape[0], size=(batch_size))
            file_list = [filenames[img_id[w]] for w in seed]
            caption_list = [captions_file[w] for w in seed]
            caption_list = np.array(caption_list)
            for j in range(batch_size):
                image_features[j] = np.load(file_list[j].replace('jpg', 'npy'))
            sess.run(d_optim,
                         feed_dict={features: image_features, captions: caption_list, z:zs})

        for ii in range(2):
            zs = np.random.uniform(-1, 1, [batch_size, 16])
            seed = np.random.randint(0, captions_file.shape[0], size=(batch_size))
            file_list = [filenames[img_id[w]] for w in seed]
            caption_list = [captions_file[w] for w in seed]
            caption_list = np.array(caption_list)
            for j in range(batch_size):
                image_features[j] = np.load(file_list[j].replace('jpg', 'npy'))
            sess.run(g_optim,
                     feed_dict={features: image_features, captions: caption_list, z: zs})
        if i % 50 == 0:
            g, d, rs = sess.run([g_loss_total, d_loss_total, result_list],
                                feed_dict={features: image_features, captions: caption_list,
                                           z: zs})
            print(decode_captions(rs[0:5], idx2word))
            print(decode_captions(caption_list[0:5], idx2word))
            print("gloss %.8f, dloss %0.8f" % (g, d))
        if i % 1000 == 0 and i!=0:
            saver.save(sess, './gmodel_4.ckpt')
            print('*********    model saved    *********')
    sess.close()

def train_noraml():
    data = load_coco_data()
    batch_size = 64
    captions_file = data['captions']
    filenames = data['file_names']
    word2idx = data['word_to_idx']
    img_id = data['image_idxs']
    model = wgenerator(batch_size=batch_size, word_to_idx=data['word_to_idx'], dropout=True, prev2out=False, ctx2out=False)
    epoches = 20001
    loss, features, captions, idx2word, result_list, z = model.build_model_noraml(mode='train')
    # 优化算法采用 Adam
    optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss)
    t= tf.global_variables()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)
    init = tf.initialize_all_variables()
    sess.run(init)
    saver.restore(sess, './gmodel_normal.ckpt')
    image_features = np.zeros((batch_size, 14, 14, 512), dtype=np.float32)
    for i in range(epoches):
        zs = np.random.uniform(-1,1, [batch_size, 16])
        seed = np.random.randint(0, captions_file.shape[0], size=(batch_size))
        file_list = [filenames[img_id[w]] for w in seed]
        caption_list = [captions_file[w] for w in seed]
        caption_list = np.array(caption_list)
        for j in range(batch_size):
            image_features[j] = np.load(file_list[j].replace('jpg', 'npy'))
        sess.run(optim,
                 feed_dict={features: image_features, captions: caption_list, z:zs})
        l, rs = sess.run([loss, result_list], feed_dict={features: image_features, captions: caption_list,
                                                                  z:zs})
        if i % 100 == 0:
            print(decode_captions(rs[0], idx2word))
            print(decode_captions(caption_list[0], idx2word))
            print("loss %.8f" % (l))
        if i % 1000 == 0:
            saver.save(sess, './gmodel_normal.ckpt')
            print('*********    model saved    *********')
def test():
    data = load_coco_data()
    batch_size = 64
    captions_file = data['captions']
    word2idx = data['word_to_idx']
    model = wgenerator(batch_size=batch_size, word_to_idx=data['word_to_idx'], dropout=False, prev2out=False, ctx2out=False)
    features, idx2word, result_list, z = model.build_model_test(mode='test')
    # 优化算法采用 Adam
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.InteractiveSession(config=config)
    saver.restore(sess, './gmodel_2.ckpt')
    image_features = np.zeros((batch_size, 14, 14, 512), dtype=np.float32)
    image_path = 'G:\\val2014\\'
    filenames = os.listdir(image_path)
    leng = len(filenames)
    answer = []
    for i in range(int(leng/batch_size)+1):
        zs = np.random.uniform(-1, 1, [batch_size, 16])
        file_list = filenames[i*batch_size:min(i*batch_size+batch_size, leng)]
        for j in range(min(batch_size, leng-i*batch_size)):
            image_features[j] = np.load(image_path + file_list[j])
        results = sess.run(result_list, feed_dict={features: image_features, z: zs})
        for j in range(min(batch_size, leng-i*batch_size)):
            ss = ''
            for jj in range(16):
                word = idx2word[int(results[jj][j])]
                if int(results[jj][j]) == 0 or int(results[jj][j]) ==1:
                    break
                if jj==0:
                    ss = word
                else:
                    ss = ss + ' ' +  word
            answer.append({'image_id': int(file_list[j][13:25]), 'caption': ss})
    dd = json.dump(answer, open('captions_val2014_normal_results.json', 'w'))

if __name__ == '__main__':
    train()
    #train_noraml()
    #test()
