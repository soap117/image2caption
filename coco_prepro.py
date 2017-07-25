from scipy import ndimage
from collections import Counter
from core.vgg_19 import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import re
import pandas as pd
import hickle
import os
import json


def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]} 
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]
    
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print ("The number of captions before deletion: %d" %len(caption_data))
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print ("The number of captions after deletion: %d" %len(caption_data))
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print ("Max length of caption: ", max_len)
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print ("Finished building caption vectors")
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # about 80000 images and 400000 captions for train dataset
    train_dataset = _process_caption_data(caption_file='./captions_train2014.json',
                                          image_dir='G:\\train2014',
                                          max_length=max_length)
#    word_to_idx = _build_vocab(annotations=val_dataset, threshold=100)
#    word_to_idx = pd.DataFrame(word_to_idx.items(),columns=['word', 'cnt'])
#    word_to_idx.to_csv('coco_voca.txt',index=False,columns=['cnt','word'],sep='\t')
#    val_dataset.to_csv('coco_voca.txt',index=False,columns=['caption'])
    # about 40000 images and 200000 captions
    val_dataset = _process_caption_data(caption_file='./captions_val2014.json',
                                        image_dir='G:\\val2014',
                                        max_length=max_length)

    train_dataset =  pd.concat((train_dataset,val_dataset),ignore_index=True)
    # about 4000 images and 20000 captions for val / test dataset
    val_cutoff = int(0.1 * len(val_dataset))
    test_cutoff = int(0.2 * len(val_dataset))
    print ('Finished processing caption data')

    save_pickle(train_dataset, './data/train/train.annotations.pkl')
    save_pickle(val_dataset, './data/val/val.annotations.pkl')
#    save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
#    save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')
        
#    for split in ['train', 'val', 'test']:
    for split in ['train']:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))

#        if split == 'train':
#            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
#            save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)
        word_to_idx_combined = load_pickle('./data/word_to_idx_combined.pkl')           
        
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx_combined, max_length=max_length)
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
        print ("Finished building %s caption dataset" %split)
if __name__ == "__main__":
    main()
