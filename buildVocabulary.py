#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:24:43 2017

@author: nlp
"""
from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *
#import sys
#sys.path.append('/usr/local/cuda/lib64')
#sys.path.append('/usr/local/cuda/include')
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json

def _process_caption_data_coco(caption_file, image_dir, max_length):
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
#        if len(caption.split(" ")) > max_length:
#            del_idx.append(i)
    
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

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 0}
    idx = 2
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print ("Max length of caption: ", max_len)
    return word_to_idx

batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
max_length = 15
# if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
word_count_threshold = 5


# about 80000 images and 400000 captions for train dataset
coco_annotations = _process_caption_data_coco(caption_file='./captions_train2014.json',
                                      image_dir='G:\\train2014',
                                      max_length=max_length)
word_to_idx = _build_vocab(annotations=coco_annotations, threshold=word_count_threshold)
save_pickle(word_to_idx, './data/word_to_idx_combined.pkl')
            