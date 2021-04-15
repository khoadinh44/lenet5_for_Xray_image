import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.image as img
import time
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def load_model(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []
    
    print("Reading file {} image: ".format(train_path))
    for fold in classes:
        index = classes.index(fold)
        print('Loading {} files (Index: {})'.format(fold, index))
        path = os.path.join(train_path, fold, '*g')
        files = glob.glob(path)
        for file in files:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            file_base = os.path.basename(file)
            
            ids.append(file_base.split('.')[-3])
            cls.append(fold)
       
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    return images, labels, ids, cls
    
class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        print("\nNum example: {}\n".format(self._num_examples))
        
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # Convert from [0, 255] -> [0.0, 1.0].
        
        images = images.astype(np.float32) / 255.0
        
        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def ids(self):
        return self._ids
    
    @property
    def cls(self):
        return self._cls
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch 
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self.num_example:
#             Finish epoch
            self._epochs_completed += 1
            
            start = 0
            self._index_in_epoch += batch_size
            assert batch_size < self.num_examples
        end = self._index_in_epoch
        
        return self._images[start: end], self._labels[start: end], self._ids[start: end], self._cls[start: end]

def read_model_sets(train_path, image_size, classes):
    class Datasets(object):
        pass
    
    datasets = Datasets()
    
    images, labels, ids, cls = load_model(train_path, image_size, classes)
    datasets.train = DataSet(images, labels, ids, cls)
    return datasets