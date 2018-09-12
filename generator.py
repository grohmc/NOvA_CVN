'''
Produce pixel map data from NOvA h5 files
'''
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import pmUtils

def data_generator(df,batch_size,num_classes,shuffle=True):
    while True:
        if shuffle:
            df = df.sample(frac=1)

        for i in range(0, df.shape[0], batch_size):
            cats = df.intcat[i:i+batch_size]
            cats = cats.values
            maps = df.cvnmap[i:i+batch_size]
            maps = pmUtils.pmdftonp(maps)

            inputx = maps[:,0]
            inputy = maps[:,1]

            yield [inputx,inputy],[to_categorical(cats,num_classes=num_classes)]
