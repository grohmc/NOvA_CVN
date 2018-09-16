'''
Useful functions for working with pixel maps
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# A var for pixel maps.
def cvnmap(tables):
    df = tables['rec.training.cvnmaps']
    return df.cvnmap

# creates a dataframe with neutrino flavor category
def intcat(tables):
    df=tables['rec.mc.nu']
    df.loc[(df.iscc==1) & (np.abs(df.pdg)==12),'intcat'] = 0
    df.loc[(df.iscc==1) & (np.abs(df.pdg)==14),'intcat'] = 1
    df.loc[(df.iscc==1) & (np.abs(df.pdg)==16),'intcat'] = 2
    df.loc[df.iscc==0,'intcat'] = 3

    return df.intcat.astype(np.uint8)

# A dataframe with flavor category and the pixelmap given some cuts
def trainingdf(tables,cut):
    cat = intcat(tables)[cut]
    pm  = cvnmap(tables)[cut]

    df = pd.concat([cat,pm], axis=1)
    df = df[df.cvnmap == df.cvnmap] # remove empty pixel maps

    return df.fillna(4)

# Convert a pixelmap from a dataframe to the correctly shaped numpy array
def pmdftonp(df):
    maps = np.array(df.values.tolist())
    # Transpose and reshape to have the correct orientation
    maps = np.reshape(maps,(-1,2,100,80,1))
    maps = np.transpose(maps,(0,1,3,2,4))
    # Keras seems to prefer floats
    return maps/255.0

# Create a hist2 of pixel maps
def viewmap(tables,cut):
    df = cvnmap(tables)[cut]
    index= df.index
    maps = pmdftonp(df)
    maps = np.reshape(maps,(-1,2,80,100))

    for map,id in zip(maps,index):
        plt.clf()
        plt.figure(1,figsize=(12,6))

        plt.subplot(121)
        plt.imshow(map[0],cmap='binary')
        plt.xlabel('planes')
        plt.ylabel('cells')
        plt.title("XZ-View")
        #plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(122)
        plt.imshow(map[1],cmap='binary')
        plt.xlabel('planes')
        plt.ylabel('cells')
        plt.title("YZ-View")
        #plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()

        plt.savefig('images/pixelmap_r'+str(id[0])+'_s'+str(id[1])+'_c'+str(id[2])+'_e'+str(id[3])+'_sl'+str(id[4])+'.png')
        plt.close()

# Split a dataframe into training and testing parts
def dfsplit(df,frac=0.2):
    train,test = train_test_split(df, test_size=frac)
    return train, test
