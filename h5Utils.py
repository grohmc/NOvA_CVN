'''
For importing an h5 file into a table dictionary
'''
import h5py
import numpy as np
import pandas as pd
from glob import glob
import os

# How to index the data
KL = ['run', 'subrun', 'cycle', 'evt', 'subevt']

def importh5(h5_path):
    # Open an h5 file
    f=h5py.File(h5_path,'r')

    tables = {}

    # Loop over all groups
    for k1 in f.keys():
        group = f.get(k1)
        values = {}
        # Loop over all datasets
        for k2 in group.keys():
            # This takes awhile to read in
            if k2 == 'slicemap':
                continue
            dataset = group.get(k2).value
            # Everything is stored as a vector, so flatten if they should be scalars
            if dataset.shape[1] == 1:
                dataset = dataset.flatten()
            else:
                dataset = list(dataset)
            values[k2] = dataset
        df = pd.DataFrame(values)
        # Hack. These trees don't have events or slices
        if not (k1.startswith('spill') or k1.startswith('neutrino')):
            # Index the data
            df.set_index(KL, inplace=True)
        tables[k1] = df

    f.close()

    return tables

def getIndices(tables):
    df = tables['rec.slc']
    indices = df.index.values
    return pd.DataFrame(indices,columns=['index'])


def importh5s(h5_path):
    fnames = glob(os.path.join(h5_path, "**.h5"))
    f = h5py.File(fnames[0],'r')
    keys = list(f.keys()).copy()
    f.close()

    tables = {}

    for k1 in keys:
        dflist = []
        for fname in fnames:
            f=h5py.File(fname,'r')

            group = f.get(k1)
            values = {}
            for k2 in group.keys():
                if k2 == 'slicemap':
                    continue
                dataset = group.get(k2).value
                if dataset.shape[1] == 1:
                    dataset = dataset.flatten()
                else:
                    dataset = list(dataset)
                values[k2] = dataset
            dflist.append(pd.DataFrame(values))
            f.close()
        df = pd.concat(dflist)
        if not (k1.startswith('spill') or k1.startswith('neutrino')):
            df.set_index(KL, inplace=True)
        tables[k1] = df

    return tables
