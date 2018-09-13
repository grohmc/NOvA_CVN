'''
For importing an h5 file into a table dictionary
'''
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        if k1.startswith('spill') or k1.startswith('neutrino'):
            continue
        # Index the data
        df.set_index(KL, inplace=True)
        tables[k1] = df

    f.close()

    return tables
'''
def importh5s(h5_path):
    fnames = glob(os.path.join(h5_path, "**.h5"))
    f=h5py.File(fnames[0],'r')
    keys = f.keys()
    f.close()

    tables = {}

    for k in keys:
        dflist = []
        for fname in fnames:
            f=h5py.File(fname,'r')

            group = f.get(k)
            values = {}
            for k2 in group.keys():
                dataset = group.get(k2)
                values[k2] = dataset.value.flatten()
            dflist.append(pandas.DataFrame(values))
            f.close()
        df = pandas.concat(dflist)
        if not k.startswith('spill') :
            df.set_index(KL, inplace=True)
        tables[k] = df

    return tables
'''
#####################################
# Make plots from data tables
#####################################

# Example 1D plot
def plt_1D(data, groupkey, varkey, title='title', xname='x', yname=' '):
    k1=groupkey
    k2=varkey
    h   = plt
    fig = h.figure(1,(6,4))
    # ax  = fig.subplot()
    h.hist(data, color='xkcd:pig pink', label=k1)
    h.title(title)
    #h.grid(True)
    h.xlabel(xname,fontsize= 14)
    h.ylabel(yname,fontsize= 14)
    h.legend(loc='upper right')
    h.text(0.99, 1.03, 'NOvA Simulation', color='gray', weight='light', fontsize= 14, horizontalalignment='right', verticalalignment='center', transform=h.gca().transAxes)
    h.show()
    # h.savefig('all/'+k1+'.'+k2+'.png')
    h.close()
