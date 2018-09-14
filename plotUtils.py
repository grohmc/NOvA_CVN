import matplotlib.pyplot as plt
import numpy as np

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
    n, bins, _ = h.hist(data, color='xkcd:pig pink', label=k1)
    h.title(title)
    #h.grid(True)
    h.xlabel(xname,fontsize= 14)
    h.ylabel(yname,fontsize= 14)
    h.legend(loc='upper right')
    h.text(0.99, 1.03, 'NOvA Simulation', color='gray', weight='light', fontsize= 14, horizontalalignment='right', verticalalignment='center', transform=h.gca().transAxes)
    h.show()
    # h.savefig('all/'+k1+'.'+k2+'.png')
    h.close()

    return n, bins

# compute the integral given hist contents and bin boundaries
def integral(n, bins, range=None, width=False):
    if not range:
        range=(0,len(bins))
    if width:
        return (np.diff(bins[slice(*range)])*n[slice(*range)]).sum()
    else:
        return n[slice(*range)].sum()
