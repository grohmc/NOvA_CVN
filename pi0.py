import h5Utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from matplotlib.patches import Rectangle

def twoprong(tables):
    df = tables['rec.vtx.elastic.fuzzyk']
    return (df.npng == 2).groupby(h5Utils.KL).agg(np.any)

def kGammaCut(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart']
    return (df.photonid > 0.5).groupby(h5Utils.KL).agg(np.all)

def kContain(tables):
    df = tables['rec.sel.nuecosrej']
    return (df.distallpngtop > 10) & \
        (df.distallpngbottom > 10) & \
        (df.distallpngfront > 10) & \
        (df.distallpngback > 10) & \
        (df.distallpngwest > 10) & \
        (df.distallpngeast > 10)

def kTruePi0(tables):
    df = tables['rec.sand.nue']
    return (df.npi0 > 0)

def kMass(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png']
    x = df['dir.x']
    y = df['dir.y']
    z = df['dir.z']

    l = np.sqrt(x*x+y*y+z*z)
    x=x/l
    y=y/l
    z=z/l

    dot = x.groupby(h5Utils.KL).prod()+y.groupby(h5Utils.KL).prod()+z.groupby(h5Utils.KL).prod()

    EProd = df.calE.groupby(h5Utils.KL).prod()

    deadscale = 0.8747

    return 1000*deadscale*np.sqrt(2*EProd*(1-dot))

tables = h5Utils.importh5('neardet_genie_nonswap_genierw_fhc_v08_1535_r00010921_s02_c002_R17-11-14-prod4reco.d_v1_20170322_204739_sim.repid.root.hdf5')

cut = twoprong(tables) & kGammaCut(tables) & kContain(tables)
cutSig = cut & kTruePi0(tables)
cutBkg = cut & ~kTruePi0(tables)

dfSig = kMass(tables)[cutSig]
dfBkg = kMass(tables)[cutBkg]

plt.figure(1,figsize=(6,4))
a,bins,_ = plt.hist([dfBkg, dfSig], 10, (0,300),\
                        histtype='step', color=['b','r'], stacked=True,\
                        label=['a','b'])

f=plt.fill_between(bins[:-1],0,a[0],step='post')

plt.legend(loc='upper right')

plt.xlabel('M$_{\gamma\gamma}$')
plt.ylabel('Events')

plt.show()
