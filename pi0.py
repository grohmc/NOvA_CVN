import h5Utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# Look for vertices with two prongs
# Note: We don't need to check if there is a vertex like in CAFAna.
def twoprong(tables):
    df = tables['rec.vtx.elastic.fuzzyk']
    return (df.npng == 2).groupby(h5Utils.KL).agg(np.any)

# Look for events where all prongs are photon like
def kGammaCut(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart']
    return (df.photonid > 0.5).groupby(h5Utils.KL).agg(np.all)

# Loose containment to ensure some energy reconstruction
def kContain(tables):
    df = tables['rec.sel.nuecosrej']
    return (df.distallpngtop > 10) & \
        (df.distallpngbottom > 10) & \
        (df.distallpngfront > 10) & \
        (df.distallpngback > 10) & \
        (df.distallpngwest > 10) & \
        (df.distallpngeast > 10)

# Does the event actually have a pi0?
def kTruePi0(tables):
    df = tables['rec.sand.nue']
    return (df.npi0 > 0)

# Computes the invariant mass of two prong events
def kMass(tables):
    # Note: We could leave this check out, but you would get a warning about taking
    # the sqrt of negative numbers at the end (it won't crash like cafana)
    check = tables['rec.vtx.elastic.fuzzyk'].npng == 2

    df = tables['rec.vtx.elastic.fuzzyk.png'][check]
    x = df['dir.x']
    y = df['dir.y']
    z = df['dir.z']

    # Compute the length of the dir vector and then normalize
    l = np.sqrt(x*x+y*y+z*z)
    x=x/l
    y=y/l
    z=z/l

    # compute the dot product
    dot = x.groupby(h5Utils.KL).prod()+y.groupby(h5Utils.KL).prod()+z.groupby(h5Utils.KL).prod()

    # multiply the energy of all prongs in each event together
    EProd = df.calE.groupby(h5Utils.KL).prod()

    deadscale = 0.8747

    # return a dataframe with a single column of the invariant mass
    return 1000*deadscale*np.sqrt(2*EProd*(1-dot))

# compute the integral given hist contents and bin boundaries
def integral(n, bins, range=None, width=False):
    if not range:
        range=(0,len(bins))
    if width:
        return (np.diff(bins[slice(*range)])*n[slice(*range)]).sum()
    else:
        return n[slice(*range)].sum()

tables = h5Utils.importh5('neardet_genie_nonswap_genierw_fhc_v08_1535_r00010921_s02_c002_R17-11-14-prod4reco.d_v1_20170322_204739_sim.repid.root.hdf5')

cut = twoprong(tables) & kGammaCut(tables) & kContain(tables)
cutSig = cut & kTruePi0(tables)
cutBkg = cut & ~kTruePi0(tables)

data = kMass(tables)[cut]
dfTot = kMass(tables)[cut]
dfBkg = kMass(tables)[cutBkg]

plt.figure(1,figsize=(6,4))

# could use np.histogram to get these values, but we are plotting anyway so grab them here
totn,bins,_ = plt.hist(dfTot, 10, (0,300), color='xkcd:red', label='$\pi^0$ Signal', histtype='step')
bkgn,_,_ = plt.hist(dfBkg, 10, (0,300), color='xkcd:dark blue', label='Background')

plt.xlabel('M$_{\gamma\gamma}$ (MeV)')
plt.ylabel('Events')

# we need to subtract the bkg
intBkg = integral(bkgn, bins)
intSig = integral(totn-bkgn, bins)

pur = intSig / (intBkg + intSig)
print('This selection has a pi0 purity of ' + str(pur))

# There must be a more direct way to do this.
d,_ = np.histogram(data,bins)
centers = (bins[:-1] + bins[1:])/2
plt.errorbar(centers,d, yerr=np.sqrt(d), fmt='ko', label='Fake Data')

plt.legend(loc='upper right')

plt.show()
