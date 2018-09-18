import h5Utils
from core import cut, spectrum

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# Look for vertices with two prongs
# Note: We don't need to explicitly check if there is a vertex like in CAFAna.
# Note 2: One liners can be done with lambda functions directly in the cut.
kTwoProng = cut(lambda tables:
                (tables['rec.vtx.elastic.fuzzyk'].npng == 2).groupby(h5Utils.KL).agg(np.any))

# Look for events where all prongs are photon like
# multiliners need to be done with def x(tables): blah return blah
# then x = cut(x)
def kGammaCut(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart'].photonid
    return (df > 0.5).groupby(h5Utils.KL).agg(np.all)
kGammaCut = cut(kGammaCut)

# Loose containment to ensure some energy reconstruction
def kContain(tables):
    df = tables['rec.vtx.elastic']
    return (df['vtx.x'] < 190) & \
        (df['vtx.x'] > -190) & \
        (df['vtx.y'] < 190) & \
        (df['vtx.y'] > -190) & \
        (df['vtx.z'] < 1200) & \
        (df['vtx.z'] > 50)
kContain = cut(kContain)

kPlaneGap = cut(lambda x:
                    (tables['rec.vtx.elastic.fuzzyk.png'].maxplanegap > 1).\
                    groupby(h5Utils.KL).agg(np.all))

# Does the event actually have a pi0?
kTruePi0 = cut(lambda tables:
                    tables['rec.sand.nue'].npi0 > 0)

# Computes the invariant mass of two prong events
def kMass(tables):
    # Note: We could leave this check out, but you would get a warning about taking
    # the sqrt of negative numbers at the end (it won't crash like cafana).
    # dataframes can support NaNs just fine.
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

    # note NaNs can be removed by (df == df)

# A 'loader'
tables = h5Utils.importh5('neardet_genie_nonswap_genierw_fhc_v08_1535_r00010921_s02_c002_R17-11-14-prod4reco.d_v1_20170322_204739_sim.repid.root.hdf5')

cutTot = kTwoProng & kGammaCut & kContain & kPlaneGap
cutBkg = cutTot & ~kTruePi0

data = spectrum(kMass, cutTot, tables)
tot = spectrum(kMass, cutTot, tables)
bkg = spectrum(kMass, cutBkg, tables)

POT=5E20

print('Found ' + str(data.POT()) + ' POT. Scaling to ' + str(POT) + ' POT.')

inttot = tot.integral(POT=POT)
intbkg = bkg.integral(POT=POT)
pur = (inttot - intbkg) / inttot
print('This selection has a pi0 purity of ' + str(pur))

d, bins = data.histogram(6,(0,300), POT=POT)
m, _    = tot.histogram(6,(0,300), POT=POT)
b, _    = bkg.histogram(6,(0,300), POT=POT)

# Do an analysis!
# Maybe calculate the purity using the binning.

plt.figure(1,figsize=(6,4))

plt.hist(bins[:-1], bins, weights=m, histtype='step', color='xkcd:red', label='$\pi^0$ Signal')
plt.hist(bins[:-1], bins, weights=b, color='xkcd:dark blue', label='Background')

centers = (bins[:-1] + bins[1:])/2
plt.errorbar(centers, d, fmt='ko', label='Fake Data')

plt.xlabel('M$_{\gamma\gamma}$')
plt.ylabel('Events')

plt.legend(loc='upper right')

plt.show()
