###############################################################
#    Reconstruct the pi0 mass peak using ND MC in h5 files
#
#                              @
#
###############################################################

# Includes
import h5Utils
from core import cut, spectrum
# analysis packages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.stats import poisson_conf_interval
# For plotting
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

###########################################
# Look for vertices with two prongs
# Note: We don't need to explicitly check if there is a vertex like in CAFAna.
# Note 2: One liners can be done with lambda functions directly in the cut.
###########################################

kTwoProng = cut(lambda tables:
                (tables['rec.vtx.elastic.fuzzyk'].npng == 2).groupby(h5Utils.KL).agg(np.any))

###########################################
# Look for events where all prongs are photon like
# multiliners need to be done with def x(tables): blah return blah
# then x = cut(x)
###########################################
def kGammaCut(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart'].photonid
    return (df > 0.5).groupby(h5Utils.KL).agg(np.all)
kGammaCut = cut(kGammaCut)

# Loose containment
def kContain(tables):
    df = tables['rec.vtx.elastic']
    return (df['vtx.x'] < 190) & \
        (df['vtx.x'] > -190) & \
        (df['vtx.y'] < 190) & \
        (df['vtx.y'] > -190) & \
        (df['vtx.z'] < 1200) & \
        (df['vtx.z'] > 10)
kContain     = cut(kContain)

kPlaneGap    = cut(lambda tables:
                    (tables['rec.vtx.elastic.fuzzyk.png'].maxplanegap > 1).\
                    groupby(h5Utils.KL).agg(np.all))

kPlaneContig = cut(lambda tables:
                        (tables['rec.vtx.elastic.fuzzyk.png'].maxplanecont > 4).\
                        groupby(h5Utils.KL).agg(np.all))

# Does the event actually have a pi0?
kTruePi0     = cut(lambda tables:
                    tables['rec.sand.nue'].npi0 > 0)

###########################################
# Computes the invariant mass of two prong events
###########################################
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
    x = x/l
    y = y/l
    z = z/l

    # compute the dot product
    dot = x.groupby(h5Utils.KL).prod()+y.groupby(h5Utils.KL).prod()+z.groupby(h5Utils.KL).prod()

    # multiply the energy of all prongs in each event together
    EProd = df.calE.groupby(h5Utils.KL).prod()

    # return a dataframe with a single column of the invariant mass
    deadscale = 0.8747
    return 1000*deadscale*np.sqrt(2*EProd*(1-dot))

    # note NaNs can be removed by (df == df)

if __name__ == '__main__':

    # A 'loader'
    #tables = h5Utils.importh5('NDMC/neardet_genie_nonswap_genierw_fhc_v08_1535_r00010921_s02_c002_R17-11-14-prod4reco.d_v1_20170322_204739_sim.repid.root.hdf5')
    tables = h5Utils.importh5s('NDMC/')

    # Define cuts
    # The cut class knows how to interpret & and ~
    cutTot = kTwoProng & kGammaCut & kContain & kPlaneContig & kPlaneGap
    cutBkg = cutTot & ~kTruePi0

    # Make Spectra
    data = spectrum(kMass, cutTot, tables)
    tot  = spectrum(kMass, cutTot, tables)
    bkg  = spectrum(kMass, cutBkg, tables)

    POT = data.POT()

    print('Found ' + str(data.POT()) + ' POT. Scaling to ' + str(POT) + ' POT.')

    print('Selected ' + str(data.entries()) + ' events in data.')
    print('Selected ' + str(tot.entries()) + ' events in MC.')
    print('Selected ' + str(bkg.entries()) + ' background.')

    # Do an analysis!
    # With Spectra
    inttot = tot.integral(POT=POT)
    intbkg = bkg.integral(POT=POT)
    pur = (inttot - intbkg) / inttot
    print('This selection has a pi0 purity of ' + str(pur))

    # With histograms
    nbins = 8
    range = (0,400)
    d, bins = data.histogram(nbins,range, POT=POT)
    m, _    = tot.histogram(nbins,range, POT=POT)
    b, _    = bkg.histogram(nbins,range, POT=POT)

    def gaussian(x, x0, a, stdev, o):
        return a * np.exp( - ((x - x0) / stdev) ** 2 / 2) + o

    centers = (bins[:-1] + bins[1:])/2

    # A bug in scipy.optimize.curvefit requires these to be float64s instead of float32s.
    dataparam, datacov = curve_fit(gaussian, centers.astype(np.float64), d, p0=[135., np.max(d), 15., 0])
    mcparam,   mccov   = curve_fit(gaussian, centers.astype(np.float64), m, p0=[135., np.max(m), 15., 0])

    dataerr = np.sqrt(np.diag(datacov))
    mcerr   = np.sqrt(np.diag(mccov))

    datamu = 'Data $\mu$: ' + '%.1f'%dataparam[0] + '$\pm$' + '%.1f'%dataerr[0]
    datasi = 'Data $\sigma$: ' + '%.1f'%dataparam[2] + '$\pm$' + '%.1f'%dataerr[2]
    mcmu   = 'MC $\mu$: ' + '%.1f'%mcparam[0] + '$\pm$' + '%.1f'%mcerr[0]
    mcsi   = 'MC $\sigma$: ' + '%.1f'%mcparam[2] + '$\pm$' + '%.1f'%mcerr[2]

    # <codecell>
    # Plots time
    plt.figure(1,figsize=(6,4))

    # A histogram with 1 entry in each bin, Use our data as the weights.
    plt.hist(bins[:-1], bins, weights=m, histtype='step', color='xkcd:red', label='$\pi^0$ Signal')
    plt.hist(bins[:-1], bins, weights=b, color='xkcd:dark blue', label='Background')

    # Compute some errors
    derr = poisson_conf_interval(d,'frequentist-confidence')
    plt.errorbar(centers, d, yerr=[d-derr[0],derr[1]-d], fmt='ko', label='Fake Data')

    plt.xlabel('M$_{\gamma\gamma}$')
    plt.ylabel('Events')

    # I want the legend for the pi0 signal to be a line instead of an empty box
    handles, labels = plt.gca().get_legend_handles_labels()
    handles[0] = plt.Line2D([], [], c=handles[0].get_edgecolor())

    # I want data listed first in the legend even tho we plotted it last
    handles[0],handles[1],handles[2] = handles[2],handles[0],handles[1]
    labels[0],labels[1],labels[2] = labels[2],labels[0],labels[1]

    plt.legend(loc='upper right', handles=handles, labels=labels)

    # Print the text for the fit parameters
    plt.text(0.7, 0.65, datamu, color='k', fontsize=12, horizontalalignment='left', verticalalignment='center', \
            transform=plt.gca().transAxes)
    plt.text(0.7, 0.59, datasi, color='k', fontsize=12, horizontalalignment='left', verticalalignment='center', \
            transform=plt.gca().transAxes)
    plt.text(0.7, 0.53, mcmu, color='xkcd:red', fontsize=12, horizontalalignment='left', verticalalignment='center', \
            transform=plt.gca().transAxes)
    plt.text(0.7, 0.48, mcsi, color='xkcd:red', fontsize=12, horizontalalignment='left', verticalalignment='center', \
            transform=plt.gca().transAxes)

    plt.show()
