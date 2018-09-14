from h5Utils import KL
import numpy as np

def kECut(tables):
    df = tables['rec.slc']
    return (df.calE > 0) & (df.calE < 5)

def kHitsCut(tables):
    df = tables['rec.slc']
    return (df.nhit > 50) & (df.nhit < 600)

def kCVNmCut(tables):
    df = tables['rec.sel.cvn2017']
    return (df.numuid > 0.5)

def kContainCut(tables):
    df = tables['rec.sel.nuecosrej']
    return (df.distallpngtop > 20) & \
        (df.distallpngbottom > 10) & \
        (df.distallpngfront > 10) & \
        (df.distallpngback > 10) & \
        (df.distallpngwest > 10) & \
        (df.distallpngeast > 10)

def kPngCut(tables):
    df = tables['rec.vtx.elastic.fuzzyk']
    df['clean'] = (df.npng < 6)
    return df.clean.groupby(KL).agg(np.any)

def kMuCut(tables):
    muid = tables['rec.vtx.elastic.fuzzyk.png.cvnpart'].muonid
    len = tables['rec.vtx.elastic.fuzzyk.png'].len
    df = (muid > 0.5) | (len > 5)
    return df.groupby(KL).agg(np.any)

def kFullCut(tables):
    return kECut(tables) & kHitsCut(tables) & kCVNmCut(tables) & \
        kContainCut(tables) & kPngCut(tables) & kMuCut(tables)

def kCalEVar(tables):
    df = tables['rec.slc']
    return df.calE

def kNuPdg(tables):
    df = tables['rec.mc.nu']
    return df.pdg

def kNuInt(tables):
    df = tables['rec.mc.nu']
    return df['mode'] # mode is panda internal function. Awkward...

######################################################################################

def twoprong(tables):
    df = tables['rec.vtx.elastic.fuzzyk']
    df['twoprong'] = (df.npng == 2)
    return df.twoprong.groupby(KL).agg(np.any)

def kGammaCut(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart']
    df['isphoton'] = (df.photonid > 0.5)
    return df.isphoton.groupby(KL).agg(np.all)

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

    dot = x.groupby(KL).prod()+y.groupby(KL).prod()+z.groupby(KL).prod()

    EProd = df.calE.groupby(KL).prod()

    return 1000*np.sqrt(2*EProd*(1-dot))
