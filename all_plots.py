import h5Utils
import matplotlib.pyplot as plt

tables = h5Utils.importh5('fardet_genie_nonswap_genierw_fhc_v08_1000_r00014041_s60_c000_R17-11-14-prod4reco.h5')

# Make plots from data tables
def H1D(data, groupkey, varkey, title='title', xname='x', yname='y'):
    k1=groupkey
    k2=varkey
    h   = plt
    fig = h.figure(1,(6,4))
    ax  = fig.add_subplot()
    h.hist(data, color='hotpink', alpha=0.5, label=k1)
    h.title(title)
    h.grid(True)
    h.xlabel(xname)
    h.ylabel(yname)
    h.legend(loc='upper right')
    h.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    h.show()
    # h.savefig('all/'+k1+'.'+k2+'.png')
    h.close()

for k1 in tables.keys():
    df = tables[k1]
    for k2 in df.keys():
        if k2 == 'cvnmap':
            continue
        if k1 != 'rec.slc':
            continue
        if k2 != 'calE':
            continue

        print(k1+'.'+k2)
        data = df[k2]

        data = data[(data==data) & (data < 1E10) & (data > -1E10)]
        H1D(data, k1, k2, ' ', 'Calorimetric Energy [GeV]')
