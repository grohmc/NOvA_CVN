import h5Utils
import matplotlib.pyplot as plt

tables = h5Utils.importh5('fardet_genie_nonswap_genierw_fhc_v08_1000_r00014041_s60_c000_R17-11-14-prod4reco.h5')

# Make plots from data tables
def H1D(data, groupkey, varkey, title='title', xname='x', yname='y'):
    k1=groupkey
    k2=varkey
    plt.figure(1,(6,4))
    plt.hist(data, color='hotpink', alpha=0.5, label=k1)
    plt.title(title)
    plt.grid(True)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(loc='upper right')
    plt.show()
    # plt.savefig('all/'+k1+'.'+k2+'.png')
    plt.close()

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
