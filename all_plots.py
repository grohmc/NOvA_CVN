import h5Utils
import matplotlib.pyplot as plt

tables = h5Utils.importh5('fardet_genie_nonswap_genierw_fhc_v08_1000_r00014041_s60_c000_R17-11-14-prod4reco.h5')

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
        h5Utils.plt_1D(data, k1, k2, ' ', 'Calorimetric Energy [GeV]')
