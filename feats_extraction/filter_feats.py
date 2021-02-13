## Not all the audios under the directory are labeled by the protocols, so we need to only keep ones that have labels.

for DataType in ['train', 'dev', 'eval']:
    wavlistfile = 'data/%s/wav.scp' %(DataType)
    orifeatsscp = 'PA_LPCQT_16msHop48BPOfmin12/%s_feats.scp' %(DataType)
    newfeatsscp = 'PA_LPCQT_16msHop48BPOfmin12/%s_filtered_feats.scp' %(DataType)

    wavlist = []
    with open(wavlistfile, 'r') as rf:
        for line in rf.readlines():
            uttid = line.split()[0]
            wavlist.append(uttid)

    with open(orifeatsscp, 'r') as rf, open(newfeatsscp, 'w') as wf:
        for line in rf.readlines():
            uttid = line.split()[0]
            if uttid in wavlist:
                wf.write(line)

