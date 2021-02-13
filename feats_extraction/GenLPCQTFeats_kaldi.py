import kaldi_io
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--access_type', action='store', type=str)
parser.add_argument('--work_dir', action='store', type=str)
opt = parser.parse_args()

access_type = opt.access_type
work_dir = opt.work_dir

cwd = os.getcwd()

for dirpath in [cwd+'/'+work_dir+'/'+access_type+'_train', cwd+'/'+work_dir+'/'+access_type+'_dev', cwd+'/'+work_dir+'/'+access_type+'_eval']:
    out_ark_scp = 'ark:| copy-feats ark:- ark,scp:%s.ark,%s.scp' %(dirpath+'/feats', dirpath+'/feats')

    data_list = [data for data in os.listdir(dirpath) if data[-4:] == '.npy']
    data_list.sort()

    with kaldi_io.open_or_fd(out_ark_scp, 'wb') as wf:
        for data in data_list:
            feats = np.load(dirpath+'/'+data)
            uttid = data[:-4]
            kaldi_io.write_mat(wf, feats, key=uttid)

