from __future__ import print_function
import os
import numpy as np
import kaldi_io as ko

import argparse

# IMPORTANT: run this with python3 (don't use Nanxin's environment)
# Python implementation of Unified Feature Map 

def tensor_cnn_utt(mat, truncate_len):
    mat = np.swapaxes(mat, 0, 1)
    max_len = truncate_len * int(np.ceil(mat.shape[1]/truncate_len))
    repetition = int(max_len/mat.shape[1])
    tensor = np.tile(mat,repetition)
    repetition = max_len % mat.shape[1]
    rest = mat[:,:repetition]
    tensor = np.hstack((tensor,rest))
    
    return tensor


def construct_tensor(orig_feat_scp, ark_scp_output, truncate_len):
    with ko.open_or_fd(ark_scp_output, 'wb') as f:
        for key,mat in ko.read_mat_scp(orig_feat_scp):
            tensor = tensor_cnn_utt(mat, truncate_len)
            repetition = int(tensor.shape[1]/truncate_len)
            for i in range(repetition):
                sub_tensor = tensor[:,i*truncate_len:(i+1)*truncate_len]
                new_key = key + '-' + str(i)
                ko.write_mat(f, sub_tensor, key=new_key)


def construct_slide_tensor(orig_feat_scp, ark_scp_output, truncate_len):
    with ko.open_or_fd(ark_scp_output, 'wb') as f:
        for key,mat in ko.read_mat_scp(orig_feat_scp):
            tensor = tensor_cnn_utt(mat, truncate_len)
            repetition = int(tensor.shape[1]/truncate_len)
            repetition = 2 * repetition - 1 # slide 
            for i in range(repetition):
                sub_tensor = tensor[:,int(truncate_len/2)*i:int(truncate_len/2)*i+truncate_len]
                new_key = key + '-' + str(i)
                ko.write_mat(f, sub_tensor, key=new_key)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-scp', type=str, default='data/spec/PA_train/feats.scp')
    parser.add_argument('--out-scp', type=str, default='data/spec/PA_train/feats_slide.scp')
    parser.add_argument('--out-ark', type=str, default='data/spec/PA_train/feats_slide.ark')
    parser.add_argument('--truncate-len', type=int, default=400)

    args = parser.parse_args()

    ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + args.out_ark + ',' + args.out_scp
    construct_slide_tensor(args.in_scp, ark_scp, args.truncate_len)
     
