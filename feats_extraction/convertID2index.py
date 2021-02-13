import argparse

# create multi-labels for model training 
def convert_combined(scp_file, systemID_file, out_file):
    ''' multi-class classification for PA and LA: totally 16 classies + 13 unseen LA attack classes
        (bonafide: 0), (AA: 1), (AB: 2), (AC: 3), (BA: 4), (BB: 5), (BC: 6), (CA: 7), (CB: 8), (CC: 9), (SS_1: 10), (SS_2: 11), (SS_4: 12), (US_1: 13), (VC_1: 14), (VC_4: 15)
    '''
    with open(scp_file) as f:
        temp = f.readlines()
    key_list = [x.strip().split()[0] for x in temp]

    with open(systemID_file) as f:
        temp = f.readlines()
    utt2ID = {x.strip().split()[0]:x.strip().split()[1] for x in temp}

    with open(out_file, 'w') as f:
        for key in key_list:
            curr_utt = ''.join(key.split('-')[0])
            label = utt2ID[curr_utt] 
            if label == '-':
                f.write('%s %d\n' % (key, 0))
            elif label == 'AA':
                f.write('%s %d\n' % (key, 1))
            elif label == 'AB':
                f.write('%s %d\n' % (key, 2))
            elif label == 'AC':
                f.write('%s %d\n' % (key, 3))
            elif label == 'BA':
                f.write('%s %d\n' % (key, 4))
            elif label == 'BB':
                f.write('%s %d\n' % (key, 5))
            elif label == 'BC':
                f.write('%s %d\n' % (key, 6))
            elif label == 'CA':
                f.write('%s %d\n' % (key, 7))
            elif label == 'CB':
                f.write('%s %d\n' % (key, 8))
            elif label == 'CC':
                f.write('%s %d\n' % (key, 9))
            elif label == 'A01':
                f.write('%s %d\n' % (key, 10))
            elif label == 'A02':
                f.write('%s %d\n' % (key, 11))
            elif label == 'A03':
                f.write('%s %d\n' % (key, 12))
            elif label == 'A04':
                f.write('%s %d\n' % (key, 13))
            elif label == 'A05':
                f.write('%s %d\n' % (key, 14))
            elif label == 'A06':
                f.write('%s %d\n' % (key, 15))
            elif label == 'A07':
                f.write('%s %d\n' % (key, 16))
            elif label == 'A08':
                f.write('%s %d\n' % (key, 17))
            elif label == 'A09':
                f.write('%s %d\n' % (key, 18))
            elif label == 'A10':
                f.write('%s %d\n' % (key, 19))
            elif label == 'A11':
                f.write('%s %d\n' % (key, 20))
            elif label == 'A12':
                f.write('%s %d\n' % (key, 21))
            elif label == 'A13':
                f.write('%s %d\n' % (key, 22))
            elif label == 'A14':
                f.write('%s %d\n' % (key, 23))
            elif label == 'A15':
                f.write('%s %d\n' % (key, 24))
            elif label == 'A16':
                f.write('%s %d\n' % (key, 25))
            elif label == 'A17':
                f.write('%s %d\n' % (key, 26))
            elif label == 'A18':
                f.write('%s %d\n' % (key, 27))
            elif label == 'A19':
                f.write('%s %d\n' % (key, 28))
            else:
                raise NameError('Unknown attack type: %s.' %(label))

def convert_la(scp_file, systemID_file, out_file):
    ''' multi-class classification for LA: SS_1, SS_2, SS_4, US_1, VC_1, VC_4 --> 7 classes
        (bonafide: 0), (SS_1: 1), (SS_2: 2), (SS_4: 3), (US_1: 4), (VC_1: 5), (VC_4: 6)
    '''
    with open(scp_file) as f:
        temp = f.readlines()
    key_list = [x.strip().split()[0] for x in temp]

    with open(systemID_file) as f:
        temp = f.readlines()
    utt2ID = {x.strip().split()[0]:x.strip().split()[1] for x in temp}

    with open(out_file, 'w') as f:
        for key in key_list:
            curr_utt = ''.join(key.split('-')[0])
            label = utt2ID[curr_utt] 
            if label == '-':
                f.write('%s %d\n' % (key, 0))
            elif label == 'A01':
                f.write('%s %d\n' % (key, 1))
            elif label == 'A02':
                f.write('%s %d\n' % (key, 2))
            elif label == 'A03':
                f.write('%s %d\n' % (key, 3))
            elif label == 'A04':
                f.write('%s %d\n' % (key, 4))
            elif label == 'A05':
                f.write('%s %d\n' % (key, 5))
            elif label == 'A06':
                f.write('%s %d\n' % (key, 6))
            elif label == 'A07':
                f.write('%s %d\n' % (key, 7))
            elif label == 'A08':
                f.write('%s %d\n' % (key, 8))
            elif label == 'A09':
                f.write('%s %d\n' % (key, 9))
            elif label == 'A10':
                f.write('%s %d\n' % (key, 10))
            elif label == 'A11':
                f.write('%s %d\n' % (key, 11))
            elif label == 'A12':
                f.write('%s %d\n' % (key, 12))
            elif label == 'A13':
                f.write('%s %d\n' % (key, 13))
            elif label == 'A14':
                f.write('%s %d\n' % (key, 14))
            elif label == 'A15':
                f.write('%s %d\n' % (key, 15))
            elif label == 'A16':
                f.write('%s %d\n' % (key, 16))
            elif label == 'A17':
                f.write('%s %d\n' % (key, 17))
            elif label == 'A18':
                f.write('%s %d\n' % (key, 18))
            elif label == 'A19':
                f.write('%s %d\n' % (key, 19))
            else:
                raise NameError('Unknown attack type: %s.' %(label))


def convert_pa(scp_file, systemID_file, out_file):
    ''' multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
        (bonafide: 0), (AA: 1), (AB: 2), (AC: 3), (BA: 4), (BB: 5), (BC: 6),
        (CA: 7), (CB: 8), (CC: 9)
    '''
    with open(scp_file) as f:
        temp = f.readlines()
    key_list = [x.strip().split()[0] for x in temp]

    with open(systemID_file) as f:
        temp = f.readlines()
    utt2ID = {x.strip().split()[0]:x.strip().split()[1] for x in temp}

    with open(out_file, 'w') as f:
        for key in key_list:
            # modified by Xu Li
            curr_utt = ''.join(key.split('-')[0])
            label = utt2ID[curr_utt] 
            if label == '-':
                f.write('%s %d\n' % (key, 0))
            elif label == 'AA':
                f.write('%s %d\n' % (key, 1))
            elif label == 'AB':
                f.write('%s %d\n' % (key, 2))
            elif label == 'AC':
                f.write('%s %d\n' % (key, 3))
            elif label == 'BA':
                f.write('%s %d\n' % (key, 4))
            elif label == 'BB':
                f.write('%s %d\n' % (key, 5))
            elif label == 'BC':
                f.write('%s %d\n' % (key, 6))
            elif label == 'CA':
                f.write('%s %d\n' % (key, 7))
            elif label == 'CB':
                f.write('%s %d\n' % (key, 8))
            elif label == 'CC':
                f.write('%s %d\n' % (key, 9))
            else:
                raise NameError('Unknown attack type: %s.' %(label))




if __name__ == '__main__':
    option = argparse.ArgumentParser()
    option.add_argument('--scp-file', type=str, default='data/spec/PA_train/feats_slicing.scp')
    option.add_argument('--sysID-file', type=str, default='data/spec/PA_train/utt2systemID')
    option.add_argument('--out-file', type=str, default='data/spec/PA_train/utt2index')
    option.add_argument('--access-type', type=str, default='PA')
    opt = option.parse_args()

    if opt.access_type == 'PA':
       convert_pa(opt.scp_file, opt.sysID_file, opt.out_file)
    elif opt.access_type == 'LA':
       convert_la(opt.scp_file, opt.sysID_file, opt.out_file)
    else:
       raise NameError('unknown access type: %s' %(opt.access_type))

