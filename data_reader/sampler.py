import numpy as np
from torch.utils.data import Sampler
from data_reader.dataset_v1 import SpoofDatsetSystemID


class CustomSampler(Sampler):
  def __init__(self, data_source, shuffle):
    self.df = data_source
    self.shuffle = shuffle

  def getIndices(self):

    labels = [0, 1]
    digit_indices = []
    digit_indices.append(np.where(self.df.labels == 0)[0])
    digit_indices.append(np.where(self.df.labels != 0)[0])


    num_genu = len(digit_indices[0])
    num_spoofed = len(digit_indices[1])

    '''
    print('genu: %d, spoofed: %d' %(num_genu, num_spoofed))
    print(digit_indices[0].shape)
    '''

    if self.shuffle:
      for i in range(len(digit_indices)):
        np.random.shuffle(digit_indices[i])

    repetition = int(num_spoofed/num_genu)
    digit_indices[0] = np.tile(digit_indices[0], repetition)
    rest_part = num_spoofed%num_genu
    rest = digit_indices[0][:rest_part]
    # print(rest.shape)
    digit_indices[0] = np.concatenate((digit_indices[0], rest), axis=0)

    '''
    num_genu = len(digit_indices[0])
    num_spoofed = len(digit_indices[1])

    print('genu: %d, spoofed: %d' %(num_genu, num_spoofed))
    '''

    return digit_indices

    '''
    tensor = np.tile(mat,repetition)
    repetition = max_len % mat.shape[1]
    rest = mat[:,:repetition]
    tensor = np.hstack((tensor,rest))


    min_size = np.size(digit_indices[0])
    for i in range(1, len(digit_indices)):
      size = np.size(digit_indices[i])
      min_size = size if size < min_size else min_size
    return digit_indices, min_size
    '''

  def __iter__(self):
    digit_indices = self.getIndices()
    assert len(digit_indices[0]) == len(digit_indices[1]), 'The amount of genuine and spoofed audios does not match!'
    num_samples = len(digit_indices[0])
    indices = []
    for i in range(num_samples):
      indices += [digit_indices[n][i] for n in range(2)]
    return iter(indices)

  def __len__(self):
    digit_indices = self.getIndices()
    return len(digit_indices[0])+len(digit_indices[1])



if __name__ == '__main__':

  data_scp = 'feats/test_samples.scp'
  data_utt2index = 'utt2systemID/test_samples_utt2index8_spectensor1'

  data = SpoofDatsetSystemID(data_scp, data_utt2index, binary_class=False, leave_one_out=False)
  sampler = CustomSampler(data, shuffle=False)

  count = 0
  for i in sampler.__iter__():
    uttid, x, y = data[i]
    print(y)
    if count == 30: break
    count += 1

  
