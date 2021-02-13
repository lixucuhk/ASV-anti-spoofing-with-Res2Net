import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from random import randint

# PyTorch implementation of Focal Loss
# source: https://github.com/clcarwin/focal_loss_pytorch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class TripletLoss(nn.Module):
	def __init__(self, dst_tp='l2', mrg=0.5, epsilon=1e-20): # dst_tp (distance_type): angular, l2; mrg (margin)
		super(TripletLoss, self).__init__()
		
		self.dst_tp = dst_tp
		self.mrg = mrg
		self.epsilon = epsilon
	
	def L2_dst(self, hd1, hd2): # hd1, hd2: N*D
		hd_dst = hd1-hd2
		return torch.sum(hd_dst*hd_dst, dim=-1) # N

	def angular_dst(self, hd1, hd2): # hd1, hd2: N*D
		hd1_norm = torch.sqrt(torch.sum(hd1*hd1, dim=-1)+self.epsilon)
		hd2_norm = torch.sqrt(torch.sum(hd2*hd2, dim=-1)+self.epsilon)

		cross_norm = torch.abs(torch.sum(hd1*hd2, dim=-1))

		return 1-cross_norm/(hd1_norm*hd2_norm)

	def forward(self, triplet_tuples): # tensor list of [a, p, n]: N*3*D
		anc = triplet_tuples[:,0,:]
		pos = triplet_tuples[:,1,:]
		neg = triplet_tuples[:,2,:]
		num_tp = len(anc)

		if self.dst_tp == 'l2':
			pos_dst = self.L2_dst(anc, pos)
			neg_dst = self.L2_dst(anc, neg)
			# print('pos avg dst: %f' %(torch.sum(pos_dst, dim=-1)/num_tp))
			# print('neg avg dst: %f' %(torch.sum(neg_dst, dim=-1)/num_tp))

			return torch.sum(torch.max(pos_dst-neg_dst+self.mrg, torch.tensor(0., device=device).expand_as(pos_dst)), dim=-1)/num_tp

		if self.dst_tp == 'angular':
			pos_dst = self.angular_dst(anc, pos)
			neg_dst = self.angular_dst(anc, neg)
			# print('pos avg dst: %f' %(torch.sum(pos_dst, dim=-1)/num_tp))
			# print('neg avg dst: %f' %(torch.sum(neg_dst, dim=-1)/num_tp))

			return torch.sum(torch.max(pos_dst-neg_dst+self.mrg, torch.tensor(0., device=device).expand_as(pos_dst)), dim=-1)/num_tp


class Triplet_Sampling(nn.Module):
	def __init__(self, dst_tp='l2', smp_mthd='rnd_anch'): ## dst_tp(distance_type): l2 or angular; smp_mthd(sampling_method): rnd_anch or genu_anch
		super(Triplet_Sampling, self).__init__()

		self.dst_tp = dst_tp
		self.smp_mthd = smp_mthd

	def mat_dst(self, hd1, hd2): # hd1, hd2: N*D
		if self.dst_tp == 'l2':
			return torch.sum(hd1*hd1, dim=-1).unsqueeze(0).t()-2*torch.matmul(hd1, hd2.t())+torch.sum(hd2*hd2, dim=-1).unsqueeze(0)  # N*N

		if self.dst_tp == 'angular':
			inner_p = torch.abs(torch.matmul(hd1, hd2.t())) # N*N
			outter_p = torch.matmul(torch.sqrt(torch.sum(hd1*hd1, dim=-1)+1e-20).unsqueeze(0).t(), torch.sqrt(torch.sum(hd2*hd2, dim=-1)+1e-20).unsqueeze(0)) # N*N

			return 1-inner_p/outter_p  # N*N

		else: raise NameError
	
	def argmax_min(self, dst_vec, tgt_list, opt): # dst_vec: dst_mat[index], tgt_list: genu_list or spoof_list, opt: 'max' or 'min'
		if opt == 'max':
			max_index = tgt_list[0]
			for i in range(len(dst_vec)):
				if i in tgt_list and dst_vec[i] >= dst_vec[max_index]:
					max_index = i
			return max_index
		else:
			min_index = tgt_list[0]
			for i in range(len(dst_vec)):
				if i in tgt_list and dst_vec[i] <= dst_vec[min_index]:
					min_index = i
			return min_index

	def forward(self, batch_hd, batch_tgt, margin): ## batch_hd: B*D, batch_tgt: B
		dst_mat = self.mat_dst(batch_hd, batch_hd)

		genu_list = [i for i in range(len(batch_tgt)) if batch_tgt[i] == 0]
		spoof_list = [i for i in range(len(batch_tgt)) if batch_tgt[i] != 0]

		if self.smp_mthd == 'rnd_anch':
			anch_list = [i for i in range(len(batch_tgt))]
		else: anch_list = genu_list
		
		if self.smp_mthd == 'rnd_anch':
			assert (len(genu_list)>1) and (len(spoof_list)>1), 'not enough samples to form triplet tuples.'
		else:
			assert len(genu_list)>1 and (len(spoof_list)>0), 'not enough samples to form triplet tuples.'

		pos_list = []
		neg_list = []

		for i in range(len(anch_list)):
			index = anch_list[i]
			if batch_tgt[index] == 0:
				pos_pool = genu_list
				neg_pool = spoof_list
			else:
				pos_pool = spoof_list
				neg_pool = genu_list

			pos_idx = randint(0, len(pos_pool)-1)
			while pos_idx == index:
				pos_idx = randint(0, len(pos_pool)-1)
			dst_pos = dst_mat[index][pos_idx]
			count = 0
			mindst = float('Inf')
			min_neg_idx = -1
			while count < 150:
				neg_idx = randint(0, len(neg_pool)-1)
				dst_neg = dst_mat[index][neg_idx]
				if dst_pos < dst_neg and dst_pos-dst_neg+margin > 0:
					min_neg_idx = neg_idx
					break
				elif dst_neg < mindst:
					mindst = dst_neg
					min_neg_idx = neg_idx
				count += 1
			pos_list.append(pos_idx)
			neg_list.append(min_neg_idx)
				
	#	# print(dst_mat)
	#	for i in range(len(anch_list)):
	#		index = anch_list[i]
	#		if batch_tgt[index] == 0:
	#			pos_idx = self.argmax_min(dst_mat[index], genu_list, 'max')
	#			neg_idx = self.argmax_min(dst_mat[index], spoof_list, 'min')
	#		else:
	#			pos_idx = self.argmax_min(dst_mat[index], spoof_list, 'max')
	#			neg_idx = self.argmax_min(dst_mat[index], genu_list, 'min')

	#		pos_list.append(pos_idx)
	#		neg_list.append(neg_idx)
		
		# print([anch_list, pos_list, neg_list])
		return torch.stack([batch_hd[anch_list], batch_hd[pos_list], batch_hd[neg_list]]).transpose(0, 1)

		

if __name__ == '__main__':
	# test Triplet Loss
	'''
	triplet_tps = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]], \
                                    [[1., 2.], [2., 3.], [1., 2.]]])

	print(triplet_tps)
	Triplet = TripletLoss(dst_tp='l2', mrg=0.5)
	print('L2 distance')
	print(Triplet(triplet_tps))

	Triplet = TripletLoss(dst_tp='angular', mrg=0.5)
	print('Angular distance')
	print(Triplet(triplet_tps))
	'''

	# test Triplet Sampling
	batch_hd = torch.tensor([[1., 1., 2.], [2., 2., 2.], [1., 1., 2.], [2., 3., 4.]])
	batch_tgt = [0, 0, 1, 1]
	dst_type = 'angular'
	smp_mthd = 'genu_anch'
	
	print(batch_hd)
	print(batch_tgt)
	print(dst_type)
	print(smp_mthd)

	triplet_sampling = Triplet_Sampling(dst_type, smp_mthd)
	tuples = triplet_sampling(batch_hd, batch_tgt)

	print(tuples)

