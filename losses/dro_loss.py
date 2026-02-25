import torch
import torch.nn as nn
import numpy as np
import math


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class DRO_Loss(nn.Module):
    def __init__(self, args,  N=1.2e6):
        super(DRO_Loss, self).__init__()
        self.temperature = args.temperature
        self.tau_plus = args.tau_plus
        self.batch_size = args.batch_size
        self.beta = args.beta
        self.estimator = args.estimator


    def forward(self, out_1, out_2):
        device = out_1.device
        # batch_size = out_1.size(0)
        if self.estimator == "easy":
            # neg score
            out = torch.cat([out_1, out_2], dim=0)
            neg_ = torch.mm(out, out.t().contiguous())
            neg = torch.exp(neg_ / self.temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(self.batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * self.batch_size, -1)
            # pos score
            pos_ = torch.sum(out_1 * out_2, dim=-1)
            pos = torch.exp(pos_ / self.temperature)
            pos = torch.cat([pos, pos], dim=0)
            Ng = neg.sum(dim=-1)

            loss = (- torch.log(pos / (pos + Ng) )).mean()

            return loss

        elif self.estimator == "HCL":
            # neg score
            batch_size = out_1.size(0)
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * batch_size, -1)
            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)
            
            N = batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))

            loss = (- torch.log(pos / (pos + Ng) )).mean()

            return loss

        elif self.estimator == "a_cl":
            representations = torch.cat([out_1, out_2], dim=0)
            similarity_matrix = self.similarity_function(representations, representations)
            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            # 2N positive pairs 
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
            # 2N * (2N - 2) negative samples. 
            # The i-th row corresponds to 2N - 1 negative samples for i-th sample.  
            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            dist_sqr = negatives - positives
            r_neg = 1 - negatives
            r_pos = 1 - positives
            w = r_neg.detach().pow(self.tau_plus)
            w = (-w / self.temperature).exp()
            w_Z = w.sum(dim=1, keepdim=True)
            w = w / (w_Z) 
            loss = (w * dist_sqr).sum(dim=1).mean()
            return loss

        elif self.estimator == "a_cl2":
            representations = torch.cat([out_1, out_2], dim=0)
            similarity_matrix = self.similarity_function(representations, representations)
            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            # 2N positive pairs 
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
            # 2N * (2N - 2) negative samples. 
            # The i-th row corresponds to 2N - 1 negative samples for i-th sample.  
            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            dist_sqr = negatives - positives
            r_neg = 1 - negatives
            r_pos = 1 - positives
            w = r_neg.detach().pow(self.tau_plus)
            w = (-w / self.temperature).exp()

            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            return loss, w

        elif self.estimator == "adnce":
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(self.batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * self.batch_size, -1)
            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)

            N = self.batch_size * 2 - 2
            mu = self.tau_plus
            sigma = self.beta
            weight = 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp( - (neg.log() * self.temperature - mu) ** 2 / (2 * math.pow(sigma, 2)))
            weight = weight / weight.mean(dim=-1, keepdim=True)
            # loss compute
            Ng = torch.sum(neg * weight.detach(), dim=1)
            loss = (- torch.log(pos / (pos + Ng) )).mean()
            return loss