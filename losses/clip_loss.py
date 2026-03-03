from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from losses.uncertainty import UncertaintyRevised


class CLIPLoss(torch.nn.Module):
    """
    Loss function for multimodal contrastive learning based off of the CLIP paper.

    Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
    similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
    Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
    """
    def __init__(self, 
                temperature: float,
                lambda_0: float = 0.5) -> None:
        super(CLIPLoss, self).__init__()

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if lambda_0 > 1 or lambda_0 < 0:
            raise ValueError('lambda_0 must be a float between 0 and 1.')
        self.lambda_0 = lambda_0
        self.lambda_1 = 1-lambda_0

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
        # normalize the embedding onto the unit hypersphere
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        #logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
        logits = torch.matmul(out0, out1.T) / self.temperature
        labels = torch.arange(len(out0), device=out0.device)
        
        loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
        loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
        loss = loss_0 + loss_1
    
        return loss, logits, labels
    
    

class MMNTXentLoss(torch.nn.Module):
    def __init__(self, args):
        """Compute loss for model.
        temperature: a `floating` number for temperature scaling.
        weights: a weighting number or vector.
        """
        super(MMNTXentLoss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = args.temperature
        self.alpha_weight = args.alpha_weight
        self.device = args.device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=True):
        temperature = self.temperature
        alpha = self.alpha_weight

        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(
            torch.arange(start=0, end=batch_size, dtype=torch.int64),
            num_classes=batch_size,
        ).float()
        labels = labels.to(self.device)

        # Different from Image-Image contrastive learning
        # In the case of Image-Gen contrastive learning we do not compute the intra-modal similarity
        # masks = F.one_hot(
        #     torch.arange(start=0, end=batch_size, dtype=torch.int64),
        #     num_classes=batch_size,
        # )
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = (
            torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        )
        logits_ba = (
            torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
        )

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)
        
        loss = alpha * loss_a + (1 - alpha) * loss_b

        return loss, logits_ab, labels



class MultimodalJointLoss(nn.Module):

    def __init__(self, temperature=0.1,
                 lambda_align=1.0,
                 lambda_var=1.0,
                 lambda_cov=0.1,
                 uncertainty=False):

        super().__init__()

        self.temperature = temperature
        self.lambda_align = lambda_align
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.uncertainty = uncertainty

        if self.uncertainty:
            self.uncert = UncertaintyRevised(num_losses=2)

    # ----------------------------------------------------------
    # InfoNCE
    # ----------------------------------------------------------
    def info_nce(self, z1, z2):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        return F.cross_entropy(logits, labels)

    # ----------------------------------------------------------
    # VICReg Regularization
    # ----------------------------------------------------------
    def variance_loss(self, z):

        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))

    def covariance_loss(self, z):

        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.size(0) - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum() / z.size(1)

    # ----------------------------------------------------------
    # Full Loss
    # ----------------------------------------------------------
    def forward(self, outputs, targets=None):

        f_s2 = outputs["s2"]
        f_s5 = outputs["s5"]
        f_txt = outputs["txt"]
        f_fused = outputs["fused"]

        # Multi-view alignment
        loss_s2_s5 = self.info_nce(f_s2, f_s5)
        loss_s2_txt = self.info_nce(f_s2, f_txt)
        loss_s5_txt = self.info_nce(f_s5, f_txt)

        align_loss = loss_s2_s5 + loss_s2_txt + loss_s5_txt

        # VICReg regularization
        var_loss = (
            self.variance_loss(f_fused)
        )

        cov_loss = (
            self.covariance_loss(f_fused)
        )

        vicreg_loss = (
            self.lambda_var * var_loss +
            self.lambda_cov * cov_loss
        )
        

        if not self.uncertainty:
            total = (
                self.lambda_align * align_loss +
                vicreg_loss
            )
        else:
            total = self.uncert(
                align_loss,
                vicreg_loss
            )

        return total