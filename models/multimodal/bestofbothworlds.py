from typing import List, Tuple, Dict
import torch
from torch import Tensor
from models.multimodal.base import BaseMultimodalModel
from losses.clip_loss import CLIPLoss
from models.backbones.model import S2Backbone
from lightly.models.modules import SimCLRProjectionHead
from models.backbones.model import TabularNet
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from lightly.models.utils import get_weight_decay_parameters


class BestofBothWorlds(BaseMultimodalModel):
    """Implementation of the paper "Multimodal Contrastive Learning with Tabular and Imaging Data" for Air Quality Data

    Args:
        LightningModule (_type_): _description_
    """    
    def __init__(self, args, data_stats) -> None:
        super().__init__(args, data_stats)
        self.online_regressor = OnlineLinearRegressor(self.data_stats, feature_dim=self.pooled_dim)

    def _build_backbones(self):
        self.backbone = S2Backbone(self.args)
        self.encoder_tabular = TabularNet(self.args)
        self.pooled_dim = 2048

    def _build_projectors(self):
        self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.args.embedding_dim, self.args.projection_dim)
        self.projector_tabular = SimCLRProjectionHead(self.args.embedding_dim, self.args.embedding_dim, self.args.projection_dim)

    def _build_losses(self):
        self.criterion = CLIPLoss(temperature=0.1, lambda_0=0.5)


    def forward_imaging(self, x: Tensor) -> Tensor:
        features = self.backbone(x).flatten(start_dim=1)
        projections = self.projector_imaging(features)
        return projections, features

    def forward_tabular(self, x: Tensor) -> Tensor:
        features = self.encoder_tabular(x).flatten(start_dim=1)
        projections = self.projector_tabular(features)
        return projections, features

    def forward(self, images: Tensor, tabular: Tensor=None) -> Tensor:
        features1 = self.backbone(images).flatten(start_dim=1)
        # features2= self.encoder_tabular(tabular.float()).flatten(start_dim=1)
        return features1

        
    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch
        
        z0, embeddings_0 = self.forward_imaging(im_views[1])
        z1, embeddings_1 = self.forward_tabular(tab_views[1])
        
        loss, logits, labels = self.criterion(z0, z1, targets)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )
        regr_loss, regr_log = self.online_regressor.training_step(
            (embeddings_0.detach(), targets.float()), batch_idx)
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-3*regr_loss
    
    
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:
        """
        Validate contrastive model
        """
        im_views, tab_views, targets, original_im = batch
        features= self.forward(im_views[0])
        regr_loss, regr_log = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(regr_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return regr_loss


    def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.args.dataset_length*self.args.cosine_anneal_mult), eta_min=0, last_epoch=-1)
        elif self.args.scheduler == 'anneal':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.warmup_epochs, max_epochs = self.args.max_epochs)
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')
        
        return scheduler
    
    
    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.encoder_tabular,  self.projector_imaging, self.projector_tabular]
        )
        optimizer = torch.optim.Adam(
        [ {"name": "decur", "params": params},
                {
                    "name": "mmc;_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.lr,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
        ])
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )