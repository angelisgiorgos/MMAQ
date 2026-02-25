# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
from typing import List, Tuple, Dict, Any
import torch
import torchmetrics
from torch import nn, Tensor
from models.backbones.model import S2Backbone, S5Backbone
from models.backbones.decur_projector import DeCURProjector
from models.backbones.aqr_projector import AQRProjector
from losses.decur_loss import MultiDeCURLoss, MultiModalDecur
from flash.core.optimizers import LARS
from pytorch_lightning import LightningModule
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters
from losses.mixup import MixUPLoss
from models.backbones.tabularnets import TabularAttention, DANet
from lightly.utils.dist import GatherLayer


class MMAQ(LightningModule):
    def __init__(self, args, data_stats):
        super().__init__()
        self.args = args

        # self.gather = GatherLayer()

        self.data_stats = data_stats

        self.encoder1 = S2Backbone(args)
        self.encoder2 = S5Backbone(args)
        sizes = [self.args.imaging_embedding] + list(map(int, '512-512-512'.split('-')))
        self.pooled_dim = 2048
        if self.args.tabular_net == "initial":
            self.tabular1 = TabularAttention(self.args)
        elif self.args.tabular_net == "danet":
            self.tabular1 = DANet()
        # self.tabular1 = TabularNet(self.args)
        self.projector1 = DeCURProjector(sizes)
        self.projector2 = DeCURProjector(sizes)
        tab_sizes = [self.args.tabular_net_features] + list(map(int, '512-512-512'.split('-')))
        self.projector_tab = DeCURProjector(tab_sizes)
        # self.projector1 = AQRProjector(args)
        # self.projector2 = AQRProjector(args)
            

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        # self.criterion = MultiDeCURLoss(args, self.bn)
        self.criterion = MultiModalDecur(args, self.bn)

        self.mixup = MixUPLoss(args, self.bn, 5.0, 0.005)

        self.online_regressor = OnlineLinearRegressor(self.data_stats, feature_dim =self.pooled_dim*2 )

        self.val_preds = []
        self.val_targets = []

        self.mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)

        self.mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)

        self.r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        
        self.mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)



    def forward(self, image: Tensor, tabular: Tensor = None, tab='img_both') -> Tensor:
        features1 = self.encoder1(image)
        if tab == "img1":
            emb = features1
        elif tab == 'img_both':
            features2 = self.encoder2(image)
            emb = torch.cat([features1, features2], axis=1)
        elif tab == "tab":
            features2 = self.tabular1(tabular)
            emb = torch.cat([features1, features2], axis=1)
        else:
            features2 = self.encoder2(image)
            features3 = self.tabular1(tabular)
            emb = torch.cat([features1, features2, features3], axis=1)
        return emb


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch

        features1 = self.encoder1(im_views[0])
        features1_hat = self.encoder1(im_views[1])
        features2 = self.encoder2(im_views[0])
        features2_hat = self.encoder2(im_views[1])
        features3 = self.tabular1(tab_views[0])
        features3_hat = self.tabular1(tab_views[1])

        z_1 = self.projector1(features1)
        z_1_1 = self.projector1(features1_hat)

        z_2 = self.projector2(features2)
        z_2_2 = self.projector2(features2_hat)

        z3 = self.projector_tab(features3)
        z_3_3 = self.projector_tab(features3_hat)

        loss_init, on_diag12_c = self.criterion(z_1, z_1_1, z_2, z_2_2, z3, z_3_3)

        emb = torch.cat([features1.detach(), features2.detach()], axis=1)
    
        ########MIXUP########
        # index = torch.randperm(self.args.batch_size)
        # alpha = np.random.beta(1.0, 1.0)
        # tab_m = tab_views[1][index, :]
        # pos_m_tab = alpha * tab_views[0] + (1 - alpha) * tab_m[:tab_views[0].shape[0], :]

        # pos_m_im = alpha * im_views[0] + (1 - alpha) * im_views[1][index, :]

        # featuresm3 = features3 = self.tabular1(pos_m_tab)
        # z_m3 = self.projector_tab(featuresm3)

        # featuresm1 = self.encoder1(pos_m_im[0])
        # z_m1 = self.projector1(featuresm1)

        # featuresm2 = self.encoder2(pos_m_im[0])
        # z_m2 = self.projector1(featuresm2)
        # z_2m = z_2[index, :]
        # z_2m= z_2m[:z_2.shape[0], :]
        # loss_mixup = self.mixup(z_1, z_2, z_2m, z_m3, alpha)

        # loss = loss_init  + loss_mixup
        loss = loss_init
        
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )
        regr_loss, regr_log = self.online_regressor.training_step(
            (emb, targets), batch_idx
        )

        # regr_log.update({"train_init": loss_init, "train_mixup": loss_mixup})
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-4*regr_loss
    
    
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:

        """
        Validate contrastive model
        """
        im_views, tab_views, targets, original_im = batch

        features = self.forward(im_views[0], tab_views[0])

        regr_loss, preds, targets = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.val_preds.append(preds.detach())
        self.val_targets.append(targets)
        return regr_loss


    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(outputs).mean()
        preds = torch.cat([pred.to(self.mae.device) for pred in self.val_preds], dim=0)
        targets = torch.cat([target.to(self.mae.device) for target in self.val_targets], dim=0)

        # preds = self.gather(preds)
        # targets = self.gather(targets)

        # print(preds.device, targets.device)


        metrics = {
            "mae": self.mae(preds, targets),
            "mape": self.mape(preds, targets),
            "r2": self.r2(preds, targets),
            "mse": self.mse(preds, targets)
        }

        self.log("val_epoch_loss", val_loss, prog_bar=True)
        self.log_dict({f"val_{k}": acc for k, acc in metrics.items()}, prog_bar=True, sync_dist=True)
    

    def configure_optimizers(self):
        lr_factor = self.args.batch_size * self.trainer.world_size / 256

        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.encoder1, self.encoder2,  self.projector1, self.projector2, self.tabular1, self.projector_tab]
        )
        optimizer = LARS(
            [
                {"name": "decur", "params": params},
                {
                    "name": "decur_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                    "lr": 0.0048 * lr_factor,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.2 * lr_factor,
            momentum=0.9,
            weight_decay=1.5e-6,
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]