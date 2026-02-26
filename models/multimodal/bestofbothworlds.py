from typing import List, Tuple, Dict
import torch
import torchmetrics
from torch import Tensor, nn
from models.multimodal.base import BaseMultimodalModel
from losses.clip_loss import CLIPLoss
from models.backbones.model import S2Backbone
from lightly.models.modules import SimCLRProjectionHead
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from lightly.models.utils import get_weight_decay_parameters


class TabularEncoder(nn.Module):
    """
    Main contrastive model used in SCARF. Consists of an encoder that takes the input and 
    creates an embedding of size {args.embedding_dim}.
    Also supports providing a checkpoint with trained weights to be loaded.
    """
    def __init__(self, args) -> None:
        super(TabularEncoder, self).__init__()
        self.args = args

        self.input_size = getattr(args, 'input_size', getattr(args, 'tabular_input', 8))
        self.encoder_num_layers = getattr(args, 'encoder_num_layers', getattr(args, 'n_layers_tabular', 3))
        self.embedding_dim = args.embedding_dim

        # Check if we are loading a pretrained model
        if getattr(args, 'checkpoint', None):
            loaded_chkpt = torch.load(args.checkpoint)
            original_args = loaded_chkpt['hyper_parameters']
            state_dict = loaded_chkpt['state_dict']
            self.input_size = original_args.get('input_size', getattr(original_args, 'tabular_input', 8))
            
            if 'encoder_tabular.encoder.1.running_mean' in state_dict.keys():
                encoder_name = 'encoder_tabular.encoder.'
                self.encoder = self.build_encoder(original_args)
            elif 'encoder_projector_tabular.encoder.2.running_mean' in state_dict.keys():
                encoder_name = 'encoder_projector_tabular.encoder.'
                self.encoder = self.build_encoder_bn_old(original_args)
            else:
                encoder_name = 'encoder_projector_tabular.encoder.'
                self.encoder = self.build_encoder_no_bn(original_args)

            # Split weights
            state_dict_encoder = {}
            for k in list(state_dict.keys()):
                if k.startswith(encoder_name):
                    state_dict_encoder[k[len(encoder_name):]] = state_dict[k]
                
            _ = self.encoder.load_state_dict(state_dict_encoder, strict=True)

            # Freeze if needed
            if getattr(args, 'finetune_strategy', '') == 'frozen':
                for _, param in self.encoder.named_parameters():
                    param.requires_grad = False
                parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
                assert len(parameters)==0
        else:
            # Build architecture
            self.encoder = self.build_encoder(args)
            self.encoder.apply(self.init_weights)

    def build_encoder(self, args) -> nn.Sequential:
        modules = [nn.Linear(self.input_size, self.embedding_dim)]
        for _ in range(self.encoder_num_layers-1):
            modules.extend([nn.BatchNorm1d(self.embedding_dim), nn.ReLU(), nn.Linear(self.embedding_dim, self.embedding_dim)])
        return nn.Sequential(*modules)
    
    def build_encoder_no_bn(self, args) -> nn.Sequential:
        modules = [nn.Linear(self.input_size, self.embedding_dim)]
        for _ in range(self.encoder_num_layers-1):
            modules.extend([nn.ReLU(), nn.Linear(self.embedding_dim, self.embedding_dim)])
        return nn.Sequential(*modules)

    def build_encoder_bn_old(self, args) -> nn.Sequential:
        modules = [nn.Linear(self.input_size, self.embedding_dim)]
        for _ in range(self.encoder_num_layers-1):
            modules.extend([nn.ReLU(), nn.BatchNorm1d(self.embedding_dim), nn.Linear(self.embedding_dim, self.embedding_dim)])
        return nn.Sequential(*modules)

    def init_weights(self, m: nn.Module, init_gain = 0.02) -> None:
        """
        Initializes weights according to desired strategy
        """
        if isinstance(m, nn.Linear):
            init_strat = getattr(self.args, 'init_strat', 'normal')
            if init_strat == 'normal':
                nn.init.normal_(m.weight.data, 0, 0.001)
            elif init_strat == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_strat == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_strat == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input through encoder and projector. 
        Output is ready for loss calculation.
        """
        x = self.encoder(x)
        return x


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
        self.encoder_tabular = TabularEncoder(self.args)
        self.pooled_dim = 2048

    def _build_projectors(self):
        self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.args.embedding_dim, self.args.projection_dim)
        self.projector_tabular = SimCLRProjectionHead(self.args.embedding_dim, self.args.embedding_dim, self.args.projection_dim)

    def _build_losses(self):
        self.criterion = CLIPLoss(temperature=0.1, lambda_0=0.5)

    def _build_metrics(self):
        self.val_mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)
        self.val_mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)
        self.val_r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        self.val_mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)


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
    
    
    def validation_step(
        self, 
        batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:
        """
        Validate contrastive model
        """
        im_views, tab_views, targets, original_im = batch
        features= self.forward(im_views[0])
        regr_loss, preds, targets = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )

        # Update metrics (TorchMetrics handles accumulation internally)
        self.val_mae.update(preds, targets)
        self.val_mape.update(preds, targets)
        self.val_r2.update(preds, targets)
        self.val_mse.update(preds, targets)

        # Log step loss (proper DDP sync)
        self.log(
            "val_step_loss",
            regr_loss,
            prog_bar=False,
            sync_dist=True,
            batch_size=targets.size(0)
        )

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        # Compute metrics (aggregated across devices)
        metrics = {
            "val_mae": self.val_mae.compute(),
            "val_mape": self.val_mape.compute(),
            "val_r2": self.val_r2.compute(),
            "val_mse": self.val_mse.compute(),
        }

        # Log once (Lightning handles sync)
        self.log_dict(
            metrics,
            prog_bar=True,
            sync_dist=True
        )

        # Reset metrics for next epoch
        self.val_mae.reset()
        self.val_mape.reset()
        self.val_r2.reset()
        self.val_mse.reset()


    def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(self.args.dataset_length*self.args.cosine_anneal_mult), eta_min=0, last_epoch=-1)
        elif self.args.scheduler == 'anneal':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.args.warmup_epochs, max_epochs = self.args.max_epochs)
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')
        
        return scheduler
    
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        # Handle legacy weights from older TabularNet architectures
        # The old tabular attention had layers like linear1, bn1, linear (attention), bn2, output
        # The new SCARF TabularEncoder has nn.Sequential blocks named net.0, net.2, etc. (if no bn)
        
        legacy_keys = list(state_dict.keys())
        current_state = self.state_dict()
        
        for k in legacy_keys:
            if "encoder_tabular." in k:
                # We attempt to dynamically map or drop old keys if they do not match the new encoder
                # In SCARF, the new TabularEncoder builds a sequential net named 'encoder'
                mapped_key = None
                
                if "linear1.weight" in k:
                    mapped_key = "encoder_tabular.encoder.0.weight"
                elif "linear1.bias" in k:
                    mapped_key = "encoder_tabular.encoder.0.bias"
                elif "bn1.weight" in k:
                    mapped_key = "encoder_tabular.encoder.1.weight"
                elif "bn1.bias" in k:
                    mapped_key = "encoder_tabular.encoder.1.bias"
                elif "bn1.running_mean" in k:
                    mapped_key = "encoder_tabular.encoder.1.running_mean"
                elif "bn1.running_var" in k:
                    mapped_key = "encoder_tabular.encoder.1.running_var"
                elif "bn1.num_batches_tracked" in k:
                    mapped_key = "encoder_tabular.encoder.1.num_batches_tracked"
                elif "linear.weight" in k:
                    mapped_key = "encoder_tabular.encoder.2.weight"
                elif "linear.bias" in k:
                    mapped_key = "encoder_tabular.encoder.2.bias"
                elif "bn2.weight" in k:
                    mapped_key = "encoder_tabular.encoder.3.weight"
                elif "bn2.bias" in k:
                    mapped_key = "encoder_tabular.encoder.3.bias"
                elif "bn2.running_mean" in k:
                    mapped_key = "encoder_tabular.encoder.3.running_mean"
                elif "bn2.running_var" in k:
                    mapped_key = "encoder_tabular.encoder.3.running_var"
                elif "bn2.num_batches_tracked" in k:
                    mapped_key = "encoder_tabular.encoder.3.num_batches_tracked"
                elif "output.weight" in k:
                    if 'encoder_tabular.encoder.4.weight' in current_state:
                         mapped_key = "encoder_tabular.encoder.4.weight"
                elif "output.bias" in k:
                    if 'encoder_tabular.encoder.4.bias' in current_state:
                         mapped_key = "encoder_tabular.encoder.4.bias"

                if mapped_key:
                    # Check if sizes match
                    if mapped_key in current_state and current_state[mapped_key].shape == state_dict[k].shape:
                        state_dict[mapped_key] = state_dict.pop(k)
                    else:
                        # Drop the mismatched key
                        state_dict.pop(k)
                else:
                    # Drop unmatched legacy tabular keys
                    if k in state_dict:
                        state_dict.pop(k)
                        
        return super().load_state_dict(state_dict, strict=True)

    
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