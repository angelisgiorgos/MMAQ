from typing import List, Tuple, Dict
import torch
from models.multimodal.pretraining import Pretraining
from losses import CLIPLoss, MMNTXentLoss, DCL


class MultimodalContrastiveSimCLR(Pretraining):
    """
    Lightning module for multimodal SSL.
    """
    def __init__(self, args):
        super().__init__(args)
        # Imaging
        self.initialize_imaging_encoder_and_projector()

        # Tabular
        self.initialize_tabular_encoder_and_projector()
        
        self.criterion_val = DCL()

        self.criterion_train = self.criterion_val

        self.initialize_regressor_and_metrics()

        print(f'Tabular model, multimodal: {self.encoder_tabular}\n{self.projector_tabular}')
        print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projector_imaging}')


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        im_views, tab_views, y, _ = batch
        # Augmented views
        z0, embeddings_0 = self.forward_imaging(im_views[1])
        z1, embeddings_1 = self.forward_tabular(tab_views[1])
        loss = self.criterion_train(z0, z1)
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)
        # if len(im_views[0]) == self.args.batch_size:
        #     self.calc_and_log_train_embedding_metrics(logits=logits, labels=labels, modality='multimodal')

        return {'loss':loss, 'embeddings':  torch.cat([embeddings_0, embeddings_1], axis=1), 'labels': y}


    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        """
        Validate contrastive model
        """
        im_views, tab_views, y, original_im = batch
    
        # Unaugmented views
        z0, embeddings_0 = self.forward_imaging(original_im)
        z1, embeddings_1 = self.forward_tabular(tab_views[0])
        loss = self.criterion_val(z0, z1)

        self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)

        # if len(im_views[0])==self.args.batch_size:
        #     self.calc_and_log_val_embedding_metrics(logits=logits, labels=labels, modality='multimodal')

        return {'sample_augmentation': im_views[1], 'embeddings': torch.cat([embeddings_0, embeddings_1], axis=1), 'labels': y}

    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
        [
            {'params': self.encoder_imaging.parameters()}, 
            {'params': self.projector_imaging.parameters()},
            {'params': self.encoder_tabular.parameters()},
            {'params': self.projector_tabular.parameters()}
        ], lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )