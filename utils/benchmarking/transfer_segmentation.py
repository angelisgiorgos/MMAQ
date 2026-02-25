from typing import Any, Dict, List, Tuple, Union
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning.pytorch.loggers import WandbLogger
from lightly.utils.benchmarking import MetricCallback
from lightning.pytorch import Trainer
import os, sys
from models import build_ssl_model
from utils import create_logdir
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.optim import SGD, Optimizer, Adam
import lightning.pytorch as pl
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor



class SegmentationDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.model == "dino":
            in_channels_list = [768, 768, 768]
        else:
            in_channels_list = [512, 1024, 2048]
        self.uc0 = torch.nn.Conv2d(in_channels=in_channels_list[0], out_channels=128, kernel_size=1)
        self.uc1 = torch.nn.Conv2d(in_channels=in_channels_list[1], out_channels=256, kernel_size=1)
        self.uc2 = torch.nn.Conv2d(in_channels=in_channels_list[2], out_channels=512, kernel_size=1)

        self.up0 = torch.nn.Upsample(size=(29, 29), mode='bilinear', align_corners=True)
        self.up1 = torch.nn.Upsample(size=(29, 29), mode='bilinear', align_corners=True)
        self.up2 = torch.nn.Upsample(size=(29, 29), mode='bilinear', align_corners=True)

        self.cls0 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        self.cls1 = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.cls2 = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

        self.final_upsample = torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def forward(self, x1):
        B, F, _ = x1['0'].shape
        uc0 = self.uc0(x1['0'].view(B, F, 14, 14))
        up0 = self.up0(uc0)
        y0 = self.cls0(up0)
        y1 = self.cls1(self.up1(self.uc1(x1['1'].view(B, F, 14, 14))))
        y2 = self.cls2(self.up2(self.uc2(x1['2'].view(B, F, 14, 14))))

        # Upsample each output to the final size
        y0 = self.final_upsample(y0)
        y1 = self.final_upsample(y1)
        y2 = self.final_upsample(y2)

        # Sum up the results
        y = y0 + y1 + y2
        return y

class DinoFeatureExtractor(nn.Module):
    def __init__(self, vit_model):
        super(DinoFeatureExtractor, self).__init__()
        self.vit_model = vit_model
        nn.ModuleList([
            self.vit_model.blocks[5],
            self.vit_model.blocks[7],
            self.vit_model.blocks[11]
        ])
    
    def get_submodule(self, target):
        submodule = self.vit_model
        for attr in target.split("."):
            submodule = getattr(submodule, attr)
        return submodule
    
    def forward(self, x):
        B, _, _, _ = x.shape
        feature_outputs = {}
        x = self.vit_model._modules['patch_embed'](x)  # Patch embedding
        x = self.vit_model._modules['pos_drop'](x)  # Position drop
        for i, layer in enumerate(self.vit_model.blocks):
            x = layer(x)
            if i == 5:
                # feature_outputs["0"] = x.view(B, 14, 14, 768)
                feature_outputs["0"] = x.permute(0, 2, 1)
            elif i == 7:
                # feature_outputs["1"] = x.view(B, 14, 14, 768)
                feature_outputs["1"] = x.permute(0, 2, 1)
            elif i == 11:
                # feature_outputs["2"] = x.view(B, 14, 14, 768)
                feature_outputs["2"] = x.permute(0, 2, 1)

        return feature_outputs
            



class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """

    def __init__(self, args, backbone, batch_size_per_device, num_classes=1, freeze=False):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.freeze = freeze
        self.batch_size_per_device = batch_size_per_device
        self.save_hyperparameters()

        m1 = backbone
        #m2 = models.resnet50()
        # Extract 4 main layers
        if self.args.model == "dino":
            self.backbone_1 = DinoFeatureExtractor(vit_model=m1)

        else:
            return_nodes = {f'layer{k}': str(v)
                             for v, k in enumerate([2, 3, 4])}
            self.backbone_1 = create_feature_extractor(
                m1, return_nodes=return_nodes)

        self.decoder = SegmentationDecoder(args)


        # Setting compute_on_step = False to compute "part IoU"
        # This is because we want to compute the IoU on the entire dataset
        # at the end to account for rare classes, rather than within each batch
        self.train_iou = torchmetrics.classification.BinaryJaccardIndex(compute_on_step=False)
        self.val_iou = torchmetrics.classification.BinaryJaccardIndex(compute_on_step=False)
        self.test_iou = torchmetrics.classification.BinaryJaccardIndex(compute_on_step=False)

        self.train_accuracy = torchmetrics.classification.BinaryAccuracy(compute_on_step=False)
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy(compute_on_step=False)
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy(compute_on_step=False)


    def forward(self, batch):
        if self.freeze:
            with torch.no_grad():
                features = self.backbone_1(batch)
        else:
            features = self.backbone_1(batch)
        logits = self.decoder(features)
        return logits.float()


    def training_step(self, batch, batch_idx):
        inputs = batch["img"].float().to(self.device)
        if torch.isnan(inputs).any():
            print(f"NaN or Inf detected in loss at batch {batch_idx}")
        labels = batch["fpt"].float().to(self.device)
        logits = self.forward(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = (logits.sigmoid()>0.5).long()
        self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)
        return loss

    def on_train_epoch_end(self):
        self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())


    def validation_step(self, batch, batch_idx):
        inputs = batch["img"].float().to(self.device)
        labels = batch["fpt"].float().to(self.device)
        logits = self.forward(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = (logits.sigmoid()>0.5).long()
        self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)
        return loss


    def on_validation_epoch_end(self):
        self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())


    def test_step(self, batch, batch_idx):
        
        inputs = batch["img"].float().to(self.device)
        labels = batch["fpt"].float().to(self.device)
        logits = self.forward(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = (logits.sigmoid()>0.5).long()
        self.test_iou(preds, labels)
        self.test_accuracy(preds, labels)


    def test_epoch_end(self, outs):
        self.log("test_iou", self.test_iou.compute())
        self.log("test_accuracy", self.test_accuracy.compute())


    
    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.decoder.parameters())
        if not self.freeze:
            parameters += self.backbone_1.parameters()
        optimizer = Adam(
            parameters,
            lr=self.args.lr * self.batch_size_per_device * self.trainer.world_size / 256,
            # momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]



def tf_segmentation(args,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    data_stats=None,
    num_classes=1) -> None:
    print("Running Transfer Learning Segmentation...")

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=base_dir,
        offline=args.offline,
        config=args
    )
    

    # Create logdir based on WandB run name
    logdir = create_logdir(args.datatype, wandb_logger)

    # Train linear classifier.
    metric_callback = MetricCallback()

    model = build_ssl_model(args, data_stats)
    if args.ckpt_path is None:
        ckpt_path = os.path.join("./checkpoints", args.model + ".ckpt")
    else:
        ckpt_path = args.ckpt_path
    model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)

    if hasattr(torch, "compile"):
        # Compile model if PyTorch supports it.
        model = torch.compile(model)

    if args.model == "decur":
        backbone = model.encoder1.backbone_S2
    elif args.model == "mmaq":
        backbone = model.encoder1.backbone_S2
    elif args.model == "dino":
        backbone = model.backbone
    else:
        backbone = model.backbone.backbone_S2

    
    model_checkpoint = ModelCheckpoint(
            filename="checkpoint_last_epoch_{epoch:02d}",
            dirpath=logdir,
            monitor="val_iou",
            mode="max",
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
    callbacks=[
            LearningRateMonitor(),
            metric_callback,
            model_checkpoint,
            ]

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        # deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=100,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.enable_progress_bar,
    )


    classifier = Segmentation(
        args=args,
        backbone=backbone,
        batch_size_per_device=args.batch_size,
        freeze=False
    )

    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    CKPT_PATH = model_checkpoint.best_model_path
    print(CKPT_PATH)
    checkpoint = torch.load(CKPT_PATH)
    classifier.load_state_dict(checkpoint["state_dict"])
    trainer.test(ckpt_path="best", dataloaders=val_dataloader)



    