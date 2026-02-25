import torch
from pytorch_lightning import LightningModule

class BaseMultimodalModel(LightningModule):
    """
    Base LightningModule for all multimodal models.
    Establishes a common structure for initialization.
    """
    def __init__(self, args, data_stats=None):
        super().__init__()
        self.args = args
        self.data_stats = data_stats
        
        if data_stats is not None:
            self.save_hyperparameters(ignore=["data_stats"])
        else:
            self.save_hyperparameters()

        # ---------- Architecture ----------
        self._build_backbones()
        self._build_projectors()
        self._build_losses()
        self._build_metrics()

    def _build_backbones(self):
        """Initialize encoders and backbones"""
        pass

    def _build_projectors(self):
        """Initialize projection heads"""
        pass

    def _build_losses(self):
        """Initialize criterion and losses"""
        pass

    def _build_metrics(self):
        """Initialize metrics and regressors"""
        pass
