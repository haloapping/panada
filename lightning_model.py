from typing import Any

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torchmetrics.functional import accuracy


class LightningModel(L.LightningModule):
    def __init__(self, model, lr: int=0.001) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        
    def forward(self, x) -> Any:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        features, true_labels = batch
        probs = self(features)
        loss = F.cross_entropy(probs, true_labels)
        acc = accuracy(probs.argmax(1), true_labels, task="multiclass", num_classes=23)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        features, true_labels = batch
        probs = self(features)
        loss = F.cross_entropy(probs, true_labels)
        acc = accuracy(probs.argmax(1), true_labels, task="multiclass", num_classes=23)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx) -> None:
        features, true_labels = batch
        probs = self(features)
        loss = F.cross_entropy(probs, true_labels)
        acc = accuracy(probs.argmax(1), true_labels, task="multiclass", num_classes=23)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        
        return optimizer