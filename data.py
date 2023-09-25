import os
import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str="dataset", batch_size: int=8) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def setup(self, stage: str) -> tuple:
        self.train_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]),
        )
        
        self.val_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]),
        )
        
        return self.train_dataset, self.val_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
