from datetime import datetime

from data import DataModule
from lightning_model import LightningModel
from model import Model

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


if __name__ == "__main__":
    # load data
    dm = DataModule()

    # model
    model = Model()
    lightning_model = LightningModel(model)

    # trainer
    pathname = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cpu",
        deterministic=True,
        logger=CSVLogger(save_dir="logs/", name=pathname),
        callbacks=[ModelCheckpoint(monitor="val_acc")],
    )

    trainer.fit(
        model=lightning_model,
        datamodule=dm,
    )