import numpy as np
import flax.linen as nn

from src.dataloader import DataLoader
from src.trainer import Trainer
from src.metrics import *
from src.schedules import *
from src.optimizers import *


class CustomDataloader(DataLoader):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        SAMPLES = 256
        self.x = np.random.normal(size=(SAMPLES, 128, 128, 3))
        self.y = np.random.normal(size=(SAMPLES, 10))

    def __len__(self):
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, index: int):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        batch_x = self.x[start:end]
        batch_y = self.y[start:end]

        return batch_x, batch_y


class CNN(nn.Module):

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        x = nn.avg_pool(x, x.shape[1:-1])
        x = nn.Dense(10)(x)
        x = nn.softmax(x)
        return x


if __name__ == '__main__':
    lr = 0.001
    warmup_epochs = 10
    steps_per_epoch = 128
    epochs = 200
    batch_size = 32

    dataloader = CustomDataloader(batch_size)
    val_dataloader = CustomDataloader(batch_size)
    model = CNN()

    optimizer = rmsprop(
        warmup_cosine(
            warmup_epochs*steps_per_epoch,
            (epochs-warmup_epochs) * steps_per_epoch,
            lr
        )
    )

    trainer = Trainer(model, dataloader, val_dataloader)
    trainer.begin(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        optimizer=optimizer,
        loss_fn=mse,
        metrics=[mape],
        val_freq=5
    )
