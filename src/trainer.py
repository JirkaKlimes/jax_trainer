from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from optax._src.base import GradientTransformation
from tqdm import tqdm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from tabulate import tabulate


from src.dataloader import DataLoader


class TrainState(train_state.TrainState):
    batch_stats: Any


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        summary: bool = True,
        seed: int = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.input_shape = dataloader[0][0].shape
        self.batch_size = self.input_shape[0]

        seed = seed if seed else jax.random.randint(
            jax.random.PRNGKey(0), (),
            jnp.iinfo(jnp.int32).min,
            jnp.iinfo(jnp.int32).max,
            dtype=jnp.int32
        )

        self.key = jax.random.PRNGKey(seed)

        self.init_params()
        if summary:
            self.tabulate_model()

    def init_params(self):
        print('Initialazing model params')
        x = jnp.ones(self.input_shape, jnp.float32)
        variables = self.model.init(self.key, x, training=False)
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']

    def tabulate_model(self):
        x = jnp.ones(self.input_shape, jnp.float32)
        print(self.model.tabulate(self.key, x))

    def begin(
        self,
        epochs: int,
        steps_per_epoch: int,
        loss_fn: callable,
        metrics: list[callable],
        optimizer: GradientTransformation,
        val_freq: int = 1,
        checkpoint_path: Path = None
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.val_freq = val_freq

        self.checkpoint_path = Path(
            f'./checkpoints/{self.model.__class__.__name__} - {dt.datetime.utcnow()}') if checkpoint_path is None else checkpoint_path
        self.checkpoint_path.mkdir(parents=True)

        self.init_train_state()
        self.compile_training_step()
        if self.val_dataloader is not None:
            self.compile_validation_step()

        self.train_loop()

    def init_train_state(self):
        print('Initialazing train state')
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            tx=self.optimizer,
            params=self.params,
            batch_stats=self.batch_stats
        )

    def compile_training_step(self):
        print('Compiling model')

        @jax.jit
        def update_step(state, batch):

            x, y = batch

            def run_inference(params):

                predictions, new_state = state.apply_fn(
                    {'params': params}, x, training=True, mutable=['batch_stats'])

                loss = self.loss_fn(y, predictions)
                metrics = {
                    m.__name__: m(y, predictions)
                    for m in self.metrics
                }
                return loss, (new_state, metrics)

            grad_fn = jax.value_and_grad(run_inference, has_aux=True)
            (loss, (new_state, metrics)), grads = grad_fn(state.params)
            state = state.apply_gradients(
                grads=grads, batch_stats=new_state['batch_stats'])

            return (loss, metrics), state

        update_step(self.state, self.dataloader[0])
        self.update_step = update_step

    def compile_validation_step(self):
        print('Compiling validation model')

        @jax.jit
        def validate_step(state, batch):

            x, y = batch

            def run_inference(params):

                predictions, _ = state.apply_fn(
                    {'params': params}, x, training=False, mutable=['batch_stats'])

                loss = self.loss_fn(y, predictions)
                metrics = {
                    m.__name__: m(y, predictions)
                    for m in self.metrics
                }
                return loss, metrics

            loss, metrics = run_inference(state.params)

            return loss, metrics

        validate_step(self.state, self.dataloader[0])
        self.validate_step = validate_step

    def train_loop(self):
        print('Starting training loop')

        best_train_loss = float('inf')
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            metrics_history = {'loss': []}
            metrics_history.update({m.__name__: [] for m in self.metrics})

            pbar = tqdm(
                list(range(self.steps_per_epoch)),
                ascii=' =', desc=f'Epoch {epoch+1}/{self.epochs}'
            )
            for step in pbar:

                batch_idx = (
                    epoch*self.steps_per_epoch + step
                ) % len(self.dataloader)

                batch = self.dataloader[batch_idx]
                (loss, metrics), self.state = self.update_step(self.state, batch)
                loss, metrics = self.validate_step(self.state, batch)
                metrics_history['loss'].append(loss)

                for m in metrics:
                    metrics_history[m].append(metrics[m])

                lr = self.state.opt_state.hyperparams['learning_rate']
                if step + 1 < self.steps_per_epoch:
                    pbar.set_postfix_str(f'Loss: {loss:.5e} | lr: {lr:.5e}')
                else:
                    loss = sum(metrics_history['loss']) / \
                        len(metrics_history['loss'])
                    pbar.set_postfix_str(f'Loss: {loss:.5e} | lr: {lr:.5e}')

            if loss < best_train_loss:
                best_train_loss = loss
                with open(self.checkpoint_path.joinpath('./best_train_params.pickle'), 'wb') as f:
                    pickle.dump(self.state.params, f)

            with open(self.checkpoint_path.joinpath('./last_train_params.pickle'), 'wb') as f:
                pickle.dump(self.state.params, f)

            if self.val_dataloader is not None and not (epoch+1) % self.val_freq:
                metrics_history = {'loss': []}
                metrics_history.update({m.__name__: [] for m in self.metrics})

                headers = ['Metric', 'Mean', 'Median', 'Min', 'Max']
                table = []

                pbar = tqdm(
                    list(range(len(self.val_dataloader))), ascii=' =',
                    desc=f'Validation {(epoch+1) // self.val_freq}/{self.epochs // self.val_freq}'
                )

                for step in pbar:
                    batch = self.val_dataloader[step]
                    loss, metrics = self.validate_step(self.state, batch)
                    metrics_history['loss'].append(loss)

                    for m in metrics:
                        metrics_history[m].append(metrics[m])

                    if step + 1 < len(self.val_dataloader):
                        pbar.set_postfix_str(f'Loss: {loss:.5e}')
                    else:
                        pbar.set_postfix_str()

                        for m, v in metrics_history.items():
                            metrics_history[m] = list(map(float, v))

                        for m, v in metrics_history.items():
                            mean = sum(v) / len(v)
                            median = sorted(v)[len(v)//2]
                            row = [m, mean, median, min(v), max(v)]
                            table.append(row)

                loss = sum(metrics_history['loss']) / \
                    len(metrics_history['loss'])

                if loss < best_val_loss:
                    best_val_loss = loss

                    with open(self.checkpoint_path.joinpath('./best_val_params.pickle'), 'wb') as f:
                        pickle.dump(self.state.params, f)

                with open(self.checkpoint_path.joinpath('./last_val_params.pickle'), 'wb') as f:
                    pickle.dump(self.state.params, f)

                print(tabulate(table, headers, tablefmt="mixed_grid", floatfmt='e'))
