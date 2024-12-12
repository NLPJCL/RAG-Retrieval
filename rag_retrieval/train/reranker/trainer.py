from __future__ import annotations


import os
import re
import shutil
from typing import Any, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        accelerator: Accelerator,
        validation_dataloader: DataLoader | None = None,
        epochs: int = 3,
        lr_scheduler: LRScheduler,
        log_interval: int = 10,
        save_on_epoch_end: bool = True,
        tokenizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end
        self.tokenizer = tokenizer

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None
        self.progress_bar = DistributedTqdmProgressBar(
            self.accelerator, self.epochs, num_steps_per_epoch=num_steps_per_epoch
        )
        self.current_step = 0

    def train(self):
        for current_epoch in range(1, self.epochs + 1):
            self.model.train()
            self.progress_bar.on_epoch_start()

            for batch_index, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()

                    batch_output = self.model(batch[0], batch[1])
                    loss = batch_output.loss

                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    self.lr_scheduler.step()
                    self.train_loss_tracker.update(loss)

                if batch_index % self.log_interval == 0:
                    training_log_dict = dict(training_loss=self.train_loss_tracker.loss)
                    lr_list = self.lr_scheduler.get_lr()
                    for group_id, lr in enumerate(lr_list, 0):
                        training_log_dict[f"lr_group_{group_id}"] = lr
                    self.log_metrics(training_log_dict, step=self.current_step)

                if (
                    self.validation_dataloader
                    and batch_index % (self.log_interval * 30) == 0
                ):
                    validation_loss = evaluate(
                        self.model,
                        self.validation_dataloader,
                        self.validation_loss_tracker,
                    )
                    self.accelerator.log(
                        {"validation_loss": validation_loss}, step=self.current_step
                    )
                    if self.accelerator.is_local_main_process and self.current_step > 0:
                        save_dir = self.get_checkpoint_dir(current_epoch)
                        save_dir = os.path.join(
                            save_dir,
                            f"_step_{self.current_step}_val_loss_{validation_loss:.3f}",
                        )
                        print(save_dir)
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        unwrapped_model.save_pretrained(
                            save_dir, safe_serialization=True
                        )
                        self.tokenizer.save_pretrained(save_dir)
                    self.accelerator.wait_for_everyone()

                self.progress_bar.update()
                self.current_step += 1

            self.train_loss_tracker.on_epoch_end()
            self.progress_bar.on_epoch_end()

            if self.validation_dataloader:
                validation_loss = evaluate(
                    self.model,
                    self.validation_dataloader,
                    self.validation_loss_tracker,
                )
                self.accelerator.print(
                    f"Epoch {current_epoch} Validation loss: {validation_loss:.6f}"
                )

            if self.save_on_epoch_end:
                if self.accelerator.is_local_main_process:
                    save_dir = self.get_checkpoint_dir(current_epoch)
                    print(save_dir)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    unwrapped_model.save_pretrained(save_dir, safe_serialization=True)
                    self.tokenizer.save_pretrained(save_dir)
                self.accelerator.wait_for_everyone()

        self.accelerator.end_training()

    def log_metrics(self, metrics: dict[str, float], step: int):
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f"{prefix}/{k}": v for k, v in values.items()}

    def get_checkpoint_dir(self, current_epoch):

        self.accelerator.project_configuration.automatic_checkpoint_naming = False
        output_dir = os.path.join(self.accelerator.project_dir, "checkpoints")
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
            folders = [
                os.path.join(output_dir, folder) for folder in os.listdir(output_dir)
            ]
            if self.accelerator.project_configuration.total_limit is not None and (
                len(folders) + 1 > self.accelerator.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(
                        map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder))
                    )[0]

                folders.sort(key=_inner)
                for folder in folders[
                    : len(folders)
                    + 1
                    - self.accelerator.project_configuration.total_limit
                ]:
                    shutil.rmtree(folder)

        output_dir = os.path.join(output_dir, f"checkpoint_{current_epoch-1}")
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
        return output_dir


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_tracker: LossTracker | None = None,
):
    model.eval()
    loss_tracker = loss_tracker or LossTracker()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = model(batch[0], batch[1])
            loss = batch_output.loss
            loss_tracker.update(loss)
    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()
    return loss


class DummyProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class DistributedTqdmProgressBar:
    def __init__(
        self, accelerator, epochs: int, num_steps_per_epoch: int | None, **kwargs
    ) -> None:
        self.accelerator = accelerator
        self.epochs = epochs
        self.current_epoch = 1
        self.num_steps_per_epoch = num_steps_per_epoch
        self.tqdm_kwargs = kwargs

    def on_epoch_start(self):
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(total=self.num_steps_per_epoch, **self.tqdm_kwargs)
        else:
            self.progress_bar = DummyProgressBar()

    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        self.progress_bar.close()

    def show_metrics(self, metrics: dict[str, float]) -> None:
        description = f"Epoch {self.current_epoch}/{self.epochs}"
        for name, score in metrics.items():
            if "lr" in name:
                continue
            description += f" - {name}: {score:.6f}"
        self.progress_bar.set_description(description)


class LossTracker:
    def __init__(
        self,
        ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        loss = loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return round(float(self._loss), self.ndigits)
