from typing import Any, Callable, Dict, TypeAlias

import pandas
import torch
from torch import  Tensor, dtype
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import numpy as np
from numpy import asarray, ndarray

from pandas import DataFrame

from pyml.mlp import MLP


#TODO: hm; maybe.
class SupervisionResults(DataFrame):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)


Predictions: TypeAlias = Tensor

Features: TypeAlias = Tensor

Targets: TypeAlias = Tensor

Loss: TypeAlias = Tensor

Optimizer: TypeAlias = Any

LearningParameters: TypeAlias = Dict

ModelParameters: TypeAlias = Any


def model_test(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    model: torch.nn.Module, 
    loss_fn: Callable[[Predictions, Targets], Loss]
) -> float:
    """
        Test a model on a batch of data given the loss-function.

        Returns the loss.
    """
    model.eval()

    with torch.inference_mode():
        pred = model(X)

        loss = loss_fn(pred, Y)

    return loss.item()


def model_train(
    X: torch.Tensor, 
    Y: torch.Tensor,
    model: torch.nn.Module, 
    loss_fn: Callable[[Predictions, Targets], Loss],
    optim: Optimizer
) -> float:
    """
        Train a model on a batch of data given the loss-function
        and optimizer.

        Returns the loss.
    """
    model.train()

    pred = model(X)

    loss = loss_fn(pred, Y)

    optim.zero_grad()

    loss.backward()

    optim.step()

    return loss.item()


def supervise(
    model: Module,
    loss_fn: Callable[[Predictions, Targets], Loss],
    optim_cls: Callable[[Any], Optimizer],
    train_dataset: Dataset,
    test_dataset: Dataset|None = None,
    epochs: int = 50,
    batch_size: int = 256,
    shuffle: bool = True,
    track_interval: int|float = 0.1,
    **optim_kwargs
) -> DataFrame:
    """
        Supervise a model on a dataset, given a loss-function and optimizer,
        trained for a number of epochs.

        Return a dataframe with the losses, tracked based-on the track-interval.

        The track-interval can be a fraction--a percentage of the total epochs--or,
        an integer number of epochs.

        Optionally, a test-dataset can be provided; the results of which will
        be included in the returned `DataFrame` based on the same track-interval.

        The extra keyword-arguments, at the end, will be assumed to be the 
        optimizer's instantiation keyword-arguments.
    """
    optim = optim_cls(model.parameters(), **optim_kwargs)

    loader = DataLoader(
        dataset = train_dataset,
        shuffle = shuffle,
        batch_size = batch_size
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=shuffle,
        batch_size=batch_size
    ) if test_dataset else None

    losses = []

    test_losses = []

    interval = track_interval if isinstance(track_interval, int) else int(epochs * track_interval)

    interval_loss = 0.

    running_loss = 0.

    test_running_loss = 0.

    test_interval_loss = 0.

    e = []

    for epoch in range(epochs+1):
        model.train()

        for features, targets in loader:
            loss = model_train(features, targets, model, loss_fn, optim)

            interval_loss += loss

            running_loss += loss
        
        if test_loader:
            for features, targets in test_loader:
                test_loss = model_test(features, targets, model, loss_fn)

                test_interval_loss += test_loss

                test_running_loss += test_loss

        if epoch % interval == 0:
            e.append(epoch)

            losses.append([
                running_loss / (epoch + 1), 
                interval_loss / interval if epoch != 0 else interval_loss,
                loss
            ])

            interval_loss = 0.

            if test_loader:
                test_losses.append([
                    test_running_loss / (epoch + 1), 
                    test_interval_loss / interval if epoch != 0 else test_interval_loss, 
                    test_loss
                ])

                test_interval_loss = 0.
        
    return DataFrame(
        data=np.c_[
            asarray(e), 
            asarray(losses)
        ],
        columns=[
            "epoch", 
            "running_loss",
            "interval_loss", 
            "epoch_loss",
        ],
    ).rename_axis("interval") if not test_losses else DataFrame(
        data=np.c_[
            asarray(e), 
            asarray(losses), 
            asarray(test_losses)
        ],
        columns=[
            "epoch",
            "running_loss",
            "interval_loss",
            "epoch_loss",
            "test_running_loss",
            "test_interval_loss",
            "test_epoch_loss"
        ],
    ).rename_axis("interval")


def hypervise_mlp(
    loss_fn: Callable[[Predictions, Targets], Loss],
    optim_cls: Callable[[Any], Optimizer],
    train_dataset: Dataset,
    test_dataset: Dataset|None = None,
    Hn: int=2,
    H: int|None=10,
    activation: Callable = torch.nn.ReLU(),
    dropouts: list[float|None] | float | None=None,
    thresholds: list | None=None,
    dtype: dtype = torch.float32,
    epochs: int = 100,
    batch_size: int = 256,
    shuffle: bool = True,
    track_interval: int|float = 0.1,
    **optim_kwargs
) -> tuple[MLP, DataFrame]:
    """
        This creates a multi-layer-perceptron `MLP` and supervises it based-on 
        a dataset.

        See `supervise`; this function only adds arguments for creating an
        `MLP` model (a multi-layer-perceptron neural-net).

        This will return the trained-model and the results of the supervision in a dataframe.
    """
    # get the feature and target dimensions using the first sample from the dataset, snarl.
    tmp_xy = train_dataset[0]

    # create the model.
    model = MLP(
        D_in=tmp_xy[0].shape[-1],
        D_out=tmp_xy[1].shape[-1],
        H=H,
        Hn=Hn,
        activation=activation,
        dropouts=dropouts,
        thresholds=thresholds,
        dtype=dtype
    )

    # supervise the model.
    results = supervise(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        optim_cls=optim_cls,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        track_interval=track_interval,
        **optim_kwargs
    )

    return model, results