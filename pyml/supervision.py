from typing import Any, Callable, Dict, TypeAlias

import torch
from torch import  Tensor, dtype
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import numpy as np
from numpy import asarray, ndarray

from pandas import DataFrame

from pyml.mlp import MLP
from pyml.util import *


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
    loss_fn_cls: Callable[[Any], Callable[[Predictions, Targets], Loss]],
    optim_cls: Callable[[Any], Optimizer],
    model: Module,
    train_dataset: Dataset,
    test_dataset: Dataset | None = None,
    epochs: int | None = 50,
    interval: int | float = 0.1,
    **hyperparams
) -> DataFrame:
    """
        Supervise a model on a dataset, given a loss-function and optimizer,
        trained for a number of epochs.

        Return a dataframe with the losses, tracked based-on the track-interval.

        The track-interval can be a fraction--a percentage of the total epochs--or,
        an integer number of epochs.

        Optionally, a test-dataset can be provided; the results of which will
        be included in the returned `DataFrame` based on the same track-interval.

        The keyword-arguments at the end are the kwargs--hyperparams--to the `DataLoader`,
        and both the loss-class and optimizer-class constructors; eg, lr, batch_size, shuffle,
        weight-decay; the constructors' parameters are each queried and matched in
        the **kwargs (called **hyperparams in this case)--kwarg name collisions aren't 
        dealt with, yet.
    """
    # get the kwargs for each of the loss, optim, and dataloader's constructor signature and apply them each in their instatiation.

    # loss kwargs from hyperparams
    loss_fn_kwargs = get_fn_sig_kwargs(loss_fn_cls.__init__, hyperparams)

    loss_fn = loss_fn_cls(**loss_fn_kwargs)

    # optim kwargs from hyperparams
    optim_kwargs = get_fn_sig_kwargs(optim_cls.__init__, hyperparams)

    optim = optim_cls(model.parameters(), **optim_kwargs)

    # loader kwargs from hyperparams
    loader_kwargs = get_fn_sig_kwargs(DataLoader.__init__, hyperparams)

    loader = DataLoader(
        dataset = train_dataset,
        **loader_kwargs
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        **loader_kwargs
    ) if test_dataset else None

    # if interval is a fraction, take the fraction of the total epochs as the interval; otherwise, keep the integer number of epochs as the interval.
    interval = interval if isinstance(interval, int) else int(epochs * interval)

    # will contain the loss from the last epoch in the interval--last batch if applicable.
    losses = []

    # will keep a running loss to be averaged each interval.
    interval_loss = 0.

    # last epoch loss for the test-data.
    test_losses = []

    # running interval loss for the test-data
    test_interval_loss = 0.

    for epoch in range(epochs+1):
        model.train()

        for features, targets in loader:
            loss = model_train(features, targets, model, loss_fn, optim)

            interval_loss += loss
        
        if test_loader:
            for features, targets in test_loader:
                test_loss = model_test(features, targets, model, loss_fn)

                test_interval_loss += test_loss

        if epoch % interval == 0:
            losses.append([
                epoch,
                interval_loss / interval if epoch != 0 else interval_loss,
                loss
            ])

            interval_loss = 0.

            if test_loader:
                test_losses.append([
                    test_interval_loss / interval if epoch != 0 else test_interval_loss, 
                    test_loss
                ])

                test_interval_loss = 0.
        
    return DataFrame(
        data=asarray(losses),
        columns=[
            "EPOCH", 
            "interval-loss", 
            "loss",
        ],
    ).rename_axis(f"INTERVAL") if not test_losses else DataFrame(
        data=np.c_[
            asarray(losses), 
            asarray(test_losses)
        ],
        columns=[
            "EPOCH",
            "interval-loss",
            "loss",
            "interval-loss-test",
            "loss-test"
        ],
    ).rename_axis("INTERVAL")


def hypervise_mlp(
    loss_fn_cls: Callable[[Any], Callable[[Predictions, Targets], Loss]],
    optim_cls: Callable[[Any], Optimizer],
    train_dataset: Dataset,
    test_dataset: Dataset | None = None,
    Hn: int = 2,
    H: int | None = 10,
    activation: Callable = torch.nn.ReLU(),
    dropouts: list[float|None] | float | None = None,
    thresholds: list | None = None,
    epochs: int = 100,
    interval: int|float = 0.1,
    **hyperparams
) -> tuple[MLP, DataFrame]:
    """
        This creates a multi-layer-perceptron `MLP` and supervises it based-on 
        a dataset.

        See `supervise`; this function only adds arguments for creating an
        `MLP` model (a multi-layer-perceptron neural-net).

        This will return the trained-model and the results of the supervision in a dataframe.
    """
    # for the shapes to determine D_in and D_out.
    sample_x, sample_y = train_dataset[0]

    # create the model.
    model = MLP(
        D_in=len(sample_x) if len(list(sample_x)) < 2 else sample_x.shape[-1],
        D_out=len(sample_y) if len(list(sample_y)) < 2 else sample_y.shape[-1],
        H=H,
        Hn=Hn,
        activation=activation,
        dropouts=dropouts,
        thresholds=thresholds,
        dtype=torch.float32
    )

    # supervise the model.
    results = supervise(
        model=model,
        loss_fn_cls=loss_fn_cls,
        optim_cls=optim_cls,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        interval=interval,
        **hyperparams
    )

    return model, results


def hypergrid_mlp(
    loss_fn_cls_array: list,
    optim_cls_array: list,
    train_dataset: Dataset,
    test_dataset: Dataset,
    H_array: list[int],
    Hn_array: list[int],
    activation_array: list[Callable],
    dropouts_array: list,
    thresholds_array: list,
    epochs_array: list,
    interval: int|float,
    hyperparams_array: list[dict]
) -> tuple[list, list, list]:
    """
        Use an array of parameters for each of the relevant parameters to `hypervise_mlp`
        and run every combination of parameters in the `hypervise_mlp` function.

        Returns an array of the resulting models, loss-dataframes, and a dictionary
        containing the in-order indices within each of parameters' arrays at that
        given `hypervise_mlp`'s call-index. mouthful
    """
    hypergrid_desc_array = []

    models_array = []

    results_array = []

    for lfc_idx, loss_fn_cls in enumerate(loss_fn_cls_array):
        for oc_idx, optim_cls in enumerate(optim_cls_array):
            for H_idx, H in enumerate(H_array):
                for Hn_idx, Hn in enumerate(Hn_array):
                    for act_idx, activation in enumerate(activation_array):
                        for do_idx, dropouts in enumerate(dropouts_array):
                            for th_idx, thresholds in enumerate(thresholds_array):
                                for ep_idx, epochs in enumerate(epochs_array):
                                    for hp_idx, hyperparams in enumerate(hyperparams_array):
                                        model, results = hypervise_mlp(
                                            loss_fn_cls=loss_fn_cls,
                                            optim_cls=optim_cls,
                                            train_dataset=train_dataset,
                                            test_dataset=test_dataset,
                                            H=H,
                                            Hn=Hn,
                                            activation=activation,
                                            dropouts=dropouts,
                                            epochs=epochs,
                                            interval=interval,
                                            **hyperparams
                                        )

                                        hypergrid_desc = {
                                            "loss_fn_cls_idx": lfc_idx,
                                            "optim_cls_idx": oc_idx,
                                            "H_idx": H_idx,
                                            "Hn_idx": Hn_idx,
                                            "activation_idx": act_idx,
                                            "dropouts_idx": do_idx,
                                            "thresholds_idx": th_idx,
                                            "epochs_idx": ep_idx,
                                            "hyperparams_idx": hp_idx
                                        }

                                        hypergrid_desc_array.append(hypergrid_desc)

                                        models_array.append(model)

                                        results_array.append(results)

    return models_array, results_array, hypergrid_desc_array

                                        