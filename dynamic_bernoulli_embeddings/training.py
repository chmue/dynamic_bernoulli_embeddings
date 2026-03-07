"""Functions for building the training loop"""
from logging import getLogger
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .embeddings import DynamicBernoulliEmbeddingModel, DynamicBernoulliEmbeddingModelDev
from .preprocessing import Data, DataFromDict

log = getLogger(__name__)

# def build_data() -> Data:
#     pass
#     DynamicBernoulliEmbeddingModelMult

# def build_model(
#     dataset: Union[pd.DataFrame, dict],
#     dictionary,
#     model_type: DynamicBernoulliEmbeddingModel = DynamicBernoulliEmbeddingModel,
#     ) -> DynamicBernoulliEmbeddingModel:
#     pass

def train(
    model: DynamicBernoulliEmbeddingModel,
    training_data: Data,
    validation_data: Optional[Data] = None,
    validation_interval: Optional[int] = None,
    minibatches: int = 300,
    epochs: int = 10,
    # negative_samples: int = 20,
    learning_rate: float = 2e-3,
    return_loss: bool = True
    ) -> Optional[pd.DataFrame]:
    """Train embedding `model` with `data`."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_history = []
    for i in range(epochs + 1):
        log.debug(f"Starting epoch {i}")

        # Initialize weights from the epoch 0 "burn in" period and reset the optimizer.
        if i == 1:
            with torch.no_grad():
                model.rho.weight = torch.nn.Parameter(
                    model.rho.weight[: model.V].repeat((model.T, 1))
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        pbar = tqdm(enumerate(data.epoch(minibatches)), total=minibatches)
        pbar.set_description(f"Epoch {i}")
        for j, (targets, contexts, times) in pbar:
            log.debug(f"Running minibatch {j} of {minibatches}")
            model.train()
            model.zero_grad()
            # The first epoch ignores time for initializing weights.
            if i == 0:
                times = torch.zeros_like(times)
            loss, L_pos, L_neg, L_prior = model(targets, times, contexts, dynamic=i > 0)
            loss.backward()
            optimizer.step()

            # Validation.
            L_pos_val = None
            if validation_data is not None and i > 0 and j % validation_interval == 0:
                L_pos_val = 0
                model.eval()
                for val_targets, val_contexts, val_times in data_val.epoch(10):
                    _, L_pos_val_batch, _, _ = model(
                        val_targets, val_times, val_contexts, validate=True
                    )
                    L_pos_val += L_pos_val_batch.item()

            # Collect loss history. Ignore the initialization epoch 0.
            if i > 0:
                batch_loss = (
                    loss.item(),
                    L_pos.item(),
                    L_neg.item(),
                    L_prior.item() if L_prior else None,
                    L_pos_val,
                )
                loss_history.append(batch_loss)

    loss_history = pd.DataFrame(
        loss_history, columns=["loss", "l_pos", "l_neg", "l_prior", "l_pos_val"]
    )
    if return_loss:
        return loss_history
    # return model, loss_history


def train_model(
    dataset,
    dictionary,
    validation=None,
    notebook=True,
    m=300,
    num_epochs=10,
    lr=2e-3,
    validate_after=100,
    **kwargs,
):
    """"Trains the model

    Parameters
    ----------
    dataset : `pd.DataFrame`
    dictionary : dict
        Maps a word to an index. If a word in the dataset is not present in this
        dictionary, it will be removed/ignored.
    validation : float
        If None, no held out validation set. Otherwise, this is the proportion of the
        dataset to use as a validation set.
    notebook : bool
        Indicates whether the function is being run in a notebook to allow for nicer
        progress bars.
    m : int
        The number of mini batches to use.
    num_epochs : int
        Number of epochs to train for, excluding the first initialization epoch.
    lr : float
        Learning rate.
    validate_after : int
        Compute the validation metric after this many minibatches.
    **kwargs
        Forwarded to init of `DynamicBernoulliEmbeddingModel`.
    """

    # Check for gpu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a validation set only if validation is set
    if validation is not None:
        assert 0 < validation < 1
        validation_mask = np.random.random(dataset.shape[0]) < validation
        data = Data(dataset[~validation_mask], dictionary, device)
        data_val = Data(dataset[validation_mask], dictionary, device)
    else:
        if isinstance(dataset, dict):
            data = DataFromDict(dataset, dictionary, device)
        else:
            data = Data(dataset, dictionary, device)

    # Build model.
    model = DynamicBernoulliEmbeddingModelDev(
        len(data.dictionary),
        data.T,
        data.m_t,
        data.dictionary,
        data.unigram_logits,
        **kwargs,
    )
    model = model.to(device)

    # Training loop.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_history = []
    for i in range(num_epochs + 1):
        log.debug(f"Starting epoch {i}")

        # Initialize weights from the epoch 0 "burn in" period and reset the optimizer.
        if i == 1:
            with torch.no_grad():
                model.rho.weight = torch.nn.Parameter(
                    model.rho.weight[: model.V].repeat((model.T, 1))
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        pbar = tqdm(enumerate(data.epoch(m)), total=m)
        pbar.set_description(f"Epoch {i}")
        for j, (targets, contexts, times) in pbar:
            log.debug(f"Running minibatch {j} of {m}")
            model.train()
            model.zero_grad()
            # The first epoch ignores time for initializing weights.
            if i == 0:
                times = torch.zeros_like(times)
            loss, L_pos, L_neg, L_prior = model(targets, times, contexts, dynamic=i > 0)
            loss.backward()
            optimizer.step()

            # Validation.
            L_pos_val = None
            if validation is not None and i > 0 and j % validate_after == 0:
                L_pos_val = 0
                model.eval()
                for val_targets, val_contexts, val_times in data_val.epoch(10):
                    _, L_pos_val_batch, _, _ = model(
                        val_targets, val_times, val_contexts, validate=True
                    )
                    L_pos_val += L_pos_val_batch.item()

            # Collect loss history. Ignore the initialization epoch 0.
            if i > 0:
                batch_loss = (
                    loss.item(),
                    L_pos.item(),
                    L_neg.item(),
                    L_prior.item() if L_prior else None,
                    L_pos_val,
                )
                loss_history.append(batch_loss)

    loss_history = pd.DataFrame(
        loss_history, columns=["loss", "l_pos", "l_neg", "l_prior", "l_pos_val"]
    )

    return model, loss_history
