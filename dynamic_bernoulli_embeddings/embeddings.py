"""Contains embedding model implementation"""
from collections import Counter
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .preprocessing import Data

log = getLogger(__name__)

class DynamicBernoulliEmbeddingModel(nn.Module):
    def __init__(
        self,
        V,
        T,
        m_t,
        dictionary,
        sampling_distribution,
        k=50,
        lambda_=1e4,
        lambda_0=1,
        ns=20,
    ):
        """
        Parameters
        ----------
        V : int
            Vocabulary size.
        T : int
            Number of timesteps.
        m_t : dict
            The total number of tokens in each timestep to compute the scaling factor
            for the pseudo log likelihoods.
        dictionary : dict
            Maps word to index.
        sampling_distribution : tensor (V,)
            The unigram distribution to use for negative sampling.
        k : int
            Embedding dimension.
        lambda_ : int
            Scaling factor on the time drift prior.
        lambda_0 : int
            Scaling factor on the embedding priors.
        ns : int
            Number of negative samples.
        """
        super().__init__()
        self.V = V  # Vocab size.
        self.T = T  # Number of timestepss.
        self.k = k  # Embedding dimension.
        self.total_tokens = sum(m_t.values())  # Used for scaling factor for pseudo LL
        self.lambda_ = lambda_  # Scaling factor on the time drift prior.
        self.lambda_0 = lambda_0  # Scaling factor on the embedding priors.
        self.sampling_distribution = Categorical(logits=sampling_distribution)
        self.negative_samples = ns  # Number of negative samples.
        self.dictionary = dictionary
        self.dictionary_reverse = {v: k for k, v in dictionary.items()}

        # Embeddings parameters.
        self.rho = nn.Embedding(V * T, k)  # Stacked dynamic embeddings
        self.alpha = nn.Embedding(V, k)  # Time independent context embeddings
        with torch.no_grad():
            nn.init.normal_(self.rho.weight, 0, 0.01)
            nn.init.normal_(self.alpha.weight, 0, 0.01)

        # Transformations
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def L_pos(self, eta):
        log.debug("Running model.L_pos()")
        return self.log_sigmoid(eta).sum()

    def L_neg(self, batch_size, times, contexts_summed):
        log.debug("Running model.L_neg()")
        neg_samples = self.sampling_distribution.sample(
            torch.Size([batch_size, self.negative_samples])
        )
        neg_samples = neg_samples + (times * self.V).reshape((-1, 1))
        neg_samples = neg_samples.T.flatten()
        context_flat = contexts_summed.repeat((self.negative_samples, 1))
        eta_neg = (self.rho(neg_samples) * context_flat).sum(axis=1)
        return (torch.log(1 - self.sigmoid(eta_neg) + 1e-7)).sum()

    def forward(self, targets, times, contexts, validate=False, dynamic=True):
        """Forward pass of the model

        Parameters
        ----------
        targets : (batch_size,)
        times : (batch_size,)
        contexts : (batch_size, 2 * context_size)
        dynamic : bool
            Indicates whether to include the drift component of the loss.

        Returns
        -------
        loss
        L_pos
        L_neg
        L_prior
        """
        log.debug("Running model.forward()")
        batch_size = targets.shape[0]

        # Since the embeddings are stacked, adjust the indices for the targets.
        # In other words, word `i` in time slice `j` would be at position
        # `j * V + i` in the embedding matrix where V is the vocab size.
        targets_adjusted = times * self.V + targets

        # -1 indicates out of bounds for the context word, so mask these out so
        # they don't contribute to the context sum.
        context_mask = contexts == -1
        contexts[context_mask] = 0
        contexts = self.alpha(contexts)
        contexts[context_mask] = 0
        contexts_summed = contexts.sum(axis=1)
        eta = (self.rho(targets_adjusted) * contexts_summed).sum(axis=1)

        # Loss
        loss, L_pos, L_neg, L_prior = None, None, None, None

        L_pos = self.L_pos(eta)
        if not validate:
            L_neg = self.L_neg(batch_size, times, contexts_summed)
            loss = (self.total_tokens / batch_size) * (L_pos + L_neg)
            L_prior = -self.lambda_0 / 2 * (self.alpha.weight ** 2).sum()
            L_prior += -self.lambda_0 / 2 * (self.rho.weight[0] ** 2).sum()
            if dynamic:
                rho_trans = self.rho.weight.reshape((self.T, self.V, self.k))
                L_prior += (
                    -self.lambda_ / 2 * ((rho_trans[1:] - rho_trans[:-1]) ** 2).sum()
                )
            loss += L_prior
            loss = -loss

        return loss, L_pos, L_neg, L_prior

    def get_embeddings(self):
        """Gets trained embeddings and reshapes them into (T, V, k)"""
        embeddings = (
            self.rho.cpu()
            .weight.data.reshape((self.T, len(self.dictionary), self.k))
            .numpy()
        )
        return embeddings


class DynamicBernoulliEmbeddingModelDev(DynamicBernoulliEmbeddingModel):
    def L_neg(self, batch_size, times, contexts_summed):
        log.debug("Running model.L_neg()")
        neg_samples = self.sampling_distribution.sample(
            torch.Size([batch_size, self.negative_samples])
        )
        neg_samples = neg_samples + (times * self.V).reshape((-1, 1))
        # neg_samples = neg_samples.T.flatten()
        # context_flat = contexts_summed.repeat((self.negative_samples, 1))
        # eta_neg = (self.rho(neg_samples) * context_flat).sum(axis=1)
        # return (torch.log(1 - self.sigmoid(eta_neg) + 1e-7)).sum()
        neg_rho = self.rho(neg_samples)
        context = contexts_summed.unsqueeze(1)
        eta_neg = (neg_rho * context).sum(dim=-1)
        return self.log_sigmoid(-eta_neg).sum()


class DynamicBernoulliEmbeddingModelNew(nn.Module):
    def __init__(
        self,
        data: Data,
        k: int = 50,
        lambda_: float = 1e4,
        lambda_0: float = 1.0,
        negative_samples: int = 20,
    ):
        """
        Parameters
        ----------
        k : int
            Embedding dimension.
        lambda_ : int
            Scaling factor on the time drift prior.
        lambda_0 : int
            Scaling factor on the embedding priors.
        negative_samples : int
            Number of negative samples.
        """
        super().__init__()

        # Set model parameters
        self.k = k  # Embedding dimension.
        self.lambda_ = lambda_  # Scaling factor on the time drift prior.
        self.lambda_0 = lambda_0  # Scaling factor on the embedding priors.
        # FIXME This negative samples could be set in the train() function
        self.negative_samples = negative_samples  # Number of negative samples.

        # Setup sampling distribution
        # FIXME data could have property "multisampling" or something for this check
        if isinstance(data.unigram_logits, dict):
            self.sampling_distribution = {time: Categorical(logits=dist) for time, dist in data.unigram_logits.items()}
        else:
            self.sampling_distribution = Categorical(logits=torch.tensor(data.unigram_logits))

        # Copy information from `data`
        self.dictionary = data.dictionary
        self.dictionary_reverse = {v: k for k, v in data.dictionary.items()}
        V = len(data.dictionary) # Vocab size.
        self.V = V  # Vocab size.
        self.T = data.T  # Number of timesteps.
        # FIXME Remove once rewrite of classes is complete
        if hasattr(data, "tokens_per_time"):
            self.total_tokens = sum(data.tokens_per_time.values())  # Used for scaling factor for pseudo LL
        else:
            self.total_tokens = sum(data.m_t.values())  # Used for scaling factor for pseudo LL
        if hasattr(data, "idx_sampling_to_overall"):
            self.sampling_map = data.idx_sampling_to_overall
        # Embeddings parameters.
        self.rho = nn.Embedding(V *  data.T, k)  # Stacked dynamic embeddings
        self.alpha = nn.Embedding(V, k)  # Time independent context embeddings
        with torch.no_grad():
            nn.init.normal_(self.rho.weight, 0, 0.01)
            nn.init.normal_(self.alpha.weight, 0, 0.01)

        # Transformations
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def L_pos(self, eta):
        log.debug("Running model.L_pos()")
        return self.log_sigmoid(eta).sum()

    def L_neg(self, batch_size, times, contexts_summed):
        log.debug("Running model.L_neg()")
        neg_samples = self.sampling_distribution.sample(
            torch.Size([batch_size, self.negative_samples])
        )
        neg_samples = neg_samples + (times * self.V).reshape((-1, 1))
        neg_rho = self.rho(neg_samples)
        context = contexts_summed.unsqueeze(1)
        eta_neg = (neg_rho * context).sum(dim=-1)
        return self.log_sigmoid(-eta_neg).sum()

    def forward(self, targets, times, contexts, validate=False, dynamic=True):
        """Forward pass of the model

        Parameters
        ----------
        targets : (batch_size,)
        times : (batch_size,)
        contexts : (batch_size, 2 * context_size)
        dynamic : bool
            Indicates whether to include the drift component of the loss.

        Returns
        -------
        loss
        L_pos
        L_neg
        L_prior
        """
        log.debug("Running model.forward()")
        batch_size = targets.shape[0]

        # Since the embeddings are stacked, adjust the indices for the targets.
        # In other words, word `i` in time slice `j` would be at position
        # `j * V + i` in the embedding matrix where V is the vocab size.
        targets_adjusted = times * self.V + targets

        # -1 indicates out of bounds for the context word, so mask these out so
        # they don't contribute to the context sum.
        context_mask = contexts == -1
        contexts[context_mask] = 0
        contexts = self.alpha(contexts)
        contexts[context_mask] = 0
        contexts_summed = contexts.sum(axis=1)
        eta = (self.rho(targets_adjusted) * contexts_summed).sum(axis=1)

        # Loss
        loss, L_pos, L_neg, L_prior = None, None, None, None

        L_pos = self.L_pos(eta)
        if not validate:
            L_neg = self.L_neg(batch_size, times, contexts_summed)
            loss = (self.total_tokens / batch_size) * (L_pos + L_neg)
            L_prior = -self.lambda_0 / 2 * (self.alpha.weight ** 2).sum()
            L_prior += -self.lambda_0 / 2 * (self.rho.weight[0] ** 2).sum()
            if dynamic:
                rho_trans = self.rho.weight.reshape((self.T, self.V, self.k))
                L_prior += (
                    -self.lambda_ / 2 * ((rho_trans[1:] - rho_trans[:-1]) ** 2).sum()
                )
            loss += L_prior
            loss = -loss

        return loss, L_pos, L_neg, L_prior

    def get_embeddings(self):
        """Gets trained embeddings and reshapes them into (T, V, k)"""
        embeddings = (
            self.rho.cpu()
            .weight.data.reshape((self.T, len(self.dictionary), self.k))
            .numpy()
        )
        return embeddings


class DynamicBernoulliEmbeddingModelMult(DynamicBernoulliEmbeddingModelNew):
    def L_neg(self, batch_size, times, contexts_summed):
        log.debug("Running model.L_neg()")

        # Allocate empty output tensor
        neg_samples = torch.zeros((batch_size, self.negative_samples), device=times.device)

        # Determine how many samples are need per time period
        times_count = Counter(times.tolist())

        # Iterate over all timesteps and sample negative examples
        for time in range(self.T):
            if not time in times_count:
                continue
            # Sample everything for this time period
            sample_raw = self.sampling_distribution[time].sample(
                torch.Size([times_count[time], self.negative_samples])
            )
            # Map time-specific samples to overall index positions
            sample_global = self.sampling_map[time][sample_raw]

            # Find the positions in `times` corresponding to this timestep
            indices = (times == time).nonzero(as_tuple=True)[0]

            neg_samples[indices] = sample_global

        neg_samples = neg_samples + (times * self.V).reshape((-1, 1))

        neg_rho = self.rho(neg_samples)
        context = contexts_summed.unsqueeze(1)
        eta_neg = (neg_rho * context).sum(dim=-1)
        # eta_neg = (self.rho(neg_samples) * contexts_summed.unsqueeze(1)).sum(dim=-1)
        return self.log_sigmoid(-eta_neg).sum()
