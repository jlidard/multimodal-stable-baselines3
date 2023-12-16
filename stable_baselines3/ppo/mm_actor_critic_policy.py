"""Policies: abstract base class and concrete implementations."""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy

# from stable_baselines3.ppo.mm_mlp_extractor import MultiModalMlpExtractor

class MultiModalActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    Policy class for **multi-modal** actor-critic algorithms (has both policy and value prediction).
    Used multi-modal PPO and the likes. Learns a **set** of policies by extending PPO to handle
    prediction of multiple actions and multiple values simultaneously. Learns the policy set in
    an on-policy way.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def forward(self, obs_dict: dict, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        obs_unpacked = obs_dict["obs"]
        mode = obs_dict["mode"]
        mode_long = mode.argmax(dim=-1)
        batch_size, num_intent = mode.shape

        # Preprocess the observation if needed
        features = self.extract_features(obs_dict)

        # Emit outputs (advantage, logits)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        # marginal_distribution_loc = th.cat((th.gather(distribution.distribution.mean, -1, mode_long[:, None]),
        #                                          distribution.distribution.mean[:, -1:]), -1)
        #
        # marginal_distribution_scale = th.cat((th.gather(distribution.distribution.scale, -1, mode_long[:, None]),
        #                                          distribution.distribution.scale[:, -1:]), -1)
        # marginal_distribution = DiagGaussianDistribution(
        #     marginal_distribution_loc.shape[-1]
        # ).proba_distribution(marginal_distribution_loc, marginal_distribution_scale.log())
        # actions = th.zeros((batch_size, num_intent+1))
        # samples = marginal_distribution.get_actions(deterministic=deterministic)
        # actions = actions.scatter(-1,  mode_long[:, None], samples[:, 0:1])
        # actions[:, -1] = samples[:, -1]
        # marginal_action = th.cat((th.gather(actions, -1, mode_long[:, None]), actions[:, -1:]), -1)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob





