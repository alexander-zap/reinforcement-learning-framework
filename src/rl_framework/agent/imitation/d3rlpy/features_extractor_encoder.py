import copy
import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
from d3rlpy.base import LearnableConfig
from d3rlpy.models.encoders import (
    EncoderFactory,
    make_encoder_field,
    register_encoder_factory,
)
from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy.types import Shape
from dataclasses_json import config

from rl_framework.util import FeaturesExtractor


class FeaturesExtractorEncoder(Encoder):
    """The features extractor followed by d3rlpy's encoder (built for the extracted features)."""

    def __init__(self, features_extractor: FeaturesExtractor, encoder: Encoder):
        super().__init__()
        self.features_extractor = features_extractor
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.features_extractor(x))


class FeaturesExtractorEncoderWithAction(EncoderWithAction):
    """The features extractor followed by d3rlpy's state-action encoder (which concatenates the action to the
    extracted features, one-hot encoded if discrete)."""

    def __init__(self, features_extractor: FeaturesExtractor, encoder: EncoderWithAction):
        super().__init__()
        self.features_extractor = features_extractor
        self.encoder = encoder

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.features_extractor(x), action)


@dataclass()
class FeaturesExtractorEncoderFactory(EncoderFactory):
    """
    d3rlpy encoder factory which puts the agent's features extractor in front of the network of another encoder
    factory (`encoder_factory`, e.g. d3rlpy's default 256x256 MLP), so that the features extractor is trained with
    the algorithm's networks (like SB3's features extractor in front of `net_arch`).

    Every created network gets its own copy of the features extractor (target networks must not share the weights of
    the online networks). The features extractor is not part of d3rlpy's serialized config: it is set again from the
    agent on load, the weights come from the saved model.
    """

    encoder_factory: EncoderFactory = make_encoder_field()
    features_extractor: Optional[FeaturesExtractor] = dataclasses.field(
        default=None, metadata=config(exclude=lambda _: True)
    )

    def create(self, observation_shape: Shape) -> Encoder:
        return FeaturesExtractorEncoder(
            copy.deepcopy(self.features_extractor),
            self.encoder_factory.create((self.features_extractor.output_dim,)),
        )

    def create_with_action(
        self, observation_shape: Shape, action_size: int, discrete_action: bool = False
    ) -> EncoderWithAction:
        return FeaturesExtractorEncoderWithAction(
            copy.deepcopy(self.features_extractor),
            self.encoder_factory.create_with_action(
                (self.features_extractor.output_dim,), action_size, discrete_action
            ),
        )

    @staticmethod
    def get_type() -> str:
        return "rl_framework_features_extractor"


register_encoder_factory(FeaturesExtractorEncoderFactory)


def encoder_factory_fields(algorithm_config: LearnableConfig) -> list[str]:
    """Names of the encoder factory fields of a d3rlpy algorithm config (e.g. `actor_encoder_factory`)."""
    return [field.name for field in dataclasses.fields(algorithm_config) if field.name.endswith("encoder_factory")]


def put_features_extractor_in_front_of_encoders(
    algorithm_config: LearnableConfig, features_extractor: FeaturesExtractor
) -> None:
    """Put the features extractor in front of every encoder of the algorithm config (in place)."""
    for name in encoder_factory_fields(algorithm_config):
        encoder_factory = FeaturesExtractorEncoderFactory(getattr(algorithm_config, name), features_extractor)
        setattr(algorithm_config, name, encoder_factory)
