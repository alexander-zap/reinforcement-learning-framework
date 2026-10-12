"""The package exports the documented public API."""

import inspect

import rl_framework
import rl_framework.agent as agent_module
import rl_framework.util as util_module
from rl_framework.agent import Agent, ILAgent, RLAgent


def test_package_metadata():
    assert rl_framework.__title__ == "Reinforcement Learning Framework"


def test_agents_are_exported_and_concrete():
    concrete = ["StableBaselinesAgent", "AsyncStableBaselinesAgent", "CustomAgent", "ImitationAgent"]
    for name in concrete:
        agent_class = getattr(agent_module, name)
        assert issubclass(agent_class, Agent)
        assert not inspect.isabstract(agent_class), name


def test_agent_base_classes_are_abstract():
    for agent_class in (Agent, RLAgent, ILAgent):
        assert inspect.isabstract(agent_class)


def test_util_exports():
    expected = {
        "ClearMLConnector",
        "ClearMLDownloadConfig",
        "ClearMLUploadConfig",
        "Connector",
        "DownloadConfig",
        "DummyConnector",
        "HuggingFaceConnector",
        "HuggingFaceDownloadConfig",
        "HuggingFaceUploadConfig",
        "UploadConfig",
        "FeaturesExtractor",
        "StableBaselinesFeaturesExtractor",
        "encode_observations_with_features_extractor",
        "get_sb3_policy_kwargs_for_features_extractor",
        "wrap_environment_with_features_extractor_preprocessor",
        "MetricAggregator",
        "apply_action_bias",
        "validate_initial_action_bias",
        "reset_optimizer_state",
        "GammaScheduleCallback",
        "LoggingCallback",
        "ResetInfoCallback",
        "SavingCallback",
        "add_callbacks_to_callback",
        "Environment",
        "EnvironmentFactory",
        "record_video",
    }
    assert expected <= set(dir(util_module))
