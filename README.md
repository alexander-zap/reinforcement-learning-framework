# Reinforcement Learning Framework

An easy-to-read Reinforcement Learning (RL) framework. Provides standardized interfaces and implementations to various Reinforcement Learning and Imitation Learning methods and utilities.

### Main Features

- Using various Reinforcement Learning algorithms to learn from gym environment interaction, which are implemented in **Stable-Baselines 3**
- Using various Imitation Learning algorithms to learn from replays, which are implemented in **Imitation**
- Integrate or implement own **custom agents and algorithms** in a standardized interface
- Upload your models (with logged metrics, checkpoints and video recordings) to **HuggingFace Hub** or **ClearML**

## Set-Up

### Install all dependencies in your development environment

To set up your local development environment, please install uv (see [installation guide](https://docs.astral.sh/uv/getting-started/installation/)) and run:

```
uv sync
```

Behind the scenes, this creates a virtual environment in `.venv` and installs `rl_framework` along with its dependencies (including the `dev` and `test` dependency groups) at the exact versions pinned in `uv.lock`. Whenever you run `uv run <command>`, that `<command>` is actually run inside this virtual environment.

You can now import functions and classes from the module with `import rl_framework`.

### Optional: Install FFMPEG to enable generation of videos (for upload)

The creation of videos for the functionality of creating video-replays of the agent performance on the environment requires installing the FFMPEG package on your machine.
This feature is important if you plan to upload replay videos to an experiment tracking service together with the agent itself.
The `ffmpeg` command needs to be available to invoke from the command line, since it is called from Python through a `os.system` invoke. Therefore, it is important that you install this package directly on your machine.

Please follow the guide which can be found [here](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) to install the FFMPEG library on your respective machine.

### Optional: Preparation for pushing your models to the HuggingFace Hub

- Create an account to HuggingFace and sign in. ➡ https://huggingface.co/join
- Create a new token with write role. ➡ https://huggingface.co/settings/tokens
- Store your authentication token from the Hugging Face website. ➡ `huggingface-cli login`

### Optional: Preparation for using a Unity environment (optional)

In order to use environments based on the Unity game framework, make sure to follow the installation procedures detailed in [following installation guideline provided by Unity Technologies](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md).
In short:

- Install Unity. ➡ https://unity.com/download
- Create a new Unity project.
- Navigate to the menu `Window -> Package Manager` and install the `com.unity.ml-agents` package in Unity. ➡ https://docs.unity3d.com/Manual/upm-ui-install.html

## Getting Started

### Configuring an environment

To integrate your environment you wish to train on, you need to create a gymnasium.Env object representing your problem.
For this you can use any existing environment with the gym interface. See [here](https://gymnasium.farama.org/api/env/) for further documentation.

### Reinforcement Learning agent

#### Class definition

To integrate the Reinforcement Learning algorithm you wish to train an agent on your environment with, you need to create an RLAgent class representing your training agent. For this you can

- use an existing Reinforcement Learning algorithm implemented in the Stable-Baselines 3 framework with [the `StableBaselinesAgent` class](src/rl_framework/agent/reinforcement/stable_baselines.py) (as seen in the [example script](exploration/train_rl_agent.py))
- create a custom Reinforcement Learning algorithm by inheriting from [the base `RLAgent` class](src/rl_framework/agent/reinforcement_learning_agent.py), which specifies the required interface

#### Training

After configuring the environment and the agent, you can start training your agent on the environment.
This can be done in one line of code:

```
agent.train(training_environments=environments, total_timesteps=N_TRAINING_TIMESTEPS)
```

Independent of which environment and which agent you choose, the unified interface allows to always start the training this way.

#### PettingZoo environments

PettingZoo `ParallelEnv`s are trained with one shared policy: each possible agent is one sub-environment ("slot") of the vectorized environment (see [`PettingZooVectorEnv`](src/rl_framework/util/pettingzoo_vector_env.py), used by the Stable-Baselines 3 and Imitation agents). The environment is reset once all agents are done.

- **Agents which restart on their own** while the others continue follow gymnasium's auto-reset convention: the done step returns the first observation of the agent's next episode and passes the last observation of the finished one as `infos[agent]["final_observation"]`.
- **Agents which leave before the others** (done, and no longer in `agents`): their done step ends their episode as usual. Until the environment is reset, their slot then gets dummy steps: an empty observation (zeros), reward 0, truncated, with the info `discard`. Dummy steps are not counted as episodes (episode logging, SB3's `ep_rew_mean`), but the algorithm still trains on them: they take up batch samples, and with normalized observations the zero observation is an ordinary state. Their share of all steps is logged as `Dummy step share` (once a dummy step occurred).
- All agents must have the same observation and action spaces.

### Imitation Learning agent

#### Class definition

To integrate the Imitation Learning algorithm you wish to train an agent on your replays with, you need to create an ILAgent class representing your training agent. For this you can

- use an existing Imitation Learning algorithm implemented in the Imitation framework with [the `ImitationAgent` class](src/rl_framework/agent/imitation/imitation/imitation.py) (as seen in the [example script](exploration/train_il_agent.py))
- create a custom Imitation Learning algorithm by inheriting from [the base `ILAgent` class](src/rl_framework/agent/imitation_learning_agent.py), which specifies the required interface

#### Training

First you need to collect the replays (recorded episode sequences) from an expert policy or a human demonstration.
They should be recorded as `imitation.TrajectoryWithRew` objects and saved with the `serialize.save` method (see  [`imitation` library documentation](https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html#storing-loading-trajectories)) and stored as files.
You can afterward load them with the following code line:

```
sequence = EpisodeSequence.from_dataset(TRAJECTORIES_PATH)
```

Afterward, you can start training your agent on the environment.
This can be done in one line of code:

```
agent.train(episode_sequence=sequence, training_environments=environments, total_timesteps=N_TRAINING_TIMESTEPS)
```

The training environments are used by the imitation learning algorithms in different ways.
Some of them only use it for the observation and action space information, while others use it for iteratively checking and improving the imitation policy.

#### Decision transformers (d3rlpy)

The decision transformers of [the `D3RLPYAgent` class](src/rl_framework/agent/imitation/d3rlpy/d3rlpy.py) (`DecisionTransformer`, `DiscreteDecisionTransformer`) choose their actions from the context of the episode and the return they aim to achieve.
Set this return with the agent parameter `target_return` (next to d3rlpy's algorithm parameters, not passed to d3rlpy's config):

```
agent = D3RLPYAgent(DiscreteDecisionTransformer, {"batch_size": 64, "context_size": 20, "target_return": 500.0})
```

- `choose_action` needs the context of the episode (the keyword arguments `reward`, `episode_start` and `episode_key`), which the agent's `evaluate` and `record_video` pass; without it, it raises a `NotImplementedError`, and without `target_return` a `ValueError`.
- During training, decision transformers are evaluated online (`episode_reward_mean`) only with `target_return`; the validation metrics on validation episodes (`action_match`, `td_error`) are not computed for them.
- Their ONNX export takes the target return as an input, see [ONNX export of agent policies](docs/onnx-export.md#decision-transformers).

### Features extractors

All agents accept a `features_extractor`: numpy preprocessing of the observations (`preprocess`) and a torch network (`forward`) in front of the agent's networks:

```
agent = StableBaselinesAgent(PPO, algorithm_parameters, features_extractor=MyFeaturesExtractor())
```

- Stable-Baselines 3, Imitation and d3rlpy agents train the features extractor together with their networks and save it with the model. For d3rlpy, the features extractor is put in front of the network of the configured encoder factory (default: 256x256 MLP). The discriminators of GAIL and AIRL get their own copy of the features extractor.
- `QLearning` does not train it: use a pretrained, frozen features extractor (a trainable one logs a warning). Its weights are saved with the Q-table.
- On load, the agent's features extractor must match the saved one (class, output size, parameter shapes), otherwise a `ValueError` is raised. Its preprocessing is not compared.

Details in [Features extractors](docs/features-extractors.md).

### Evaluation

Once you trained the agent, you can evaluate the agent policy on the environment and get the average accumulated reward (and standard deviation) as evaluation metric.
This evaluation method is implemented in the [evaluate function of the agent](src/rl_framework/agent/base_agent.py) and called with one line of code:

```
agent.evaluate(evaluation_environment=environment, n_eval_episodes=100, deterministic=False)
```

### Exporting the policy as ONNX model

The policies of the Stable-Baselines 3, Imitation and d3rlpy agents can be exported as ONNX model, e.g., for inference outside of Python:

```
agent.save_policy_as_onnx("policy.onnx", deterministic=True)
```

The exported inputs and outputs, the supported algorithms and further details are described in [ONNX export of agent policies](docs/onnx-export.md).

### Uploading and downloading models from a experiment registry

Once you trained the agent, you can upload the agent model to an experiment registry (HuggingFace Hub or ClearML) in order to share and compare your agent to others. You can also download yours or other agents from the same service and use them for solving environments or re-training.
The object which allows for this functionality is `HuggingFaceConnector` and `ClearMLConnector`, which can be found in the [connection collection package](src/rl_framework/util/connector).

### Examples

In [this RL example script](exploration/train_rl_agent.py) and in [this IL example script](exploration/train_il_agent.py) you can see all of the above steps unified.

## Development

### Notebooks

You can use your module code (`src/`) in Jupyter notebooks without running into import errors by running:

```
uv run --with jupyter jupyter notebook
```

or

```
uv run --with jupyter jupyter lab
```

This starts the jupyter server inside the project's virtualenv.

If you use a separately installed Jupyter, you can make your virtual environment available as a separate kernel by running:

```
uv add --dev ipykernel
uv run python -m ipykernel install --user --name="reinforcement-learning-framework"
```

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Testing

We use `pytest` as test framework. To execute the tests, please run

```
uv run pytest tests
```

To run the tests with coverage information, please use

```
uv run pytest tests --cov=src --cov-report=html --cov-report=term
```

Have a look at the `htmlcov` folder, after the tests are done.

The ONNX export tests evaluate the exported models with ONNX Runtime (`onnxruntime`, part of the `dev` dependency group), see [ONNX export of agent policies](docs/onnx-export.md#development-notes).

The tests run with the torch of the projects using this framework: `constraint-dependencies` in `pyproject.toml` locks torch 2.10 for Python >= 3.10 (torch 2.9 dropped Python 3.9, which gets torch 2.8). It only applies to this repository's lock, not to the published package. Update it together with those projects.

### Distribution Package

To release a new version, bump the version number, build the source distribution and wheel, and upload them to PyPI:

```
uv version --bump patch
uv build
uv publish
```

`uv build` writes the packages to the `dist` folder. `uv publish` uploads everything in `dist`, so remove old builds before building. It reads the PyPI token from the `UV_PUBLISH_TOKEN` environment variable (or pass it with `--token`).

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

```
uv run pre-commit install
```

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## License

© Alexander Zap
