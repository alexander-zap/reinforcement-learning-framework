# Reinforcement Learning Framework

An easy-to-read Reinforcement Learning (RL) framework. Provides standardized interfaces and implementations to various Reinforcement Learning and Imitation Learning methods and utilities.

### Main Features

- Using various Reinforcement Learning algorithms to learn from gym environment interaction, which are implemented in **Stable-Baselines 3**
- Using various Imitation Learning algorithms to learn from replays, which are implemented in **Imitation**
- Integrate or implement own **custom agents and algorithms** in a standardized interface
- Upload your models (with logged metrics, checkpoints and video recordings) to **HuggingFace Hub** or **ClearML**

## Set-Up

### Install all dependencies in your development environment

To set up your local development environment, please install poetry (see (tutorial)\[https://python-poetry.org/docs/\]) and run:

```
poetry install
```

Behind the scenes, this creates a virtual environment and installs `rl_framework` along with its dependencies into a new virtualenv. Whenever you run `poetry run <command>`, that `<command>` is actually run inside the virtualenv managed by poetry.

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

### Evaluation

Once you trained the agent, you can evaluate the agent policy on the environment and get the average accumulated reward (and standard deviation) as evaluation metric.
This evaluation method is implemented in the [evaluate function of the agent](src/rl_framework/agent/base_agent.py) and called with one line of code:

```
agent.evaluate(evaluation_environment=environment, n_eval_episodes=100, deterministic=False)
```

### Uploading and downloading models from a experiment registry

Once you trained the agent, you can upload the agent model to an experiment registry (HuggingFace Hub or ClearML) in order to share and compare your agent to others. You can also download yours or other agents from the same service and use them for solving environments or re-training.
The object which allows for this functionality is `HuggingFaceConnector` and `ClearMLConnector`, which can be found in the [connection collection package](src/rl_framework/util/connector).

### Examples

In [this RL example script](exploration/train_rl_agent.py) and in [this IL example script](exploration/train_il_agent.py) you can see all of the above steps unified.

## Development

### Notebooks

You can use your module code (`src/`) in Jupyter notebooks without running into import errors by running:

```
poetry run jupyter notebook
```

or

```
poetry run jupyter-lab
```

This starts the jupyter server inside the project's virtualenv.

Assuming you already have Jupyter installed, you can make your virtual environment available as a separate kernel by running:

```
poetry add ipykernel
poetry run python -m ipykernel install --user --name="reinforcement-learning-framework"
```

Note that we mainly use notebooks for experiments, visualizations and reports. Every piece of functionality that is meant to be reused should go into module code and be imported into notebooks.

### Testing

We use `pytest` as test framework. To execute the tests, please run

```
pytest tests
```

To run the tests with coverage information, please use

```
pytest tests --cov=src --cov-report=html --cov-report=term
```

Have a look at the `htmlcov` folder, after the tests are done.

### Distribution Package

To build a distribution package (wheel), please use

```
python setup.py bdist_wheel
```

This will clean up the build folder and then run the `bdist_wheel` command.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

```
pip install -U pre-commit
pre-commit install
```

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## License

© Alexander Zap
