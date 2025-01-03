# Reinforcement Learning Framework

An easy-to-read Reinforcement Learning (RL) framework. Provides standardized interfaces and implementations to various Reinforcement Learning methods and environments. Also, this is the main place to start your journey with Reinforcement Learning and learn from tutorials and examples.

### Main Features

- Choose from a growing number of **Gym environments** and **MLAgent environments**
- Using various Reinforcement Learning algorithms for learning, which are implemented in **Stable-Baselines 3**
- Integrate or implement own **custom environments and agents** in a standardized interface
- Upload your models to the **HuggingFace Hub**

## Set-Up

### Activate your development environment

If you are on a UNIX-based OS:
You are fine. Continue with the next step.

If you are on Windows:
Make sure to use a WSL Python interpreter as your development environment, since we require a UNIX-based system underneath Python to run a lot of the environments and algorithms.
For users using PyCharm, see https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html for more information.
For users using Visual Studio Code, see https://code.visualstudio.com/docs/remote/wsl-tutorial and https://code.visualstudio.com/docs/remote/wsl for more information.

### Install all dependencies in your development environment

To set up your local development environment, please run:

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

To integrate your environment you wish to train on, you need to create an Environment class representing your problem. For this you can

- you use an existing Gym environment with [the `GymEnvironment` class](src/rl_framework/environment/gym_environment.py)
- you use an existing MLAgent environment with [the `MLAgentsEnvironment` class](src/rl_framework/environment/mlagents_environment.py)
- create a custom environment by inheriting from [the base `Environment` class](src/rl_framework/environment/base_environment.py), which specifies the required interface

### Configuring an agent

To integrate the Reinforcement Learning algorithm you wish to train an agent on your environment with, you need to create an Agent class representing your training agent. For this you can

- you use an existing Reinforcement Learning algorithm implemented in the Stable-Baselines 3 framework with [the `StableBaselinesAgent` class](src/rl_framework/agent/stable_baselines.py) (see the Example section below)
- create a custom Reinforcement Learning algorithm by inheriting from [the base `BaseAgent` class](src/rl_framework/agent/base_agent.py), which specifies the required interface

### Training

After configuring the environment and the agent, you can start training your agent on the environment.
This can be done in one line of code:

```
agent.train(environments=environments, total_timesteps=100000)
```

Independent of which environment and which agent you choose, the unified interface allows to always start the training this way.

### Evaluating

Once you trained the agent, you can evaluate the agent policy on the environment and get the average accumulated reward (and standard deviation) as evaluation metric.
This evaluation method is implemented in the [evaluate function of the agent](src/rl_framework/agent/base_agent.py) and called with one line of code:

```
agent.evaluate(evaluation_environment=environment, n_eval_episodes=100, deterministic=False)
```

### Uploading and downloading models from the HuggingFace Hub

Once you trained the agent, you can upload the agent model to the HuggingFace Hub in order to share and compare your agent to others. You can also download yours or other agents from the same HuggingFace Hub and use them for solving environments or re-training.
The object which allows for this functionality is `HuggingFaceConnector`, which can be found in the [connection collection package](src/rl_framework/util/saving_and_loading/connector).

### Example

In [this example script](exploration/train_sb3_agent.py) you can see all of the above steps unified.

For a quick impression in this README, find a minimal training and evaluation example here:

```
# Create environment(s); multiple environments for parallel training
environments = [GymEnvironmentWrapper(ENV_ID) for _ in range(PARALLEL_ENVIRONMENTS)]

# Create new agent
agent = StableBaselinesAgent(
    algorithm=StableBaselinesAlgorithm.PPO,
    algorithm_parameters={
        "policy": "MlpPolicy"
    }
)
# Train agent
agent.train(environments=environments, total_timesteps=100000)

# Evaluate the model
mean_reward, std_reward = agent.evaluate(evaluation_environment=environments[0])
```

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

and have a look at the `htmlcov` folder, after the tests are done.

### Distribution Package

To build a distribution package (wheel), please use

```
python setup.py bdist_wheel
```

this will clean up the build folder and then run the `bdist_wheel` command.

### Contributions

Before contributing, please set up the pre-commit hooks to reduce errors and ensure consistency

```
pip install -U pre-commit
pre-commit install
```

If you run into any issues, you can remove the hooks again with `pre-commit uninstall`.

## License

© Alexander Zap
