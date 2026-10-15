# ONNX export of agent policies

`StableBaselinesAgent` (including `AsyncStableBaselinesAgent`) and `ImitationAgent` export their policy with

```python
agent.save_policy_as_onnx("policy.onnx")                       # deterministic (default)
agent.save_policy_as_onnx("policy.onnx", deterministic=False)  # stochastic, sampling from noise inputs
```

Both use `export_sb3_policy_as_onnx` in [`rl_framework/util/onnx_export.py`](../src/rl_framework/util/onnx_export.py).
`D3RLPYAgent` exports its algorithms in the same contract with `export_d3rlpy_algorithm_as_onnx` (see
[d3rlpy algorithms](#d3rlpy-algorithms)); its decision transformers need further inputs. The `ClearMLConnector`
uploads this export (with the default arguments) as artifact `<model name>_ONNX` together with every model and
checkpoint. `CustomAgent` has no ONNX export.

## Inputs

All inputs are float32 and have a dynamic batch axis `batch`. In this order:

| Input | Shape | Description |
| --- | --- | --- |
| `obs` | (batch, \*observation shape) | The observation. Dict observation spaces: one input per key, named like the key. Discrete, MultiDiscrete and image observations are fed as float, too: the graph preprocesses them like SB3 (one-hot encoding, image normalization). Observations are laid out like the policy's observation space, e.g., images channel-first (SB3 transposes channel-last images in `predict` and, with `VecTransposeImage`, in training). |
| `action_masks` | (batch, number of action logits) | MaskablePPO only: 1 for allowed, 0 for masked actions, like SB3-contrib's `action_masks`. All ones: no masking. |
| `episode_starts` | (batch,) | RecurrentPPO only: 1 at the first step of an episode (resets the LSTM states). |
| `lstm_actor_h`, `lstm_actor_c` | (n_lstm_layers, batch, lstm_hidden_size) | RecurrentPPO only: the actor's LSTM states (**batch axis 1**). Zeros at the start, afterward the `*_out` outputs of the previous step. |
| `lstm_critic_h`, `lstm_critic_c` | (n_lstm_layers, batch, lstm_hidden_size) | RecurrentPPO with a separate critic LSTM (`enable_critic_lstm`, the default) only, like the actor's states. |
| `noise_u` | (batch, number of action logits or binary actions) | Stochastic export of discrete actions only: Uniform(0, 1) noise. Discrete/MultiDiscrete: one value per action logit, sampled with the Gumbel-max trick (the noise is clamped to \[1e-6, 1 - 1e-6\]). MultiBinary: one value per binary action, which is 1 if the noise is below its probability. |
| `noise_eps` | (batch, action dimension) | Stochastic export of Box actions only: Normal(0, 1) noise, sampled by reparameterization (`mean + std * noise_eps`). |

The stochastic export takes the noise as inputs instead of using random number nodes, so that sampling is
reproducible and the consumer controls it. A deterministic export has no noise input.

## Outputs

All outputs have a dynamic batch axis `batch` (axis 1 for the LSTM states).

| Output | Shape | Policies | Description |
| --- | --- | --- | --- |
| `action` | (batch, number of action heads) or (batch, \*action shape) | all | Discrete, MultiDiscrete and MultiBinary actions as int64, one column per action head, i.e. (batch, 1) for Discrete. Box actions postprocessed like SB3's `predict`: clipped to the action space, or unscaled from \[-1, 1\] for squashing policies (e.g., SAC, TD3). |
| `values` | (batch, 1) | actor-critic (PPO, A2C, TRPO, MaskablePPO, RecurrentPPO) | State values. |
| `log_prob` | (batch,) | actor-critic; SAC, TQC, CrossQ | Log-probability of the action: for actor-critic policies of the unpostprocessed action (like SB3's `policy.forward`), for SAC, TQC and CrossQ of the squashed action. |
| `q_values` | (batch, number of actions) | DQN, QR-DQN | Q-values (QR-DQN: the mean over the quantiles). |
| `lstm_actor_h_out`, `lstm_actor_c_out` (`lstm_critic_h_out`, `lstm_critic_c_out`) | (n_lstm_layers, batch, lstm_hidden_size) | RecurrentPPO | The new LSTM states, to feed into the next step. |

The graph includes the observation preprocessing and the features extractor's network (`forward`), but not the
features extractor's `preprocess`, which is applied to the environment's observations before the policy.

## Supported StableBaseline3/Imitation policies

| Policies | Action spaces | Stochastic export |
| --- | --- | --- |
| PPO, A2C, TRPO, MaskablePPO, RecurrentPPO (actor-critic) | Discrete, MultiDiscrete, MultiBinary, Box | yes |
| SAC, TQC, CrossQ (squashed Gaussian actor) | Box | yes |
| TD3, DDPG, DQN, QR-DQN, ARS | Box (TD3, DDPG), Discrete (DQN, QR-DQN), Box or Discrete (ARS) | no: their exploration is added by the algorithm, not the policy. `deterministic=False` raises a `ValueError`. |

This covers the policies of the imitation agents, too (ActorCriticPolicy, DQNPolicy, SACPolicy). Other policy types
raise a `NotImplementedError`; the `ClearMLConnector` then uploads the model without ONNX artifact.

Policies trained with **gSDE** (`use_sde=True`) use the same contract as other Box policies. The stochastic export
samples from gSDE's per-step distribution: a Gaussian with the state-dependent standard deviation
`sqrt(latent_sde² @ std²)`, i.e. like gSDE resampling its exploration matrix at every step (`sde_sample_freq=1`). This is
exact per step; only the temporal correlation of an exploration matrix kept over several steps (smooth exploration in
training) is not reproduced.

**RecurrentPPO** is exported for one step per call: start with zero LSTM states and `episode_starts` = 1, then feed
the `*_out` states back as inputs.

## d3rlpy algorithms

### Q-learning algorithms

`D3RLPYAgent` exports the greedy policy of its Q-learning algorithms (all registered algorithms except the decision
transformers and the random policies) in the contract above, deterministic only (`deterministic=False` raises a
`ValueError`):

- Input `obs`: the observation, before d3rlpy's observation scaler (the graph applies it). Tuple observations are not
  supported.
- Output `action`: like the algorithm's `predict` (the action scaler reversed): discrete actions as int64 (batch, 1),
  continuous actions as float (batch, action dimension), clipped to the action space (see below).
- Output `q_values` (batch, number of actions): for the algorithms with a discrete Q-function (DQN, DoubleDQN, NFQ,
  DiscreteCQL, DiscreteBCQ, DiscreteSAC), the mean over the Q-function ensemble, like `predict_value`. The action is
  their argmax, except for DiscreteBCQ (the best action its imitator allows) and DiscreteSAC (the most likely action
  of its policy).
- BCQ, BEAR and CRR choose their action among sampled candidates, so their graph contains random nodes (like
  d3rlpy's `predict`, which differs between calls).

`RandomPolicy` and `DiscreteRandomPolicy` have no networks and raise a `NotImplementedError` (the `ClearMLConnector`
then uploads the model without ONNX artifact).

d3rlpy knows only the action size, not the action space's bounds. `D3RLPYAgent` keeps the action space of its training
environment (`action_space`), stores it in its saved file (an additional key `action_space` of d3rlpy's file, which
`d3rlpy.load_learnable` still loads) and clips continuous actions to it, in `choose_action` and in the export (like
SB3's `predict`). Files saved before have no action space: their continuous actions are not clipped, and the export
logs a warning.

### Decision transformers

`DecisionTransformer` and `DiscreteDecisionTransformer` condition their action on the last `context_size` steps
(observations, actions, returns-to-go and timesteps) and on the return still to achieve. They are exported for one
step per call, like d3rlpy's `StatefulTransformerWrapper` (and like RecurrentPPO with its LSTM states): the context
window is an input, and its update an output to feed into the next step. Unlike the other exports, they need the
reward and the target return, so a consumer of the other exports cannot run them unchanged.

Inputs, in this order (all float32, batch axis 0):

| Input | Shape | Description |
| --- | --- | --- |
| `obs` | (batch, \*observation shape) | The observation, before d3rlpy's observation scaler. |
| `reward` | (batch,) | The reward of the previous step; 0 at the episode start. |
| `target_return` | (batch,) | The return to achieve; read at the episode start only. In the framework, the `D3RLPYAgent` parameter `target_return`. |
| `episode_starts` | (batch,) | 1 at the first step of an episode: resets the context window, the return still to achieve (to `target_return`) and the timestep (to 1). |
| `dt_observations` | (batch, context_size, \*observation shape) | Context window (unscaled): zeros at the start, afterward the `*_out` outputs of the previous step. |
| `dt_actions` | (batch, context_size, 1 or action dimension) | Actions of the context (discrete: the action index as float). |
| `dt_returns_to_go` | (batch, context_size, 1) | Returns-to-go of the context. |
| `dt_timesteps` | (batch, context_size) | Timesteps of the context (1 at the episode start, at most `max_timestep`). |
| `dt_return_rest` | (batch,) | The return still to achieve. |
| `dt_timestep` | (batch,) | The timestep of the next step. |
| `noise_u` | (batch, number of actions) | Stochastic export of discrete decision transformers only: Uniform(0, 1) noise for the Gumbel-max trick. |

Outputs: `action` (discrete: int64 (batch, 1), the argmax of the logits or sampled from `noise_u` like d3rlpy's
`SoftmaxTransformerActionSampler`, the default of `as_stateful_wrapper`; continuous: (batch, action dimension) like
`IdentityTransformerActionSampler`, the action scaler reversed and clipped to the action space; the context window
keeps the unclipped action, like the stateful wrapper) and `<input>_out` for every `dt_*` input. The observation, action and reward scalers are applied to the context window in the graph, like d3rlpy's
`predict`. Continuous decision transformers are deterministic only (`deterministic=False` raises a `ValueError`).

## Export details

- `torch.onnx.export` with opset 17, zero tensors as dummy inputs, and dynamic batch axes.
- The policy is exported in evaluation mode (e.g., for CrossQ's BatchRenorm layers); its training mode is restored
  afterward. d3rlpy's modules stay in evaluation mode, like after d3rlpy's `predict`.
- The legacy TorchScript exporter (`dynamo=False`): the dynamo exporter, the default since torch 2.9, cannot map the
  dynamic axes to the wrapper's variadic `forward(*inputs)`. d3rlpy's own `save_policy` (d3rlpy's format, not used by
  the agent) is patched to use it, too (`patch_d3rlpy_onnx_export`).
- There is no metadata in the file and no parity check at export time; the parity with the policies is tested (see
  below).

## Development notes

- Consumers: the `ClearMLConnector` uploads the export with the default arguments, so a change of the defaults or the
  contract changes every uploaded `_ONNX` artifact.
- Tests: [`tests/test_onnx_export.py`](../tests/test_onnx_export.py) checks the contract and the parity with the SB3 and
  SB3-contrib policies (actions, values, log-probabilities, Q-values, masks, noise, gSDE, several RecurrentPPO steps)
  and with d3rlpy's `predict` and `predict_value` for every registered Q-learning algorithm (decision transformers:
  with `StatefulTransformerWrapper` over several steps, with a sliding context window), and the agent tests check
  the export through the agents. The exported graphs are evaluated with ONNX Runtime
  (`onnxruntime`, dev dependency; capped per Python version in `pyproject.toml`: 1.24 dropped Python 3.10, 1.20
  dropped Python 3.9).
- `onnx.reference.ReferenceEvaluator` (onnx 1.17) is no alternative for all graphs: it fails on negative
  `GatherElements` axes (the export uses positive axes) and its `LSTM` does not return the cell state.
- Build the models in tests in-process (e.g., `PPO("MlpPolicy", env)`). Scripts which train through the agents with
  `SubprocVecEnv` need a `if __name__ == "__main__":` guard on Windows.
