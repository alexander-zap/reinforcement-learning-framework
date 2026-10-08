# Test suite and bug report (branch `test_suite`)

**Date:** 2026-10-08
**Base:** `save_as_onnx` @ 827c39b (rl_framework 0.11.4). Locked dependencies: stable-baselines3 2.7.1, imitation 1.0.0, d3rlpy 2.6.1, torch 2.7.1, gymnasium 0.29.1.
**Downstream reference:** `../synthetic-player-experiment-space` (usage survey below).

The test suite itself changed no source code (see "Fix status" for the follow-up fixes). Each bug has a **strict xfail** test that asserts the desired behavior and states the cause in its `reason`. A fix makes the test XPASS, which strict mode reports as a failure, so the marker has to be removed together with the fix.

## Summary

| | Before | After |
|---|---|---|
| Tests | 64 passed, 12 xfailed, 1 skipped | 253 passed, 38 xfailed, 1 skipped |
| Line coverage | 61% | 98% (97% line+branch) |
| Runtime (local, CPU) | ~3 min | ~3.5 min |

Run with `uv run pytest tests --cov=rl_framework --cov-branch`. Run with `--runxfail` to see every bug reproduce: each xfail fails at the line its `reason` names.

The uncovered rest is mostly abstract-method bodies (`raise NotImplementedError`), the wait-for-file `time.sleep` loops in the ClearML connector, and the dead `LoggingCallback._supports_episode_aggregation`.

## Fix status (follow-up, uncommitted)

Fixed in the working tree after this report; their xfail markers are removed. Now: 275 passed, 24 xfailed, 1 skipped.

| Finding | Fix |
|---|---|
| B-01 PettingZoo returns | `AutoResetSB3VecEnvWrapper` passes `MarkovVectorEnv`'s step results through (it already auto-resets when all agents are done) and only adds `TimeLimit.truncated` / `terminal_observation`. An agent finishing before the others still fails inside supersuit (`MarkovVectorEnv` asserts on missing agents), as before. |
| B-02 dropped imitation callbacks | `add_callbacks_to_callback` returns a new `CallbackList` (arguments unchanged); `AlgorithmWrapper.combine_with_library_callback` assigns it to `gen_callback` / `wrapper_callback`, remembering the library's own callback so retraining does not accumulate earlier callbacks. |
| B-03 SQIL | `patch_imitation_sqil_replay_buffer` (in `util/util.py`, applied on import like the other patches) concatenates only non-None sample fields. No dependency pin changed. |
| B-04 GAIL/AIRL load | Reward net loaded with `torch.load(..., weights_only=False)` (the file is written by the framework itself). |
| B-06, B-07 and the known stale-index finding | `MetricAggregator` resets the episode reward for every finished episode, checks `infos[done_index]`, and skips logging an empty reward window; `LoggingCallback` ignores discarded episodes for its logging frequency. |
| B-08 HuggingFace JSON | `json.dump(..., default=_numpy_to_json)` converts numpy scalars and arrays. |
| B-09 async shutdown | `shutdown()` in a `finally` block. |

### New test files

| File | Covers |
|---|---|
| `tests/toys.py`, `tests/conftest.py` | Shared toy envs (gym, PettingZoo), in-process agents, recording connector. A session fixture redirects `tempfile` so per-agent tensorboard dirs stay in pytest's tmp dir. |
| `test_stable_baselines_agent.py` | Parameters, training on gym / VecEnv / EnvironmentFactory / PettingZoo, checkpoints, retraining, features extractor, actions, save/load/download, ONNX |
| `test_async_stable_baselines_agent.py` | Injected algorithm, `IndexableMultiEnv`, utilization callback, worker shutdown |
| `test_custom_agent.py` | QLearning update rule, training, epsilon schedule, features extractor, save/load |
| `test_imitation_agent.py` | BC, GAIL, AIRL, Density, SQIL: training, connector logging, validation, PettingZoo, features extractor, save/load, cross-algorithm load, ONNX |
| `test_d3rlpy_agent.py` | Offline training, logging/checkpoints, validation, features extractor, save/load, ONNX, `ConnectorAdapter` |
| `test_episode_sequence.py` | Construction (episodes, generator, dataset), save/load, generic/d3rlpy conversion |
| `test_training_callbacks.py` | Logging/Saving/Pruning/ResetInfo/GammaSchedule callbacks, both `_on_step` and `process_episode` |
| `test_evaluate_environment_types.py` | `Agent.evaluate` for every environment type, preprocessing, logging |
| `test_base_connector.py`, `test_hugging_face_connector.py` (+ additions to `test_clearml_connector.py`) | Connector bookkeeping, HF upload/download and ClearML logging/upload/download with the services mocked |
| `test_features_extractor_utils.py`, `test_sb3_optimizer_reset.py`, `test_util_patches.py`, `test_video_recording.py`, `test_public_api.py` (+ additions to `test_metric_aggregator.py`) | The remaining utilities |

## New findings

Ranked by impact. Line numbers refer to 827c39b.

### High

**B-01: PettingZoo training trains on (and logs) wrong episode returns.**
`agent/reinforcement/stable_baselines.py:146-196` (`AutoResetSB3VecEnvWrapper.step_wait`)

- The wrapper reports every done one step late, so the step before the done is repeated.
- `MarkovVectorEnv` has already reset by then, so the first transition of the next episode is replaced by the old terminal transition.
- Result: with rewards 1..5 (true return 15), SB3 sees episodes of length 6 with return 19, then length 5 with return 18.
- PPO's advantages and the logged "Episode reward" are both wrong. Each episode's first real transition is never trained on, and the action taken on the repeated observation is attributed to the wrong episode.
- Test: `test_stable_baselines_agent.py::test_pettingzoo_training_sees_the_true_episode_returns`.

**B-02: GAIL, AIRL and Density training never call the framework callbacks.**
`util/sb3_training_callbacks.py:21-29`; used at `gail.py:82`, `airl.py:82`, `density.py:64`.

- `add_callbacks_to_callback` wraps a `None` or single-callback target in a new local `CallbackList` and returns nothing.
- In imitation 1.0, `gen_callback` and `wrapper_callback` are single callbacks, so `SavingCallback` and `LoggingCallback` are silently dropped.
- With any connector, these algorithms log no episode metrics and upload no checkpoints.
- Tests: `test_training_callbacks.py::test_add_callbacks_to_a_single_callback_makes_them_reachable`, `test_imitation_agent.py::test_environment_interacting_algorithms_log_episodes_to_the_connector[GAIL|AIRL|DensityAlgorithm]`.

**B-03: SQIL cannot train with the locked dependencies.**
This is a dependency incompatibility, not rl_framework code.

- stable-baselines3 2.7 added `ReplayBufferSamples.discounts`, which is `None` for plain buffers.
- imitation 1.0's `SQILReplayBuffer.sample` `th.cat`s every field, so the first gradient step raises `TypeError`.
- `pyproject.toml` allows `stable-baselines3>=2.2,<3`, and `uv.lock` resolves to 2.7.1.
- Fix options: pin SB3 below the version that added `discounts`, or patch `SQILReplayBuffer.sample` the way `util/util.py` patches other libraries.
- Test: `test_imitation_agent.py::test_every_algorithm_trains_and_acts[SQIL]`. The other SQIL tests set `learning_starts` past the run length so that saving and logging stay covered.

**B-04: Saved GAIL and AIRL agents cannot be loaded.**
`gail.py:96`, `airl.py:96`

- The reward net is saved as a pickled module (`torch.save(algorithm._reward_net)`) and loaded with `torch.load(path)`.
- Since torch 2.6, `torch.load` defaults to `weights_only=True`, so loading raises `UnpicklingError`.
- `load_from_file` only catches `FileNotFoundError`, so the whole load fails, including the policy.
- Test: `test_imitation_agent.py::test_adversarial_save_and_load_round_trip_keeps_the_policy[GAIL|AIRL]`.

### Medium

**B-05: Retraining an SB3 agent with a features extractor and user `policy_kwargs` crashes.**
`stable_baselines.py:249-261`

- The second `train()` saves and reloads with `custom_objects=self.algorithm_parameters`.
- The user's `policy_kwargs` (without the features extractor entries) replace the saved ones, so the rebuilt policy has no extractor.
- Result: `RuntimeError: Error(s) in loading state_dict`.
- Without user `policy_kwargs` it works.
- Test: `test_stable_baselines_agent.py::test_retraining_with_features_extractor_and_policy_kwargs_works`.

**B-06: Discarded episodes crash `LoggingCallback` or log NaN.**
`sb3_training_callbacks.py:139-152` together with `metric_logging_utils.py:79`

- `_log_if_due` also runs for discarded episodes.
- If an env's first episode is discarded, `episode_rewards[index]` does not exist yet, so it raises `KeyError`.
- After a logging reset, a discarded episode logs `np.mean([]) = NaN` as "Episode reward".
- This hits async training with `skip_truncated` or worker discards, which downstream uses.
- Tests: `test_training_callbacks.py::test_logging_callback_survives_a_discarded_first_episode`, `::test_logging_callback_logs_no_nan_reward_for_a_discarded_episode`.

**B-07: A discarded episode's reward is added to the next episode.**
`metric_logging_utils.py:64-70`

- `episode_reward[done_index]` is only reset for kept episodes.
- This is separate from the known stale-index bug at line 66.
- Test: `test_metric_aggregator.py::test_discarded_episode_reward_does_not_leak_into_next_episode`.

**B-08: The HuggingFace final upload fails after normal training.**
`hugging_face_connector.py:144-147`

- `json.dump` of the logged sequences fails on numpy values.
- SB3 VecEnvs return float32 rewards, so `MetricAggregator` logs `np.float32` means. Histograms are numpy arrays.
- Test: `test_hugging_face_connector.py::test_final_upload_serializes_numpy_values_logged_during_training`.

**B-09: `AsyncStableBaselinesAgent` leaves workers running when training fails.**
`async_stable_baselines.py:152-156`

- `shutdown()` is not in a `finally` block.
- Downstream this is the known "process never exits after AsyncWorkerFailureError" hang.
- Test: `test_async_stable_baselines_agent.py::test_train_shuts_down_workers_when_training_fails`.

**B-10: `D3RLPYAgent` crashes on short training runs.**
`d3rlpy.py:315-326`

- `logging_steps = ceil(10000 / batch_size)` is used even when `n_steps` is smaller, and d3rlpy's fitter then reads `metrics` before assignment (`UnboundLocalError`).
- With batch size 32, any `total_timesteps < 10016` fails.
- Test: `test_d3rlpy_agent.py::test_short_training_runs`.

**B-11: `D3RLPYAgent` evaluation metrics are lost.**
`d3rlpy.py:306-326`

- The agent uses `LoggingStrategy.STEPS`, but d3rlpy adds evaluator scores (`episode_reward_mean`, `td_error`) after the epoch's last step commit.
- They are therefore logged in the next epoch at the wrong step, and never for the last epoch.
- Runs under about 50k timesteps are a single epoch and log no evaluation metrics at all.
- Test: `test_d3rlpy_agent.py::test_evaluation_metrics_are_logged_to_the_connector`.

**B-12: `D3RLPYAgent` with validation episodes crashes for value-free algorithms.**
`d3rlpy.py:309-313`

- A `TDErrorEvaluator` is always added, but (Discrete)BC raises `NotImplementedError: BC does not support value estimation`.
- Test: `test_d3rlpy_agent.py::test_behavior_cloning_trains_with_validation_episodes`.

**B-13: `D3RLPYAgent` cannot train on PettingZoo environments.**
`d3rlpy.py:272-307`

- The vectorized env is built but never used: the raw `ParallelEnv` is passed to `ReplayBuffer` and `EnvironmentEvaluator`.
- Result: `ValueError: Unsupported observation type: <class 'dict'>`.
- Test: `test_d3rlpy_agent.py::test_training_on_pettingzoo_environment`.

**B-14: BC validation runs out of batches.**
`imitation/algorithms/bc.py:89-91,127`

- One iterator over the validation data is used for the whole training.
- When training logs more often than there are validation batches, `next()` raises `StopIteration` and aborts training. Example: `log_interval=1` with 500 validation transitions at batch size 32 fails at batch 16.
- Test: `test_imitation_agent.py::test_bc_validation_does_not_run_out_of_batches`.

**B-15: The ClearML histogram logging leaks matplotlib figures.**
`clearml_connector.py:106-113`

- Every histogram creates a seaborn `FacetGrid` figure that is never closed.
- With `callback_log_distributions`, memory grows during training and matplotlib warns after 20 open figures.
- Test: `test_clearml_connector.py::test_log_histogram_does_not_leak_matplotlib_figures`.

### Low

**B-16: The QLearning epsilon schedule is broken.**
`q_learning.py:201-205`

- Epsilon is set to `1 - 2t/T` while it is above `epsilon_min`. This overshoots below `epsilon_min`, even below 0 (−0.2 in the test).
- It also ignores the configured `epsilon`: 0.2 jumps to 0.8 after the first episode.
- Tests: `test_custom_agent.py::test_epsilon_never_falls_below_epsilon_min`, `::test_configured_initial_epsilon_is_not_increased`.

**B-17: `QLearning.save_to_file("q.pkl")` fails.**
`q_learning.py:221`

- `os.makedirs("")` fails for a bare file name.
- Test: `test_custom_agent.py::test_save_to_a_bare_file_name_in_the_working_directory`.

**B-18: Video recording breaks on paths with spaces.**
`video_recording.py:66`

- The ffmpeg command is built without quoting. The failure is only logged (lines 70-71), and the upload continues without a video.
- Test: `test_video_recording.py::test_sb3_recording_writes_the_video_to_a_path_with_spaces`.

**B-19: SB3 ONNX export fails for Dict observation spaces.**
`stable_baselines.py:345-347`

- `observation_space.shape` is `None`, which raises `TypeError`.
- Test: `test_stable_baselines_agent.py::test_onnx_export_supports_dict_observations`.

**B-20: The SB3 ONNX export is stochastic.**
`stable_baselines.py:341`

- This is already marked FIXME in the code.
- The exported graph samples actions with a `Multinomial` node.
- Test: `test_stable_baselines_agent.py::test_onnx_export_is_deterministic`.

## Previously known findings (existing xfails, still reproducing)

These are unchanged. Their `reason` strings cite line numbers from an older revision (current lines in brackets):

- `MetricAggregator` checks `infos[agent_index]` instead of `infos[done_index]` \[metric_logging_utils.py:66\].
- `evaluate` episode quota is `n // len(envs) + 1` per env \[base_agent.py:225\].
- `evaluate` env threads log colliding series \[base_agent.py:206-213\].
- `evaluate` never resets step-metric windows \[base_agent.py:206-213\].
- `load_from_file` passes `callback_kwargs` and `reset_optimizer` to SB3 instead of the agent \[stable_baselines.py:366-370\].
- `reset_optimizer` on a fresh agent hits the placeholder algorithm \[stable_baselines.py:230-232\].
- `SavingCallback.process_episode` uploads once per crossed checkpoint \[sb3_training_callbacks.py:227-236\].
- Async callbacks attribute all workers to agent 0 \[sb3_training_callbacks.py:108, 367-374\].
- ClearML multi-dot file names \[clearml_connector.py:170\].
- ClearML `file_name[:-4]` extension assumption \[clearml_connector.py:245\].

Several downstream findings (U-07, U-08, U-09) are fixed on this branch, and tests guard them: `test_features_extractor.py`, `test_imitation_learning_agents_are_instantiable`, and the PettingZoo wrapper tests.

## Limitations and observations (no xfail)

These are documented or deliberate behaviors rather than bugs, but worth deciding on:

- **Features extractor on VecEnv or EnvironmentFactory:** `wrap_environment_with_features_extractor_preprocessor` raises `TypeError`, so `train` and `evaluate` with a features extractor only work for gym and PettingZoo envs. That includes the downstream Cold War factory format.
- **Imitation checkpoint interval is fixed:** `ImitationAgent` uses `SavingCallback(self, connector)` with the default 50,000-step interval and has no `callback_kwargs`.
- **Video errors are swallowed:** `record_video` logs and swallows all exceptions, so a failing policy produces an upload without a video and no error.
- **`D3RLPYAgent.save_to_file` needs a `Path`:** it calls `file_path.as_posix()` and fails for `str`. The other agents accept both.
- **`D3RLPYAgent.choose_action` guard is dead:** `if not self.algorithm` is never true because the algorithm is created in `__init__`. An untrained agent raises d3rlpy's own error.
- **Dead code:** `LoggingCallback._supports_episode_aggregation` is unused.
- **Optimizer reset repeats:** `reset_optimizer=True` resets the optimizer on every `train()` call, not only on the first one after loading.

## Downstream relevance (synthetic-player-experiment-space)

The experiment space trains through config with `StableBaselinesAgent` / `AsyncStableBaselinesAgent` (PPO, MultiDiscrete actions, EnvironmentFactory tuples), `CustomAgent`+`QLearning` (Taxi) and `ImitationAgent`. It uses ClearML and HuggingFace connectors and `step_metric_*` infos.

- **Directly reachable with shipped configs:**
  - B-01: `configs/unity/unity_configuration.yaml` trains SB3 PPO on a `UnityPettingZoo` ParallelEnv.
  - B-08: `configs/general/default_config.yaml` uses SB3 PPO with the HuggingFace connector.
  - B-06 / B-07: async discards in Cold War AsyncSB3 runs give NaN or crashing episode-reward logs and wrong rewards.
  - B-09: hang after a worker failure.
  - B-15: when distributions are logged.
  - The previously known evaluate and callback-attribution items.
- **Reachable by config choice (commented-out alternatives in `default_config.yaml`):**
  - B-02 / B-03 / B-04 / B-14: the imitation enum entries for GAIL, AIRL, Density, SQIL and BC with validation.
  - B-16: `Custom` + `Q_LEARNING`.
  - B-05: `features_extractor` with `policy_kwargs`, plus download-and-retrain.
- **Not used downstream:**
  - `D3RLPYAgent` (B-10 to B-13).
  - The rl_framework ONNX export (B-19, B-20). Downstream exports ONNX elsewhere.
