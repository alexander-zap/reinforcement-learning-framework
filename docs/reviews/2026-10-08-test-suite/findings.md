# Test suite and bug report (branch `test_suite`)

**Date:** 2026-10-08 (status updated 2026-10-12)
**Base:** `save_as_onnx` @ 827c39b (rl_framework 0.11.4). Locked dependencies: stable-baselines3 2.7.1, imitation 1.0.0, d3rlpy 2.6.1, torch 2.7.1, gymnasium 0.29.1, async-gym-agents 0.5.0.
**Downstream reference:** `../synthetic-player-experiment-space` (usage survey below).

The test suite itself changed no source code. Each bug got a **strict xfail** test that asserts the desired behavior. A fix makes the test XPASS, which strict mode reports as a failure, so the marker was removed together with each fix.

**Status: every finding below is fixed and no xfail is left.** The remaining open items are listed under "Open items".

## Summary

| | Before | Report (`test_suite`) | Now |
|---|---|---|---|
| Tests | 64 passed, 12 xfailed, 1 skipped | 253 passed, 38 xfailed, 1 skipped | 359 passed, 0 xfailed, 1 skipped |
| Line coverage | 61% | 98% (97% line+branch) | 98% (97% line+branch) |
| Runtime (local, CPU) | ~3 min | ~3.5 min | ~4 min |

Run with `uv run pytest tests --cov=rl_framework --cov-branch`.

The uncovered rest is mostly abstract-method bodies (`raise NotImplementedError`), the wait-for-file `time.sleep` loops in the ClearML connector, and the dead `LoggingCallback._supports_episode_aggregation`.

## Fix status

"#54" = merged with the test suite (PR #54). "Follow-up" = branch `findings_bug_fixes`.

| Finding | Where | Fix |
|---|---|---|
| B-01 PettingZoo returns | #54 | `AutoResetSB3VecEnvWrapper` passes `MarkovVectorEnv`'s step results through (it already auto-resets when all agents are done) and only adds `TimeLimit.truncated` / `terminal_observation`. Agents which restart on their own follow gymnasium's auto-reset convention: the done step returns the next episode's start observation and passes the finished episode's last observation as `infos[agent]["final_observation"]` (documented in `StableBaselinesAgent.train` and `Agent.evaluate`). Verified on Unity builds (TanksGym, 3DBall, Walker) with the matching ml-agents-envs fork change. |
| B-02 dropped imitation callbacks | #54 | `AlgorithmWrapper.set_additional_callbacks` sets the framework callbacks as `gen_callback` / `wrapper_callback` next to the library's own callback; retraining does not accumulate earlier callbacks. |
| B-03 SQIL | #54 | `patch_imitation_sqil_replay_buffer` (in `util/util.py`) concatenates only non-None sample fields. No dependency pin changed. |
| B-04 GAIL/AIRL load | #54 | Reward net loaded with `weights_only=False` (the file is written by the framework itself). |
| B-05 retraining with extractor and `policy_kwargs` | #54 | Framework parameters are separated centrally; on load the saved `policy_kwargs` and features extractor are compared with the requested ones, and a mismatch raises a `ValueError` instead of a state-dict error. |
| B-06, B-07, stale `infos` index | #54 | `MetricAggregator` resets the episode reward for every finished episode, checks `infos[done_index]`, and skips logging an empty reward window; `LoggingCallback` ignores discarded episodes for its logging frequency. |
| B-08 HuggingFace JSON | #54 | numpy scalars and arrays are converted on `json.dump`; other unserializable values still raise. |
| B-09 async shutdown | #54 | `shutdown()` in a `finally` block. |
| B-10 to B-13 d3rlpy | Follow-up | Logging and epoch lengths are capped at the training length. Evaluators run in an epoch callback, so their metrics are logged at the end of every epoch. Validation logs `action_match` for every algorithm and `td_error` only for value-based ones. PettingZoo environments only provide the spaces (no online evaluation, with a warning); replay buffers take the action space from the environment. |
| B-14 BC validation batches | Follow-up | Validation batches cycle endlessly; validation data smaller than one batch raises a `ValueError` up front. |
| B-15 ClearML figure leak | Follow-up | The histogram's own figure is reported and closed. |
| B-16 QLearning epsilon | Follow-up | Epsilon decays linearly from its current value to `epsilon_min` at half of the training, never rises, and continues across `train()` calls. |
| B-17 QLearning bare file name | Follow-up | A directory is created only if the path has one. |
| B-18 video path with spaces | Follow-up | ffmpeg runs with an argument list; failures are logged with stderr. |
| B-19, B-20 SB3 ONNX | Follow-up | Dict observations become one input per key; the export is deterministic by default (`deterministic=True`); outputs are named `actions` / `values` / `log_prob`. |
| Evaluation quota (`n // len(envs) + 1`) | #54 | Episodes are split evenly over environments and PettingZoo agents (`split_evenly`). |
| Evaluation logging (colliding series, windows never reset) | Follow-up | Evaluation logs at the lock-protected count of evaluated episodes and resets the step-metric windows after each log. |
| `load_from_file` passes `callback_kwargs` / `reset_optimizer` to SB3 | #54 | Framework parameters are taken out before the SB3 load and applied to the agent. |
| `reset_optimizer` on a fresh agent | #54 | Applied to the loaded/trained algorithm, not the placeholder. |
| `SavingCallback.process_episode` uploads per crossed checkpoint | Follow-up | One upload per episode at the first crossed checkpoint; the schedule then continues after the episode end. |
| Async callbacks attribute all workers to agent 0 | #54 | Episodes are attributed per (worker, sub-environment) via `EpisodeCallbackContext.worker_index` / `env_index` (async-gym-agents 0.5.0); `LoggingCallback` aggregates metrics per agent. |
| ClearML multi-dot names / `file_name[:-4]` | Follow-up | Names use `os.path.splitext`; uploads wait for the file SB3 saves with `.zip` appended; download by task matches the name without extension. |

### Found while fixing (all fixed in #54)

- **Evaluation dropped one-step episodes after an agent restart.** They count now when `final_observation` is passed.
- **Loading did not check the policy architecture.** A different `net_arch` or features extractor now raises a `ValueError` on load (SB3 and imitation agents).
- **PettingZoo training could not be seeded.** Environment seeding is supported for reproducible runs.
- **Video recording skipped the features extractor's preprocessing.** Fixed.
- **ml-agents-envs:** the matching `final_observation` and episode-end fixes live in the ml-agents-envs fork (artifactory 1.35), not here.

## Open items (not fixed)

- **`record_video` rejects EnvironmentFactory tuples in SB3 mode**, so the experiment space's factory configs upload without a video.
- **Same-class features extractors with different preprocessing are not detected on load.** The load check compares classes and kwargs, not the preprocessing.
- **PettingZoo agents which leave without restarting** still fail inside supersuit (`MarkovVectorEnv` asserts on missing agents).
- **Trainable features extractors for d3rlpy and QLearning** are not supported, and d3rlpy does not save the extractor. The plan is in `.scratch/trainable-features-extractor/README.md`.
- **Experiment space** needs the rl_framework release that requires async-gym-agents 0.5.0 (`EpisodeCallbackContext.worker_index` / `env_index`); until then its venv (0.4.1) fails `test_launcher.py::test_cartpole_config_trains_and_evaluates_through_main`.

## Findings (as reported)

Ranked by impact. Line numbers refer to 827c39b.

### High

**B-01: PettingZoo training trains on (and logs) wrong episode returns.**
`agent/reinforcement/stable_baselines.py:146-196` (`AutoResetSB3VecEnvWrapper.step_wait`)

- The wrapper reports every done one step late, so the step before the done is repeated.
- `MarkovVectorEnv` has already reset by then, so the first transition of the next episode is replaced by the old terminal transition.
- Result: with rewards 1..5 (true return 15), SB3 sees episodes of length 6 with return 19, then length 5 with return 18.
- PPO's advantages and the logged "Episode reward" are both wrong.

**B-02: GAIL, AIRL and Density training never call the framework callbacks.**
`util/sb3_training_callbacks.py:21-29`; used at `gail.py:82`, `airl.py:82`, `density.py:64`.

- `add_callbacks_to_callback` wraps a `None` or single-callback target in a new local `CallbackList` and returns nothing.
- In imitation 1.0, `gen_callback` and `wrapper_callback` are single callbacks, so `SavingCallback` and `LoggingCallback` are silently dropped.

**B-03: SQIL cannot train with the locked dependencies.**

- stable-baselines3 2.7 added `ReplayBufferSamples.discounts`, which is `None` for plain buffers.
- imitation 1.0's `SQILReplayBuffer.sample` `th.cat`s every field, so the first gradient step raises `TypeError`.

**B-04: Saved GAIL and AIRL agents cannot be loaded.**
`gail.py:96`, `airl.py:96`

- The reward net is saved as a pickled module and loaded with `torch.load(path)`, which defaults to `weights_only=True` since torch 2.6.

### Medium

**B-05: Retraining an SB3 agent with a features extractor and user `policy_kwargs` crashes** (`stable_baselines.py:249-261`). The user's `policy_kwargs` replace the saved ones on reload, so the rebuilt policy has no extractor.

**B-06: Discarded episodes crash `LoggingCallback` or log NaN** (`sb3_training_callbacks.py:139-152`, `metric_logging_utils.py:79`).

**B-07: A discarded episode's reward is added to the next episode** (`metric_logging_utils.py:64-70`).

**B-08: The HuggingFace final upload fails after normal training** (`hugging_face_connector.py:144-147`): `json.dump` fails on numpy values.

**B-09: `AsyncStableBaselinesAgent` leaves workers running when training fails** (`async_stable_baselines.py:152-156`).

**B-10: `D3RLPYAgent` crashes on short training runs** (`d3rlpy.py:315-326`): with batch size 32, any `total_timesteps < 10016` fails.

**B-11: `D3RLPYAgent` evaluation metrics are lost** (`d3rlpy.py:306-326`): evaluator scores land after the epoch's last step commit, so they are logged one epoch late and never for the last epoch.

**B-12: `D3RLPYAgent` with validation episodes crashes for value-free algorithms** (`d3rlpy.py:309-313`): the `TDErrorEvaluator` fails for (Discrete)BC.

**B-13: `D3RLPYAgent` cannot train on PettingZoo environments** (`d3rlpy.py:272-307`).

**B-14: BC validation runs out of batches** (`imitation/algorithms/bc.py:89-91,127`).

**B-15: The ClearML histogram logging leaks matplotlib figures** (`clearml_connector.py:106-113`).

### Low

**B-16: The QLearning epsilon schedule is broken** (`q_learning.py:201-205`): it overshoots below `epsilon_min` and ignores the configured `epsilon`.

**B-17: `QLearning.save_to_file("q.pkl")` fails** (`q_learning.py:221`): `os.makedirs("")`.

**B-18: Video recording breaks on paths with spaces** (`video_recording.py:66`).

**B-19: SB3 ONNX export fails for Dict observation spaces** (`stable_baselines.py:345-347`).

**B-20: The SB3 ONNX export is stochastic** (`stable_baselines.py:341`).

## Limitations and observations (no xfail)

These are documented or deliberate behaviors rather than bugs, but worth deciding on:

- **Features extractor on VecEnv or EnvironmentFactory:** `wrap_environment_with_features_extractor_preprocessor` raises `TypeError`, so `train` and `evaluate` with a features extractor only work for gym and PettingZoo envs.
- **Imitation checkpoint interval is fixed:** `ImitationAgent` uses `SavingCallback(self, connector)` with the default 50,000-step interval and has no `callback_kwargs`.
- **Video errors are swallowed:** `record_video` logs and swallows all exceptions, so a failing policy produces an upload without a video.
- **`D3RLPYAgent.save_to_file` needs a `Path`:** it calls `file_path.as_posix()` and fails for `str`.
- **`D3RLPYAgent.choose_action` guard is dead:** `if not self.algorithm` is never true because the algorithm is created in `__init__`.
- **Dead code:** `LoggingCallback._supports_episode_aggregation` is unused.
- **Optimizer reset repeats:** `reset_optimizer=True` resets the optimizer on every `train()` call, not only on the first one after loading.

## Downstream relevance (synthetic-player-experiment-space)

The experiment space trains through config with `StableBaselinesAgent` / `AsyncStableBaselinesAgent` (PPO, MultiDiscrete actions, EnvironmentFactory tuples), `CustomAgent`+`QLearning` (Taxi) and `ImitationAgent`. It uses ClearML and HuggingFace connectors and `step_metric_*` infos.

- **Directly reachable with shipped configs:** B-01 (Unity PettingZoo), B-08 (HuggingFace connector), B-06 / B-07 (async discards), B-09 (hang after a worker failure), B-15 (distribution logging), the evaluation and callback-attribution items.
- **Reachable by config choice:** B-02 / B-03 / B-04 / B-14 (imitation algorithms), B-16 (`Custom` + `Q_LEARNING`), B-05 (`features_extractor` with `policy_kwargs`).
- **Not used downstream:** `D3RLPYAgent` (B-10 to B-13) and the rl_framework ONNX export (B-19, B-20).

The experiment space requires `reinforcement-learning-framework>=0.11.4` for these fixes; its `AGENTS.md` documents the split of responsibilities and per-worker logging.
