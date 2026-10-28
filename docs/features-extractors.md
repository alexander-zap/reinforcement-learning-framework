# Features extractors

A [`FeaturesExtractor`](../src/rl_framework/util/features_extractor_utils.py) is passed to an agent with the
`features_extractor` argument. It has two parts:

- `preprocess(observations) -> np.ndarray`: numpy preprocessing before the model. The agents apply it to the
  observations of the environments (training, `evaluate`, `record_video`) and to recorded demonstrations. The default
  only converts to a numpy array. It gets a batch of observations: one for a gym or PettingZoo environment, all
  sub-environments' for a `VecEnv` (dict observations as an array of one dict per sub-environment). The environments
  of an `EnvironmentFactory` are preprocessed when the factory creates them (also in subprocesses).
- `forward(tensor) -> tensor`: a torch module inside the model, with the attribute `output_dim` (size of its output)
  and optionally `preprocessed_observation_space` (the observation space after `preprocess`).

## How each agent uses it

| Agent | Network | Trained | Saved with the model |
| --- | --- | --- | --- |
| `StableBaselinesAgent`, `AsyncStableBaselinesAgent` | features extractor → policy/value networks (`net_arch`) → output layers | yes | yes (in the SB3 policy) |
| `ImitationAgent` (BC, GAIL, AIRL, Density, SQIL) | as SB3 (the policy is an SB3 policy); GAIL/AIRL discriminator: see below | yes | yes (in the policy) |
| `D3RLPYAgent` | features extractor → network of the encoder factory → output layer | yes | yes (in the d3rlpy model) |
| `CustomAgent` + `QLearning` | features extractor maps an observation to a Q-table row | **no**, must be frozen | yes (its weights, next to the Q-table) |

Every network (e.g. actor, critic, target networks) gets its own copy of the features extractor, so target networks do
not share the weights of the online networks. The features extractor object passed to the agent therefore keeps its
initial weights; the trained ones are in the agent's model.

### GAIL and AIRL (ImitationAgent)

Next to the policy (the generator), GAIL and AIRL train a discriminator (reward network). With a features extractor,
the discriminator gets **its own copy** of the features extractor in front of imitation's `BasicRewardNet`, which then
works on the extracted features (`output_dim`). The copy is trained with the discriminator loss, independent of the
policy's copy. The discriminator is saved with the agent (so training continues with it after loading), including its
copy of the features extractor; only the policy's copy is used to act.

### D3RLPYAgent

d3rlpy builds every network from an encoder factory (`encoder_factory`, or `actor_encoder_factory`,
`critic_encoder_factory`, ... depending on the algorithm) followed by a linear output layer. With a features extractor,
the features extractor is put **in front of** the configured encoder, which is built for the extracted features
(`output_dim`). The encoder factory parameters therefore define the network after the features extractor, as without
one (d3rlpy's default: a 256x256 MLP):

```
agent = D3RLPYAgent(
    SAC,
    {
        "actor_encoder_factory": VectorEncoderFactory(hidden_units=[128, 128]),
        "critic_encoder_factory": VectorEncoderFactory(hidden_units=[]),  # features extractor -> output layer
    },
    features_extractor=MyFeaturesExtractor(),
)
```

Critics of continuous actions (and BCQ/BEAR/PLAS imitators) use d3rlpy's state-action encoder, which concatenates the
action (one-hot encoded if discrete) to the extracted features.

A preprocessing-only features extractor uses the identity as `forward`; the network is then d3rlpy's encoder on the
preprocessed observations.

### QLearning

`QLearning` looks up its Q-table with the output of the features extractor (one integer, `output_dim = 1`). It does not
train the features extractor, and training it jointly would not work: the table lookup passes no gradient to it, and
changing it would remap the rows already learned (a trainable Q-function on top of the features is DQN, available with
`StableBaselinesAgent`). Pretrain the features extractor (e.g. in another agent) and freeze it
(`features_extractor.requires_grad_(False)`); a features extractor with trainable parameters logs a warning.

Its weights are saved with the Q-table and restored on load, also if they are untrained random weights (e.g. a random
projection which discretizes observations): they define which observation maps to which row.

## Loading

On load, the agent's features extractor must match the one saved with the model; otherwise loading raises a
`ValueError` which lists the differences. Compared are the class, `output_dim`, `preprocessed_observation_space` and the
shapes of the parameters (not their values, which come from the file). Loading also raises if only one of the saved
model and the agent has a features extractor. For GAIL and AIRL, the features extractor of the saved discriminator is
checked as well.

## Limitations

- **Dict observations with GAIL and AIRL are not supported.** imitation's adversarial training requires array
  observations (also after `preprocess`).

- **Preprocessing is not checked on load.** `preprocess` is arbitrary user code and cannot be compared. Two features
  extractors of the same class with the same structure but a different preprocessing (e.g. other normalization
  constants stored as plain attributes) are not detected. Keeping the preprocessing consistent between saving and
  loading is the user's responsibility.
