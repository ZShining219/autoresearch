# LibSignal Training & Testing Guide

A practical guide to training RL agents and running inference with trained models.

**Environment:** Use `./setup.sh` (see [README.md](../README.md)) to create the `traffic` conda env. Team experiments use **`--world sumo`**; CityFlow may work upstream-style but is not validated in this fork.

## Quick Start

### Run a Baseline (No Training Needed)
```bash
python run.py --agent maxpressure --world sumo --network sumo1x1 --seed 42 --ngpu 0
```
This runs MaxPressure (heuristic algorithm) - instant results, no training.

### Train a DQN Agent on SUMO
```bash
python run.py --agent dqn --world sumo --network sumo1x1 --seed 42 --ngpu 0
```
Models save to: `data/output_data/tsc/sumo_dqn_test/model/`

**Speed tips:** Training is env-bound (SUMO + observation plumbing), not GPU-bound for
default DQN — prefer `--ngpu -1` on CPU-only hosts. Set `trainer.test_when_train: False`
in the agent YAML to skip the per-episode eval rollout (~2× less simulation). See
[EFFICIENCY_AUDIT.md](./EFFICIENCY_AUDIT.md) for the full breakdown and profiling scripts.

### Test a Pre-trained Model
```bash
python run.py --agent dqn --world sumo --network sumo1x1 --seed 42 --ngpu 0
```
Then modify the config to test without retraining (see below).

### Ghost-Physics Ceiling Baselines (Unrealistic Benchmark)

These baselines use `physics_mode: ghost` so vehicles **ignore each other** (no car-following, no right-of-way conflicts) but **still obey red lights**. Signal control still matters; only vehicle interaction is removed. This is an unrealistic ceiling for comparing how close real methods get.

Requires **libsumo** (`--interface libsumo`, the default).

```bash
# Fixed-time + ghost physics
python run.py --agent fixedtime_ncollision --world sumo --network sumo1x1 --interface libsumo --prefix ceiling

# MaxPressure + ghost physics
python run.py --agent maxpressure_ncollision --world sumo --network sumo1x1 --interface libsumo --prefix ceiling
```

Configs: `configs/tsc/fixedtime_ncollision.yml` and `configs/tsc/maxpressure_ncollision.yml` (set `world.physics_mode: ghost`).

**Do not compare** standard RL agents directly to ghost baselines without noting the physics difference — ghost runs use different SUMO vehicle behavior. Use ghost results as a ceiling, not as peers.

---

## Understanding Training vs Testing

### The Training Loop

When you run with a RL agent (dqn, ppo, etc.), here's what happens:

**1. Environment Setup**
- Simulator initializes (SUMO) based on `configs/sim/{network}.cfg`
- Agents created from registered model type
- TSCEnv wraps everything into OpenAI Gym interface

**2. Training Loop (in TSCTrainer.train())**
```
for episode in range(episodes):
    obs = env.reset()
    
    for step in range(episode_steps):
        if step % action_interval == 0:
            # Get agent actions
            actions = agent.get_action(obs, current_phase, test=False)  # explores
            
            # Take steps in simulator
            obs, rewards, dones, info = env.step(actions)
            
            # Store transition in replay buffer
            agent.remember(obs, action, reward, next_obs, ...)
        
        # Train model if enough samples in buffer
        if total_steps > learning_start:
            loss = agent.train()  # sample from replay buffer, update network
    
    # Save checkpoint every N episodes
    if episode % save_rate == 0:
        agent.save_model(e=episode)
```

**3. Model Checkpoints**
- Saved periodically during training: `{output_dir}/model/{episode}_{agent_rank}.pt`
- Example: After episode 100 of DQN training on sumo1x1:
  ```
  data/output_data/tsc/sumo_dqn_test/model/100_0.pt
  data/output_data/tsc/sumo_dqn_test/model/200_0.pt
  ```

**4. Metrics Logged During Training**
- Reward per episode
- Queue length
- Travel time
- Throughput
- Q-loss (for RL agents)

---

## Config System

All settings defined in YAML configs that inherit from base:

### Structure
```
configs/tsc/
├── base.yml          # Default settings for all agents
├── dqn.yml           # DQN-specific overrides
├── maxpressure.yml   # MaxPressure-specific overrides
└── ...
```

### Key Training Settings in `configs/tsc/base.yml`

```yaml
trainer:
  episodes: 200              # How many episodes to run
  steps: 3600               # Steps per episode (time in sim)
  test_steps: 3600          # Steps during test/eval
  action_interval: 10       # How many sim steps between agent decisions
  learning_start: 1000      # Don't train until this many decisions made
  buffer_size: 5000         # Replay buffer size
  update_model_rate: 1      # Train every N decisions
  update_target_rate: 10    # Update target network every N decisions

model:
  train_model: True         # Enable training
  test_model: True          # Run test after each episode
  load_model: False         # Load pretrained weights (False = train from scratch)
  save_model: True          # Save checkpoints
  learning_rate: 0.001
  batch_size: 64
  gamma: 0.95               # Discount factor
  epsilon: 0.5              # Exploration rate

logger:
  save_rate: 5              # Save model every N episodes
  root_dir: "data/output_data/"
```

### Agent-Specific Settings

**DQN** (`configs/tsc/dqn.yml`):
```yaml
model:
  train_model: True
  epsilon: 0.1              # Starting exploration probability
  epsilon_decay: 0.995      # Decay per decision
  epsilon_min: 0.01         # Min exploration
  one_hot: True             # One-hot encode phase as part of obs
  phase: True               # Include current phase in observation
```

**MaxPressure** (`configs/tsc/maxpressure.yml`):
```yaml
model:
  train_model: False        # No training - pure heuristic
  test_model: True
```

---

## Training Workflow

### Step 1: Train a Model

```bash
# Train DQN on 4x4 grid for 200 episodes
python run.py \
  --agent dqn \
  --world sumo \
  --network sumo4x4 \
  --seed 42 \
  --ngpu 0 \
  --prefix my_dqn_run

# Output goes to: data/output_data/tsc/sumo_dqn_my_dqn_run/
# Models saved to: data/output_data/tsc/sumo_dqn_my_dqn_run/model/
```

### Step 2: Monitor Training

Training logs go to: `data/output_data/tsc/{experiment_name}/logger/`

Check logs for:
- Episode rewards
- Queue length trends
- Travel time improvements
- Q-loss (should decrease)

### Step 3: Choose Best Checkpoint

After training, models are at: `model/{episode}_{agent_rank}.pt`

Common practice: use final model or best performing episode

Example: If episode 150 had best metrics, you'd use `150_0.pt`

---

## Testing/Inference Workflow

### Option 1: Test During Training

Default config already does this:
- `test_when_train: True` in base.yml
- After each episode, runs test on same network
- Evaluates current policy without exploration

### Option 2: Test Pre-trained Model

Create a test config:

```yaml
# configs/tsc/dqn_test.yml
includes:
  - configs/tsc/dqn.yml

model:
  train_model: False        # Skip training
  test_model: True          # Run test
  load_model: True          # Load pretrained weights
```

Then run:
```bash
python run.py \
  --agent dqn \
  --world sumo \
  --network sumo1x1 \
  --seed 42

# Code will:
# 1. Load model from latest checkpoint
# 2. Run test_steps (3600 steps default)
# 3. Report final metrics
```

**Note**: Load behavior (in `agent/dqn.py`):
```python
def load_model(self, e):
    # Loads from: {output_dir}/model/{e}_{rank}.pt
    model_name = os.path.join(
        Registry.mapping['logger_mapping']['path'].path,
        'model', 
        f'{e}_{self.rank}.pt'
    )
    self.model.load_state_dict(torch.load(model_name))
```

### Option 3: Custom Evaluation Script

To test on multiple seeds or networks:

```python
import run
# Modify run.py or create custom script
# that calls TSCTrainer.test() with different configs
```

---

## Output Directory Structure

After running an experiment:

```
data/output_data/tsc/sumo_dqn_test/
├── logger/
│   ├── {date}_{time}_BRF.log    # Brief logs (per episode)
│   └── {date}_{time}_DTL.log    # Detailed logs (per step)
├── model/
│   ├── 0_0.pt                    # Agent 0, episode 0
│   ├── 5_0.pt
│   ├── 10_0.pt
│   └── ...
├── replay/                        # CityFlow only
└── dataset/
```

---

## Running Different Baselines

### Non-RL Baselines (Instant)

**MaxPressure** - Queue-based pressure:
```bash
python run.py --agent maxpressure --world sumo --network sumo1x1
```

**SOTL (Self-Optimizing Traffic Light)** - Fixed cycle with adaptive timing:
```bash
python run.py --agent sotl --world sumo --network sumo1x1
```

**FixedTime** - Simple fixed timing:
```bash
python run.py --agent fixedtime --world sumo --network sumo1x1
```

### RL Baselines (Requires Training)

**DQN** - Deep Q-Network:
```bash
python run.py --agent dqn --world sumo --network sumo1x1
```

**PPO** - Proximal Policy Optimization:
```bash
python run.py --agent ppo --world sumo --network sumo1x1
```

**MADDPG** - Multi-Agent DDPG:
```bash
python run.py --agent maddpg --world sumo --network sumo1x1
```

---

## Common Scenarios

### Scenario 1: Quick Baseline Comparison
```bash
# Run 3 baselines - no training, instant results
python run.py --agent maxpressure --world sumo --network sumo1x1
python run.py --agent sotl --world sumo --network sumo1x1
python run.py --agent fixedtime --world sumo --network sumo1x1
```

### Scenario 2: Train and Save
```bash
# Train DQN (200 episodes, saves every 5)
python run.py --agent dqn --world sumo --network sumo1x1

# Models: model/{0,5,10,...,200}_0.pt
```

### Scenario 3: Compare Multiple Seeds
```bash
for seed in 1 2 3; do
  python run.py --agent dqn --world sumo --network sumo1x1 --seed $seed
done
# Outputs in different dirs: sumo_dqn_test, sumo_dqn_test_1, etc.
```

### Scenario 4: Test Different Networks
```bash
# Same agent, different road topologies
python run.py --agent dqn --world sumo --network sumo1x1
python run.py --agent dqn --world sumo --network sumo4x4
```

---

## GPU vs CPU

### Enable GPU (if available)
```bash
python run.py --agent dqn --world sumo --network sumo1x1 --ngpu 0
# Uses GPU 0
```

### Force CPU
```bash
python run.py --agent dqn --world sumo --network sumo1x1 --ngpu -1
# Uses CPU only (slower for RL training)
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `run.py` | Entry point, parses CLI args, creates Runner |
| `trainer/tsc_trainer.py` | Training loop, calls `agent.train()` each decision |
| `agent/dqn.py` | DQN implementation, `save_model()` and `load_model()` |
| `configs/tsc/base.yml` | Default settings for all runs |
| `configs/tsc/{agent}.yml` | Agent-specific overrides |
| `environment.py` | TSCEnv gym wrapper |

---

## Troubleshooting

**Q: Models not saving?**
A: Check `logger.save_model: True` in config and `model_dir` is writable

**Q: Load_model not working?**
A: Ensure episode number exists in model/ directory and matches agent rank

**Q: GPU out of memory?**
A: Use `--ngpu -1` for CPU, or reduce `batch_size` in config

**Q: Training too slow on CPU?**
A: Run baselines (maxpressure, sotl) instead - no training needed
