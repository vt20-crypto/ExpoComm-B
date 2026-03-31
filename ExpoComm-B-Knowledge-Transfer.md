# ExpoComm-B: Knowledge Transfer Document

> **What is this document?**
> This is a living knowledge-transfer document for the ExpoComm-B project. It is written so that someone new to this area can understand: **what** we're building, **why** we're building it, and **how** the pieces fit together. Prerequisites are listed before each section so you can self-learn anything unfamiliar.

> **Project**: Adapting Sparse Communication Graph Learning for Multi-Agent Coordination in Bandwidth-Constrained Settings
> **Course**: COMP 559 – Machine Learning with Graphs (Prof. Arlei Silva, Rice University, Spring 2026)
> **Team**: Aneesh Thatiparti (vt20), Ansh Dabral (ad258), Madhu Thatiparti (mt180)

---

## Table of Contents
1. [Background & Prerequisites](#1-background--prerequisites)
2. [What Problem Are We Solving?](#2-what-problem-are-we-solving)
3. [The Two Papers We Build On](#3-the-two-papers-we-build-on)
4. [Our Contribution: ExpoComm-B](#4-our-contribution-expocomm-b)
5. [Codebase Architecture](#5-codebase-architecture)
6. [BVME Integration: The Technical Plan](#6-bvme-integration-the-technical-plan)
7. [Environment Setup & Tools](#7-environment-setup--tools)
8. [Team Responsibilities](#8-team-responsibilities)
9. [Decisions Log](#9-decisions-log)
10. [Glossary](#10-glossary)

---

## 1. Background & Prerequisites

Before reading this document, you should have a basic understanding of:

### Must-Know Concepts

| Concept | What it is | Where to learn |
|---------|-----------|----------------|
| **Reinforcement Learning (RL)** | Agents learn by interacting with an environment and receiving rewards | [Sutton & Barto Ch. 1-3](http://incompleteideas.net/book/the-book.html) |
| **Multi-Agent RL (MARL)** | Multiple agents learn simultaneously in the same environment | [MARL overview paper](https://arxiv.org/abs/2404.04898) |
| **Graph Neural Networks (GNNs)** | Neural networks that operate on graph-structured data (nodes + edges) | [Kipf & Welling 2017](https://arxiv.org/abs/1609.02907) |
| **Partial Observability** | Each agent can only see a local part of the world, not the full state | Covered in Dec-POMDP formulation below |
| **PyTorch** | The deep learning framework used in our codebase | [pytorch.org/tutorials](https://pytorch.org/tutorials/) |
| **Conda** | Package/environment manager for Python | [docs.conda.io](https://docs.conda.io/) |

### Helpful but Optional

| Concept | What it is |
|---------|-----------|
| **Value Decomposition (QMIX/VDN)** | Methods to decompose a team reward into per-agent Q-values |
| **Variational Autoencoders (VAEs)** | Neural networks that learn compressed representations using Gaussian distributions |
| **KL Divergence** | A measure of how different two probability distributions are |
| **CTDE** | Centralized Training, Decentralized Execution — train with full info, deploy with local info only |

---

## 2. What Problem Are We Solving?

### The Real-World Scenario
Imagine a team of search-and-rescue drones. Each drone can only see what's directly around it (partial observability). To coordinate — say, to avoid searching the same area twice — they need to **communicate**. But their wireless radios have limited bandwidth. They can't send everything they know to every other drone.

### The Research Gap
Current methods in multi-agent RL focus on learning **who** should communicate (sparse communication graphs). But they assume unlimited bandwidth — every message is a full 64-dimensional vector. Nobody has studied what happens when you also need to **compress** those messages.

### What We're Doing
We combine:
- **ExpoComm** (ICLR 2025): learns a sparse communication topology (who talks to whom)
- **BVME** (AAMAS 2026): compresses messages to fit within a bandwidth budget (what to say)

Neither paper does both. We combine them into **ExpoComm-B** and evaluate on benchmarks (SMACv2, MPE) where this combination hasn't been tested.

### Why This Matters for Our Course
This is a **"GraphML problem in a new setting"** (project type 3 from Prof. Silva's guidelines). The graph is the communication topology between agents. The new setting is bandwidth constraints.

---

## 3. The Two Papers We Build On

### Paper 1: ExpoComm (ICLR 2025)
- **Title**: Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning
- **Paper**: https://arxiv.org/abs/2502.19717
- **Code**: https://github.com/LXXXXR/ExpoComm (public, Apache-2.0)

**What it does**: Instead of letting every agent talk to every other agent (O(N²) connections), ExpoComm uses an **exponential graph** topology. Agent `i` talks to agents at positions `i+1, i+2, i+4, i+8, ...` (modulo N). This gives O(log N) connections per agent while maintaining a small graph diameter (any agent can reach any other in O(log N) hops).

**Key innovation**: Uses memory-based message processors (GRU) and auxiliary tasks (predict the global state from messages) to ensure messages carry useful information.

**Limitation**: Assumes unlimited bandwidth — messages are full 64-dimensional hidden state vectors.

### Paper 2: BVME (AAMAS 2026)
- **Title**: Bandwidth-constrained Variational Message Encoding for Cooperative MARL
- **Paper**: https://arxiv.org/abs/2512.11179
- **Code**: ❌ No public code (we implement from the paper)

**What it does**: Treats each message as a **sample from a learned Gaussian distribution** instead of a fixed vector. Uses KL divergence regularization to control how much information each message carries. The bandwidth budget B controls the compression ratio `r = d_msg / d_obs`.

**Key innovation**: Provides principled, tunable control over compression. Works especially well on **sparse graphs** where each message matters more.

**Key finding**: Performance shows U-shaped sensitivity to bandwidth — BVME excels at extreme compression (r ≤ 0.05, i.e., 95% reduction) while adding < 5% training overhead.

**Limitation**: Uses GACG (Group-Aware Coordination Graphs) as its topology method. Never tested with ExpoComm's exponential topology.

---

## 4. Our Contribution: ExpoComm-B

### Architecture Flow
```
Agent observation (o_i)
    ↓
FC layer + GRU encoder
    ↓
ExpoComm sparse topology selects neighbors
    ↓
_communicate(): receive messages from selected neighbors
    ↓
★ BVME compression: encode message → (μ, σ) → sample z ~ N(μ, σ²) ← OUR ADDITION
    ↓
[hidden_state ; compressed_message] → Q-network → action
```

### What's Novel
1. **First combination** of ExpoComm's exponential topology + BVME's variational compression
2. **First evaluation** of ExpoComm on SMACv2 and MPE benchmarks
3. **Empirical study** of how sparse topology + compression interact under varying bandwidth

### Experiment Plan
We compare 4 methods:

| Method | Topology | Compression | Source |
|--------|----------|-------------|--------|
| Full-comm (QMIX) | Full (all-to-all) | None | Baseline |
| ExpoComm | Sparse (exponential) | None | Paper 1 |
| BVME only | Full/GACG | Variational | Paper 2 |
| **ExpoComm-B (ours)** | **Sparse (exponential)** | **Variational** | **Our work** |

---

## 5. Codebase Architecture

### Prerequisites for this section
- Familiarity with Python classes and inheritance
- Basic understanding of PyTorch `nn.Module`
- Understanding of YAML configuration files

### Overview
The codebase is built on **EPyMARL** (Extended Python Multi-Agent Reinforcement Learning framework). It uses a **registry pattern** where components are registered by name in `__init__.py` files and instantiated from YAML configs.

### Directory Structure
```
ExpoComm/
├── src/
│   ├── main.py              # Entry point - loads configs, initializes W&B
│   ├── run.py               # Training loop - the core orchestration
│   ├── config/
│   │   ├── default.yaml     # Base hyperparameters (lr, gamma, hidden_dim, etc.)
│   │   ├── algs/            # Algorithm configs (agent type, learner, mixer)
│   │   ├── envs/            # Environment configs (SC2 maps, MAgent scenarios)
│   │   └── exp/             # Experiment configs (seed, env-specific overrides)
│   ├── controllers/         # Multi-Agent Controllers (MACs) - manage all agents
│   │   └── ExpoComm_controller.py  # THE exponential topology logic lives here
│   ├── modules/
│   │   ├── agents/          # Individual agent neural networks
│   │   │   └── ExpoComm_agent.py   # THE message passing + Q-network
│   │   └── mixers/          # Value decomposition (QMIX, VDN)
│   ├── learners/            # Training algorithms
│   │   └── q_learner.py     # QLearner, AuxQLearner (ExpoComm), ContQLearner
│   ├── runners/             # Environment interaction (episode collection)
│   ├── envs/                # Environment wrappers
│   └── components/          # Replay buffer, action selectors, etc.
├── env/                     # Custom MAgent environment files
├── docker/                  # Docker configuration
├── ExpoComm_env.yaml        # Full conda environment specification
└── install_sc2.sh           # StarCraft II installation script
```

### Data Flow (One Training Step)
```
1. Runner.run() → collects one episode by stepping the environment
2. Episode stored in ReplayBuffer
3. Buffer.sample() → gets a batch of episodes
4. For each timestep t in batch:
   a. Controller._build_inputs() → constructs agent observations
   b. Controller._build_inputs() → computes exponential neighbors (topk_indices)
   c. Agent.forward(inputs, hidden_info, topk_indices):
      - obs → fc1 → GRU → h (hidden state)
      - h + prev_messages + topk_indices → _communicate() → new_msg
      - [h ; msg] → fc2 → Q-values
      - msg → predict_net → predicted_state (for aux loss)
5. Learner.train():
   - Q-values + actions → chosen_action_qvals
   - QMIX mixer combines per-agent Q-values into Q_total
   - TD loss = (Q_total - target)²
   - Aux loss = (predicted_state - actual_state)² (grounds messages)
   - Total loss = TD_loss + aux_coef × Aux_loss
   - Backprop and update
```

### The Exponential Graph (How Neighbors Are Selected)

For N agents with `topk=6`:
```
Agent 0 talks to: [0, 1, 2, 4, 8, 16]  (itself + powers of 2)
Agent 1 talks to: [1, 2, 3, 5, 9, 17]
Agent 2 talks to: [2, 3, 4, 6, 10, 18]
...all modulo N
```

With `one_peer=True` (used in practice): at timestep t, agent only sends to ONE neighbor:
```
t=0: talk to self (index 0)
t=1: talk to agent i+1 (index 1)
t=2: talk to agent i+2 (index 2)
t=3: talk to agent i+4 (index 3)
t=4: talk to agent i+8 (index 4)
t=5: talk to agent i+16 (index 5)
t=6: back to self (index 0, wraps around)
```

This round-robin across exponential neighbors is what makes ExpoComm scalable AND bandwidth-efficient.

### Registry System
Components are registered by string name:

```python
# controllers/__init__.py
REGISTRY["ExpoComm_mac"] = ExpoCommMAC

# modules/agents/__init__.py
REGISTRY["ExpoComm_one_peer"] = ExpoCommOAgent

# learners/__init__.py
REGISTRY["aux_q_learner"] = AuxQLearner
```

YAML config selects which component to use:
```yaml
mac: "ExpoComm_mac"          # → ExpoCommMAC
agent: "ExpoComm_one_peer"   # → ExpoCommOAgent
learner: "aux_q_learner"     # → AuxQLearner
mixer: "qmix"                # → QMixer
```

---

## 6. BVME Integration: The Technical Plan

### Prerequisites for this section
- Understanding of Gaussian distributions (mean μ, variance σ²)
- KL divergence: KL(P || Q) measures how P differs from Q
- Reparameterization trick: z = μ + σ × ε, where ε ~ N(0, I) — allows gradients to flow through sampling

### What BVME Does (Step by Step)

1. **Takes a message vector** `msg ∈ ℝ^{d_msg}` (the output of ExpoComm's `_communicate()`)
2. **Encodes it as a Gaussian distribution**:
   - `μ = Enc_μ(msg)` (single-layer MLP, outputs mean)
   - `log_σ = Enc_σ(msg)` (single-layer MLP, outputs log-variance, clamped to [-5, 3])
3. **Samples a compressed message** (during training):
   - `z = μ + σ × ε`, where `ε ~ N(0, I)` (reparameterization trick)
4. **During evaluation**: uses deterministic `z = μ` (no sampling, for stability)
5. **Computes KL loss** to control information capacity:
   ```
   KL = Σ_d [ (σ²_d + μ²_d) / σ₀² - log(σ²_d / σ₀²) - 1 ] / (2 × d_msg)
   ```
   Where `σ₀` is the prior standard deviation (hyperparameter controlling capacity)

### BVME Hyperparameters

| Param | Symbol | Meaning | Typical Values |
|-------|--------|---------|:---:|
| Compression ratio | r | d_msg / d_obs. Lower = more compression | 0.02 – 0.30 |
| Prior scale | σ₀ | Controls per-dimension information capacity | 0.005 – 0.02 |
| KL weight | λ_KL | How strongly to enforce bandwidth constraint | 0.5 – 2.0 |

### Exact Code Insertion Points

**File**: `src/modules/agents/ExpoComm_agent.py`
**Location**: After `_communicate()`, before `th.cat([h, msg])`

**File**: `src/learners/q_learner.py`
**Location**: After `aux_loss` computed, add `kl_loss` to total

See the architecture walkthrough document for exact line numbers.

---

## 7. Environment Setup & Tools

### Compute Resources
- **Local**: Apple M4 Pro (14-core CPU, 20-core GPU, 24GB unified memory)
  - Good for: code development, small test runs, MPE experiments
  - Not for: GPU training (MPS has limited PyTorch MARL support)
- **NOTS Cluster** (Rice University): NVIDIA A40, V100, K80 GPUs
  - Good for: all training runs, SMACv2 (requires Linux), large-scale experiments
- **AWS Credits**: $100 (backup option)

### Software Stack
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8.18 | Required by ExpoComm dependencies |
| PyTorch | 1.13.1+cu117 | Deep learning framework |
| CUDA | 11.7 | GPU acceleration |
| Weights & Biases | Latest | Experiment tracking & visualization |
| Conda | Latest | Environment management |

### Weights & Biases (W&B) Setup

**What is W&B?** A cloud-based tool for tracking ML experiments. It logs metrics (loss, reward, etc.) during training and lets you visualize them in your browser from anywhere. This is critical because training runs on the NOTS cluster take hours/days, and you need to monitor them remotely.

**Setup steps:**
1. Create account at https://wandb.ai/
2. Get API key from https://wandb.ai/authorize
3. On NOTS, run: `wandb login` and paste your API key
4. In our code, change entity in `main.py` from `"lxxxxr"` to your W&B username/team name

**What to configure in `main.py`:**
```python
wandb.init(
    project=exp_name,                    # Auto-set from exp config
    name=run_name + "-" + algo_name,     # Auto-set
    entity="YOUR_WANDB_USERNAME",        # ← Change this
    config=config,                       # Logs all hyperparameters
)
```

**Environment variable alternative** (no code change needed):
```bash
export WANDB_ENTITY="your_username"
export WANDB_API_KEY="your_key"
export WANDB_PROJECT="ExpoComm-B"
```

Then modify `main.py` to use:
```python
wandb.init(
    project=os.environ.get("WANDB_PROJECT", exp_name),
    entity=os.environ.get("WANDB_ENTITY", "default"),
    config=config,
)
```

### Environments We Use

| Environment | What it is | Agents | Why we use it |
|-------------|-----------|:---:|---------------|
| **SMACv2** | StarCraft II micromanagement battles | 5-25 | Standard MARL benchmark, randomized, challenging |
| **MPE Tag** | 2D predator-prey with obstacles | 10+3 | Lightweight, fast iteration, communication-critical |
| **MAgent** | Large-scale grid-world battles | 25-100 | Already in ExpoComm codebase, scalability tests |

---

## 8. Team Responsibilities

| Member | Role | Primary Tasks |
|--------|------|---------------|
| **Aneesh (vt20)** | Integration Lead | ExpoComm setup, BVME implementation, SMACv2 experiments |
| **Ansh (ad258)** | Baselines & MPE | MPE environment, baseline comparisons, ablation studies |
| **Madhu (mt180)** | Analysis & Report | Scalability experiments, figures/tables, report writing |

---

## 9. Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-29 | Use ExpoComm codebase as base (not BVME) | ExpoComm has public code, BVME doesn't. ExpoComm's EPyMARL base is well-structured. |
| 2026-03-29 | Keep W&B logging (not switch to TensorBoard) | Cloud-based monitoring essential for NOTS cluster. Team can view remotely. |
| 2026-03-29 | Implement BVME module from paper, not copy code | BVME has no public code. Our implementation is original work. |
| 2026-03-29 | Pivot from SMACv2 to MAgent/MPE on local cluster | Rice NOTS cluster has a strict 10GB hard limit on `~/home` storage, preventing the 3.5GB StarCraft II installation. SMACv2 deferred to AWS. |

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **Agent** | An autonomous entity in the environment (e.g., one drone, one game unit) |
| **CTDE** | Centralized Training, Decentralized Execution — during training, a central critic sees everything; during deployment, each agent acts independently |
| **Dec-POMDP** | Decentralized Partially Observable Markov Decision Process — the formal mathematical framework for cooperative MARL |
| **EPyMARL** | Extended Python Multi-Agent Reinforcement Learning — the framework ExpoComm is built on |
| **GRU** | Gated Recurrent Unit — a type of recurrent neural network that maintains memory over time |
| **Hidden state** | The internal memory vector of an agent's RNN, carried across timesteps |
| **KL Divergence** | Kullback-Leibler divergence — measures how one probability distribution differs from a reference distribution |
| **MAC** | Multi-Agent Controller — manages action selection for all agents |
| **Message passing** | The process of agents sending information (vectors) to each other through graph edges |
| **Mixer** | The module that combines per-agent Q-values into a team Q-value (e.g., QMIX) |
| **Observation** | What one agent can see at a given timestep (partial view of the world) |
| **QMIX** | A value decomposition method that ensures Q_total is monotonic in individual Q_i values |
| **Reparameterization trick** | Technique to sample from a distribution while still allowing gradient backpropagation: z = μ + σ × ε |
| **Sparse graph** | A graph where each node has few edges (O(log N) instead of O(N)) |
| **Topology** | The structure of connections between agents (who can talk to whom) |
| **Value decomposition** | Factoring team reward into per-agent components for decentralized execution |
| **Variational encoding** | Representing data as probability distributions rather than fixed vectors |

---

*This document is updated as the project progresses. Last updated: 2026-03-29.*
