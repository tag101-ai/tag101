# Tag101

Tag101 - The Decentralized Tagger for Social Posts

## Overview

Tag101 models a decentralized semantic tagging network where miners transform X (formerly Twitter) posts into concise, structured tags. Miners submit tags that identify the key entities, topics, events, or contextual meanings within each post. The best-performing miners are those that consistently produce tags aligned with the semantic consensus formed by the network.

These tags turn unstructured social posts into structured semantic signals that help systems index real-time information, track emerging topics, and provide cleaner context for downstream AI workflows. Instead of depending on a single tagging model, Tag101 evaluates tagging quality through a decentralized network, making the system more adaptive to diverse topics and phrasing styles.

## Task Definition

Given a Twitter / X post `p`, each miner produces a small set of tags:

Each miner returns:

$$
T_m = \lbrace t_1, t_2, \dots, t_K\rbrace
$$

Each tag `t` receives an individual score. Tags are expected to be relevant to the post, non-redundant, and well-formed.

## Data Source

Tasks are generated from a centralized database of Twitter / X posts collected from a publicly disclosed whitelist of accounts. In the initial phase, the whitelist is focused on AI-related accounts, forming a topic season that gives the subnet a consistent domain for evaluating miner behavior.

This centralized source helps maintain task quality, source transparency, and consistent validator behavior while the subnet’s scoring and miner ecosystem mature.

The whitelist may be updated periodically as new topic seasons are introduced, allowing Tag101 to expand into different domains over time. Validators retrieve posts from the database via API, with each post distributed to miners as an independent tagging task.

## Reward Mechanism

$$
\text{TagScore}(t)
= 0.6 \times C(t)+0.4 \times V(t)\times D(t)
$$

where:

- $C(t)$ is the consensus score
- $V(t)$ is the validity score
- $D(t)$ is the diversity score

The scoring design rewards tags that are aligned with other high-quality submissions, grounded in the original post, and not redundant with the miner’s own tags. The miner’s final task score is computed from its submitted tag scores.

### Consensus Score

Tags from all miners are embedded and clustered. The score reflects how strongly a tag aligns with the dominant interpretation:

$$
C(t)
= S(c) \times P(t)
$$

- $S(c)$: cluster support
- $P(t)$: centroid proximity, which rewards tags closer to the semantic center of their cluster

### Validity Score

Validity measures whether a tag is relevant to the original post and follows basic formatting requirements. The score combines content relevance with format checks:

$$
V(t) = \text{tier}(\text{BaseScore}(t) \times \text{FormatScore}(t))
$$

where:

- $\text{BaseScore}(t)$: content relevance based on semantic similarity or lexical overlap
- $\text{FormatScore}(t)$: basic tag format validation

The raw validity score is mapped into discrete tiers:

$$
V(t) \in \lbrace 0,\ 0.3,\ 0.6,\ 1.0 \rbrace
$$

This keeps validity stable while filtering tags that are unrelated, malformed, or outside the expected tag format.

### Diversity Score

Diversity measures whether a miner’s tags are semantically distinct from one another.

$$
D(t) = \text{novelty}(\max_{t' \in T_m, t' \neq t} \text{sim}(t, t'))
$$

where $D(t) \in [0, 1]$. 

The $\text{novelty}()$ function maps the highest similarity to a diversity score: sufficiently distinct tags receive full credit, near-duplicate tags receive no credit, and intermediate cases decay linearly.

Tags that are too similar to the miner’s other tags receive lower diversity scores, reducing the reward for repeated or redundant submissions.

### Aggregation

After each submitted tag is scored independently, the miner’s task score is computed by averaging its tag scores:

$$
\text{MinerScore}(m)
= \frac{1}{|T_m|}
\sum_{t \in T_m}
\text{TagScore}(t)
$$

This score represents the miner’s performance on a single tagging task.

### Duplicate Penalty

`TagScorer` then scales each miner’s score down when the same tag set appears multiple times on one task. Let $n$ be the number of miners sharing that set:

$$
\text{AdjustedMinerScore}(m)
= \text{MinerScore}(m) \times \frac{1}{1 + \exp\bigl(k \cdot (n - c)\bigr)}
$$

Reference defaults: $k = 0.1$, $c = 50$. Larger identical groups are penalized more sharply.

Over time, validator scoreboards aggregate miner performance across multiple tasks and convert recent task-level scores into relative miner scores for weight calculation.

## **Quick Start**

Assume your wallet hotkey is already created and registered on the target
subnet. Replace the placeholder values with the network, wallet, hotkey, axon
port, and runtime values for your deployment.

### Install

Prerequisite: Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run with Docker

Docker is the recommended way to run a miner or validator. It keeps the runtime
environment consistent and enables the built-in Docker auto-update flow by
default. After a successful update, the Docker monitor re-execs itself so the
watcher also runs the updated code.

Build the miner/validator image:

```bash
python -m tag101.deploy.docker_node build --image tag101:latest
```

#### Miner

Prepare a miner env file:

```bash
cp deploy/miner.docker.example miner.env
```

Edit `miner.env` for your host paths, network, wallet, hotkey, axon settings,
and OpenAI credentials. The default SN101 reference miner calls the OpenAI API,
so miners should set `OPENAI_API_KEY`; optional overrides are
`OPENAI_BASE_URL` and `OPENAI_MODEL`.

```bash
python -m tag101.deploy.docker_node start \
  --role miner \
  --name sn101-miner \
  --env-file miner.env
```

#### Validator

Prepare a validator env file:

```bash
cp deploy/validator.docker.example validator.env
```

Edit `validator.env` for your host paths, network, wallet, hotkey, and axon
settings, then start one validator container:

```bash
python -m tag101.deploy.docker_node start \
  --role validator \
  --name sn101-validator \
  --env-file validator.env
```

#### Useful commands

```bash
python -m tag101.deploy.docker_node restart --role miner --name sn101-miner --env-file miner.env
python -m tag101.deploy.docker_node status --name sn101-miner
docker logs -f sn101-miner
python -m tag101.deploy.docker_node stop --name sn101-miner
```

### Run with PM2

PM2 can also run a miner or validator directly on the host. Install PM2
separately, keep using the Python virtual environment from the install step,
and point `PYTHON_BIN` at that venv Python in the env file.

#### Miner

```bash
cp deploy/miner.pm2.env.example miner.pm2.env
$EDITOR miner.pm2.env

python -m tag101.deploy.pm2_node start \
  --role miner \
  --name sn101-miner \
  --env-file miner.pm2.env
```

#### Validator

```bash
cp deploy/validator.pm2.env.example validator.pm2.env
$EDITOR validator.pm2.env

python -m tag101.deploy.pm2_node start \
  --role validator \
  --name sn101-validator \
  --env-file validator.pm2.env
```

#### Useful commands

```bash
python -m tag101.deploy.pm2_node restart --role miner --name sn101-miner --env-file miner.pm2.env
python -m tag101.deploy.pm2_node status --name sn101-miner
pm2 logs sn101-miner
python -m tag101.deploy.pm2_node stop --name sn101-miner
```
