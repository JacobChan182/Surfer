# Surfer

A closed-loop, self-optimizing Wikipedia agent that navigates from a starting page toward **Counter-Strike** by clicking only in-page wiki links.

## How it works

1. **Start** at any English Wikipedia URL.
2. **Observe** the current page and score it for Counter-Strike-related keywords (Valve, FPS, esports, etc.).
3. **Learn** which link keywords correlate with higher rewards using an online contextual bandit.
4. **Act** by clicking the highest-scoring unvisited link on the page.
5. **Finish** when landing on the Counter-Strike article, or stop after 200 attempts.

With `--llm`, an NVIDIA LLM critic also evaluates each step, blends its progress score into learning, and distills keyword weight updates that **persist across runs** in `.surfer/weights.json`.

```
Step 12 | Counter-Strike: Global Offensive | reward=0.55 effective=0.68 | hits: counter-strike, valve
  [llm] progress=0.72 | On Valve page, one hop from CS franchise
  -> picked: Valve Corporation | top weights: valve=1.20, steam=0.85, game=0.40

Success in 18 steps!
Path: Dog -> Mammal -> Video game -> Valve Corporation -> Counter-Strike (video game)
```

## Install

```bash
cd Surfer
pip install -e ".[dev]"
```

Create a `.env` file with your NVIDIA API key (for `--llm`):

```
NVIDIA_API_KEY=nvapi-...
```

## Usage

```bash
# Basic run (bandit only, weights still persist)
python -m surfer --start "https://en.wikipedia.org/wiki/Dog"

# LLM critic + distillation (requires NVIDIA_API_KEY in .env)
python -m surfer --start "https://en.wikipedia.org/wiki/Dog" --llm --verbose

# Second run starts with distilled weights from prior runs
python -m surfer --start "https://en.wikipedia.org/wiki/Dog" --llm --verbose

# Custom model and blend settings
python -m surfer --start "https://en.wikipedia.org/wiki/Video_game" --llm \
  --llm-model meta/llama-3.1-8b-instruct --llm-blend 0.6

# Reset learned weights
python -m surfer --start "https://en.wikipedia.org/wiki/Dog" --reset-weights

# LLM picks from all page links (default for simulate.py)
python simulate.py --runs 10
python simulate.py --runs 10 --pick bandit   # switch back to bandit selection

# Single run with LLM picker
python -m surfer --start "https://en.wikipedia.org/wiki/Video_game" --pick llm --verbose
python -m surfer --start "..." --pick bandit --llm   # bandit picks, LLM still critiques

# Save full trace to JSON
python -m surfer --start "https://en.wikipedia.org/wiki/Dog" --json-out run.json

# View top 15 learned keyword weights
python -m surfer --top-weights
```

### Top weights

Distilled keyword weights live in `.surfer/weights.json`. To inspect the highest-scoring terms after training:

```bash
python -m surfer --top-weights
```

Example output:

```
Top 15 weights from .surfer/weights.json (runs=42, successes=28)
   1. esports       +1.7580
   2. first-person  +1.6750
   3. counter       +1.5070
   4. strike        +1.3040
   5. video         +1.1388
   ...
```

Use `--weights-file PATH` to read from a different file. In code:

```python
from surfer.weights_store import get_top_weights, load_weights, print_top_weights

print_top_weights()                              # print top 15
top = get_top_weights(load_weights().weights, 15)  # list of (word, weight)
```

### LLM / distillation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--llm` | off | Enable NVIDIA LLM critic each step |
| `--llm-model` | `meta/llama-3.1-8b-instruct` | NVIDIA NIM model id |
| `--llm-blend` | `0.6` | How much LLM progress score vs keyword reward |
| `--llm-delta-scale` | `0.5` | Scale for LLM weight_updates |
| `--weights-file` | `.surfer/weights.json` | Distilled weights path |
| `--top-weights` | off | Print top 15 saved weights and exit |
| `--reset-weights` | off | Delete saved weights before run |
| `--no-save-weights` | off | Skip persisting weights after run |

## Cross-run self-improvement

Each run loads weights from `.surfer/weights.json` (if present) and saves updated weights on exit. On success, run stats (`runs`, `successes`, `last_success_steps`) are incremented. The next run starts with boosted keywords the LLM and bandit learned (e.g. `valve`, `steam`, `shooter`) and penalized off-topic terms (e.g. `politician`).

## Reward tiers

| Tier   | Keywords                                              | Points |
|--------|-------------------------------------------------------|--------|
| Strong | counter-strike, cs2, csgo                             | +0.40  |
| Medium | valve, steam, esports, tactical shooter, source engine | +0.15  |
| Weak   | fps, multiplayer, bomb, terrorist, defusal, shooter   | +0.05  |

Landing on **Counter-Strike** or **Counter-Strike (video game)** ends the run immediately.

## Run tests

```bash
python -m pytest
```

## Architecture

```
surfer/
  agent.py          # closed-loop: observe → reward → learn → act
  wiki.py           # MediaWiki API client
  reward.py         # CS keyword scoring + terminal check
  bandit.py         # keyword-weight link selection
  keywords.py       # keyword tiers and tokenization
  nvidia_llm.py     # NVIDIA NIM API client
  llm_critic.py     # LLM progress scoring + weight deltas
  weights_store.py  # cross-run weight persistence
```

The bandit maintains per-word weights updated after each step. With `--llm`, the critic adds a second learning channel: explicit `weight_updates` that distill into the saved weights file for future runs.
