# Sugarscape ABM

An extensible Agent-Based Model framework built from first principles, inspired by Epstein & Axtell's *Growing Artificial Societies* (1996) — but aiming to push beyond the classic Sugarscape.

## Motivation

A perfectly fair coin flip between equal agents, repeated thousands of times, produces startling inequality. No rigged rules, no talent differences, no institutional bias — just statistics. The wealth distribution converges to an exponential (Boltzmann) distribution with a Gini coefficient of 0.5, the same law governing energy distribution among gas molecules.

Yet real-world wealth Gini coefficients range from 0.54 (Japan) to 0.82 (Brazil). What mechanisms drive inequality beyond the statistical baseline? This project explores that question through simulation:

- **Additive vs multiplicative exchange** — fixed-amount trades yield exponential distributions; proportional trades yield power-law (Pareto) tails
- **Absorbing walls** — bankruptcy as permanent exit creates winner-take-all dynamics, yet paradoxically *lowers* measured Gini through survivorship bias
- **Taxation and redistribution** — how flat taxes reshape the equilibrium distribution
- **Game-theoretic strategies** — cooperation, defection, and the evolution of Tit-for-Tat
- **Multi-asset economies** — subjective valuation, endogenous prices, and consumption pressure
- **Institutions and shocks** — governments, central banks, natural disasters, and wars as external forces

## Architecture

Designed for incremental complexity — each phase adds new components without modifying existing ones.

```
Agent (ABC)
  ├── portfolio: {Asset → qty}     # multi-asset holdings
  ├── decide()                     # strategy logic
  └── step()                       # lifecycle (aging, metabolism)

Environment
  ├── agents: [Agent]              # dynamic pool (birth/death)
  ├── matcher                      # Factor → Combiner → Selector pipeline
  ├── interaction                  # pairwise interaction rules
  └── events: [Event]              # external forces (government, disasters)

Simulation
  └── tick loop: Events → Match → Interact → Lifecycle → Stats
```

**Matcher pipeline** — matching is decomposed into scoring (multiple Factors, asymmetric allowed), combination (weighted sum, min, product), and selection (random, probabilistic, greedy). Swap any layer independently.

**Organization as Agent** — groups of agents can form companies or institutions that participate in the economy as first-class entities, with internal governance and profit distribution.

## Quick Start

```bash
# Run Phase 1: random exchange, 500 agents, 5000 ticks
python run.py --config configs/phase1.yaml

# Run without visualization
python run.py --config configs/phase1.yaml --no-viz
```

### Configuration

All parameters are driven by YAML config:

```yaml
seed: 42
simulation:
  max_ticks: 5000
agents:
  count: 500
  initial_wealth: 100.0
interaction:
  type: random_exchange
  amount: 1.0
```

### Output

- Animated wealth distribution histogram (MP4) with theoretical exponential overlay
- Gini coefficient tracking over time
- Summary statistics (mean, median, min, max, percentiles)

## Phase 1 Results

500 agents, equal initial wealth (100 each), fair coin-flip exchange of 1 unit per interaction:

| Metric | 5,000 ticks | 100,000 ticks | Theory |
|--------|-------------|---------------|--------|
| Mean | 100.0 | 100.0 | 100.0 |
| Gini | 0.345 | 0.495 | 0.500 |
| Distribution | Approaching exponential | Exponential | Boltzmann |

Equilibration time scales as **(mean_wealth / delta)²** — independent of population size.

## Project Structure

```
sugarscape/
├── src/
│   ├── agent.py          # Portfolio, Agent ABC, CoinAgent
│   ├── interaction.py    # Interaction ABC, RandomExchange
│   ├── matcher.py        # Factor/Combiner/Selector pipeline
│   ├── environment.py    # Event ABC, Environment
│   ├── simulation.py     # Simulation engine
│   └── stats.py          # StatsCollector, Gini computation
├── viz/
│   └── animate.py        # Wealth distribution animation
├── configs/
│   └── phase1.yaml       # Phase 1 configuration
├── doc/
│   ├── questions.md      # Open research questions
│   └── *.pdf             # Reference papers
└── run.py                # Entry point
```

## References

- Epstein, J.M. & Axtell, R.L. (1996). *Growing Artificial Societies: Social Science from the Bottom Up*
- Drăgulescu, A. & Yakovenko, V.M. (2000). Statistical mechanics of money
- Peters, O. (2019). The ergodicity problem in economics. *Nature Physics*

## License

MIT
