# micrograd-to-transformer

Building deep learning from the gradients up: a scalar autograd engine first, then character-level language models, then a small GPT.

> Follows Andrej Karpathy's [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero). All code here is written from scratch.

## Stages
| Stage | What it covers | Status |
|---|---|---|
| 1. micrograd | Scalar autograd engine, reverse-mode backprop, a tiny MLP library | In progress |
| 2. makemore | Bigram and MLP character models, batch norm, activation statistics | Planned |
| 3. GPT | Self-attention and a small transformer trained on text | Planned |

## What I built
- Stage 1: a `Value` class that records a computation graph and backpropagates through it, checked against finite differences

## Results
Coming as each stage is finished: training curves, final losses, and samples.

## How to run
```bash
git clone https://github.com/alfietry/micrograd-to-transformer.git
cd micrograd-to-transformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Repository layout
```
src/mg2t/micrograd/   stage 1: autograd engine (engine.py) and neural nets (nn.py)
tests/                gradient checks and training tests
notebooks/            exploration and figures
results/figures/      saved plots
NOTES.md              derivations
```

## Credits
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) (MIT, Copyright (c) 2022 Andrej Karpathy): the lecture series this project follows

## License
MIT. See [LICENSE](LICENSE).
