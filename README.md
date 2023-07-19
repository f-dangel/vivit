# <img alt="ViViT" src="./docs/rtd/assets/vivit_logo.svg" height="90"> ViViT: Curvature access through the generalized Gauss-Newton's low-rank structure

[![Python
3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
![tests](https://github.com/f-dangel/vivit/actions/workflows/test.yaml/badge.svg)

ViViT is a collection of numerical tricks to efficiently access curvature from
the generalized Gauss-Newton (GGN) matrix based on its low-rank structure.
Provided functionality includes computing
- GGN eigenvalues ([basic
  example](https://vivit.readthedocs.io/en/latest/basic_usage/example_eigvalsh.html#computing-ggn-eigenvalues))
- GGN eigenpairs (eigenvalues + eigenvector, [basic
  example](https://vivit.readthedocs.io/en/latest/basic_usage/example_eigh.html#computing-ggn-eigenpairs))
- 1ˢᵗ- and 2ⁿᵈ-order directional derivatives along GGN eigenvectors ([basic
  example](https://vivit.readthedocs.io/en/latest/basic_usage/example_directional_derivatives.html#computing-directional-derivatives-along-ggn-eigenvectors))
- Directionally damped Newton steps ([basic
  example](https://vivit.readthedocs.io/en/latest/basic_usage/example_directional_damped_newton.html#computing-directionally-damped-newton-steps))

These operations can also further approximate the GGN to reduce cost via
sub-sampling, Monte-Carlo approximation, and block-diagonal approximation.
- **Documentation:** https://vivit.readthedocs.io/en/latest/
- **Bug reports & feature requests:** https://github.com/f-dangel/vivit/issues

**How does it work?** ViViT uses and extends
 [BackPACK](https://github.com/f-dangel/backpack) for
 [PyTorch](https://github.com/pytorch/pytorch). The described functionality is
 realized through a combination of existing and new BackPACK extensions and
 hooks into its backpropagation.

## Installation

```bash
pip install vivit-for-pytorch
```

## Examples

Basic and advanced demos can be found in the
[documentation](https://vivit.readthedocs.io/en/latest/basic_usage/index.html).

## How to cite
If you are using ViViT, consider citing the [paper](https://openreview.net/pdf?id=DzJ7JfPXkE)
```

@article{dangel2022vivit,
  title =        {Vi{V}i{T}: Curvature Access Through The Generalized
                  Gauss-Newton{\textquoteright}s Low-Rank Structure},
  author =       {Felix Dangel and Lukas Tatzel and Philipp Hennig},
  journal =      {Transactions on Machine Learning Research (TMLR)},
  year =         2022,
}

```
