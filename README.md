# desr

*desr* is a package used for Differental Equation Symmetry Reduction and is particularly useful for reducing the number of parameters in dynamical systems. It implements algorithms outlined by Evelyne Hubert and George Labahn <sup>[1](#myfootnote1)</sup>. The Masters dissertation [Differential Algebra and Applications](http://tanbur.github.io/desr/dissertation/differential_algebra_and_applications.pdf) that inspired this project places the algorithms into the theoretical framework of <em>differential algebraic geometry</em> and shows how to extend them to parameter reduction of arbitrary systems of partial differential equations, though this is not yet implemented.

<a name="myfootnote1">1</a>: Hubert, E., & Labahn, G. (2013). Scaling Invariants and Symmetry Reduction of Dynamical Systems. Foundations of Computational Mathematics, 13(4), 479–516. http://doi.org/10.1007/s10208-013-9165-9


## Getting Started

### Prerequisites

This package requires the Sympy package.

### Installing

To install, download the package and run:

`$ python setup.py install`

## Running the tests

Doctests are included in most files. To run them, simply run the module. E.g. "python -m doctest -v module.py"
To run all tests, use Sphinx's `make doctest`.

## Built With

* [Sphinx](http://www.sphinx-doc.org/en/stable/) - Used to generate the docs.

## Contributing

Submissions for contribution are always welcome.

## Authors

* **Richard Tanburn** - *Initial work* - [tanbur](https://github.com/tanbur)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Dr Heather Harrington and Dr Emilie Dufresne for their supervision of the dissertation.
* Thomas Close for writing his diophantine module, which is included in this package.
