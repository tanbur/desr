Introduction
============

*desr* is a package used for Differental Equation Symmetry Reduction and is particularly useful for reducing the number of parameters in dynamical systems.
It implements algorithms outlined by Evelyne Hubert and George Labahn :cite:`Hubert2013c`.

The Masters dissertation `Differential Algebra and Applications <http://tanbur.github.io/desr/dissertation/differential_algebra_and_applications.pdf>`_ that inspired this project places the algorithms into the theoretical framework of *differential algebraic geometry* and shows how to extend them to parameter reduction of arbitrary systems of partial differential equations, though this is not yet implemented.





Prerequisites
-------------

This package requires the numpy and sympy packages.

Installing
----------

Simply download this repository and add the path to your PYTHONPATH.

Running the tests
-----------------

Doctests are included in most files. To run them, simply run the module. E.g. "python -m doctest -v module.py"

Built With
----------

- `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ - Used to generate the docs.

Contributing
------------

Submissions for contribution are always welcome.

Authors
-------

- **Richard Tanburn** - *Initial work*

License
-------

This project is licensed under the Apache 2.0 License.

Acknowledgments
---------------

- Dr Heather Harrington and Dr Emilie Dufresne for their supervision of the dissertation.
- Thomas Close for writing his diophantine module, which is included in this package.

.. bibliography:: desr.bib