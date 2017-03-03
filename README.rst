cochlear_nucleus
================

Computational models of `globular bushy cells`_ (GBCs) in `ventral
cochlear nucleus`_.

.. _`globular bushy cells`: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2518325/
.. _`ventral cochlear nucleus`: https://en.wikipedia.org/wiki/Ventral_cochlear_nucleus


Usage
-----

Look for examples in the examples_ directory (e.g. nrn_gbc_demo.py and
nrn_gbc_demo_with_default_weights.py).

.. _examples: https://github.com/mrkrd/cochlear_nucleus/tree/master/examples


Installation
------------

Dependencies:

  - Python 2.7
  - Numpy
  - NEURON (http://neuron.yale.edu/neuron/)
  - Brian (http://briansimulator.org/)


Clone the repo with::

  git clone https://github.com/mrkrd/cochlear_nucleus.git
  cd cochlear_nucleus

Install the package::

  python setup.py install --user

or::

  python setup.py develop --user



Citing
------

A manuscript with simulations using *cochlear_nucleus* hast been
accepted for publication in Frontiers in Computational Neuroscience.



License
-------

The project is licensed under the GNU General Public License v3 or
later (GPLv3+).
