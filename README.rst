cochlear_nucleus
================

Computational models of `globular bushy cells`_ (GBCs) in the `ventral
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

Install the dependencies:

- Python 2.7
- Numpy
- NEURON (with Python bindings): http://neuron.yale.edu/neuron/
- Brian: http://briansimulator.org/


Clone the git repo with::

  git clone https://github.com/mrkrd/cochlear_nucleus.git
  cd cochlear_nucleus

Compile NEURON's models::

  make

or manually::

  cd cochlear_nucleus/nrn && nrnivmodl
  cd -

Install the package itself::

  python setup.py develop --user



Citing
------

A manuscript with simulations using *cochlear_nucleus* hast been
accepted for publication in Frontiers in Computational Neuroscience.


Acknowledgments
---------------

This work was supported by the German Research Foundation (DFG) within
the Priority Program “Ultrafast and temporally precise information
processing: normal and dysfunctional hearing” SPP 1608 (HE6713), the
Technische Universität München within the funding programme Open
Access Publishing, and the German Federal Ministry of Education and
Research within the Munich Bernstein Center for Computational
Neuroscience (reference number 01GQ1004B).


License
-------

The project is licensed under the GNU General Public License v3 or
later (GPLv3+).
