cochlear_nucleus
================

A Python package implementing computational models of the `globular
bushy cells`_ (GBCs) in the mammalian `ventral cochlear nucleus`_.
Phenomenological models of endbulbs of Held with with short term
synaptic depressions are also included.

.. _`globular bushy cells`: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2518325/
.. _`ventral cochlear nucleus`: https://en.wikipedia.org/wiki/Ventral_cochlear_nucleus


Usage
-----

Look for examples in the examples_ directory (e.g. nrn_gbc_demo.py and
nrn_gbc_demo_with_default_weights.py).  Note that some examples
require cochlea_ to be installed.

.. _examples: https://github.com/mrkrd/cochlear_nucleus/tree/master/examples
.. _cochlea: https://github.com/mrkrd/cochlea


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

Rudnicki M. and Hemmert W. (2017).  *High entrainment constrains
synaptic depression levels of an in vivo globular bushy cell model*.
Frontiers in Computational Neuroscience, Frontiers Media SA, 2017, 11,
pp. 1-11.
doi:10.3389/fncom.2017.00016
https://www.frontiersin.org/articles/10.3389/fncom.2017.00016/full

BibTeX entry::

  @Article{Rudnicki2017,
    author    = {Marek Rudnicki and Werner Hemmert},
    title     = {High Entrainment Constrains Synaptic Depression Levels of an In vivo Globular Bushy Cell Model},
    journal   = {Frontiers in Computational Neuroscience},
    year      = {2017},
    volume    = {11},
    pages     = {1--11},
    month     = {mar},
    doi       = {10.3389/fncom.2017.00016},
    publisher = {Frontiers Media {SA}},
    url       = {https://www.frontiersin.org/articles/10.3389/fncom.2017.00016/full},
  }



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
