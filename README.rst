cochlear_nucleus
================

.. image:: https://zenodo.org/badge/24233/mrkrd/cochlear_nucleus.svg
   :target: https://zenodo.org/badge/latestdoi/24233/mrkrd/cochlear_nucleus

A Python package implementing computational models of the `globular
bushy cells`_ (GBCs) in the mammalian `ventral cochlear nucleus`_.
Phenomenological models of endbulbs of Held with with short term
synaptic depressions are also included.

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

Please use our DOI when citing the models (click on the link for a
full reference):

.. image:: https://zenodo.org/badge/24233/mrkrd/cochlear_nucleus.svg
   :target: https://zenodo.org/badge/latestdoi/24233/mrkrd/cochlear_nucleus


We also published a manuscript with simulations of GBCs using this
software:

`Rudnicki M and Hemmert W (2017)`_. *High entrainment constrains
synaptic depression levels of an in vivo globular bushy cell
model*. Front. Comput. Neurosci. 11:16. doi:10.3389/fncom.2017.00016

.. _`Rudnicki M and Hemmert W (2017)`: http://journal.frontiersin.org/article/10.3389/fncom.2017.00016/abstract


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
