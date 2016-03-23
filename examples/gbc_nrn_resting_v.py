# -*- coding: utf-8 -*-

"""Run an “empty” simulation of a GBC and display the resting membrane
potential at 37°C.

"""
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

__author__ = "Marek Rudnicki"
__copyright__ = "Copyright 2015, Marek Rudnicki"
__license__ = "GPLv3+"


import cochlear_nucleus.nrn as cn
import thorns as th

def main():

    fs = 50e3
    duration = 1

    cn.set_fs(fs)
    cn.set_celsius(37)

    gbc = cn.GBC_Point(
        convergence=(0,0,0),
        cf=1e3,
        record_voltages=True,
    )

    gbc.set_endbulb_weights((0,0,0))

    cn.run(
        duration,
        [gbc]
    )

    v = gbc.get_voltages()

    print(v[-1])

    th.plot_signal(v, fs)
    th.show()

if __name__ == "__main__":
    main()
