#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run a simulation of a globular bushy cell with ANF input.

1. Create sound (pure tone).
2. Run inner ear model.
3. Run GBC model.

"""


from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import cochlear_nucleus.nrn as cn
import cochlea

import thorns as th
import thorns.waves as wv


def main():
    fs = 100e3
    cf = 500
    convergence = (35, 0, 0)


    # Generate sound
    sound = wv.ramped_tone(
        fs=fs,
        freq=cf,
        duration=50e-3,
        pad=30e-3,
        dbspl=50
    )


    # Run inner ear model
    anf_trains = cochlea.run_zilany2014(
        sound=sound,
        fs=fs,
        cf=cf,
        anf_num=convergence,
        species='cat',
        seed=0
    )


    # Run GBC
    cn.set_celsius(37)
    cn.set_fs(fs)

    gbc = cn.GBC_Point(
        convergence=convergence,
        cf=cf,
        endbulb_class='tonic',
        record_voltages=True
    )

    gbc.load_anf_trains(anf_trains, seed=0)

    cn.run(
        duration=len(sound)/fs,
        objects=[gbc]
    )


    # Collect the results
    gbc_trains = gbc.get_trains()
    voltages = gbc.get_voltages()


    # Present the results
    print(gbc_trains)

    fig, ax = plt.subplots(2, 1)

    th.plot_raster(anf_trains, ax=ax[0])
    ax[0].set_title("ANF input")

    th.plot_signal(
        voltages,
        fs=fs,
        ax=ax[1]
    )

    th.show()


if __name__ == "__main__":
    main()
