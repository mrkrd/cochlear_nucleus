#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brian
from brian import siemens

import cochlea
import cochlear_nucleus.brn as cn
import thorns as th
import thorns.waves as wv


def main():

    fs = 100e3                  # s
    cf = 600                    # Hz
    duration = 50e-3            # s

    # Simulation sampling frequency
    cn.set_fs(40e3)             # Hz

    # Generate sound
    sound = wv.ramped_tone(
        fs=fs,
        freq=cf,
        duration=duration,
        dbspl=30,
    )

    # Generate ANF trains
    anf_trains = cochlea.run_zilany2014(
        sound=sound,
        fs=fs,
        anf_num=(300, 0, 0),    # (HSR, MSR, LSR)
        cf=cf,
        species='cat',
        seed=0,
    )

    # Generate ANF and GBC groups
    anfs = cn.make_anf_group(anf_trains)
    gbcs = cn.make_gbc_group(100)

    # Connect ANFs and GBCs
    synapses = brian.Connection(
        anfs,
        gbcs,
        'ge_syn',
    )

    convergence = 20

    weight = cn.synaptic_weight(
        pre='anf',
        post='gbc',
        convergence=convergence
    )

    synapses.connect_random(
        anfs,
        gbcs,
        p=convergence/len(anfs),
        fixed=True,
        weight=weight
    )

    # Monitors for the GBCs
    spikes = brian.SpikeMonitor(gbcs)

    # Run the simulation
    cn.run(
        duration=len(sound)/fs,
        objects=[anfs, gbcs, synapses, spikes]
    )

    # Present the results
    gbc_trains = th.make_trains(spikes)

    fig, ax = plt.subplots(2, 1)

    th.plot_raster(anf_trains, ax=ax[0])
    th.plot_raster(gbc_trains, ax=ax[1])

    plt.show()



if __name__ == "__main__":
    main()
