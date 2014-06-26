#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cochlear_nucleus.nrn as cn



def main():
    cn.set_celsius(37)

    gbc = cn.GBC_Point(
        convergence=(2,1,1),
        cf=1000,
        endbulb_class='tonic',
        record_voltages=True
    )


    gbc.set_endbulb_weights(
        (6e-8, 5e-8, 3e-8)
    )


    anf_trains = pd.DataFrame([
        {'spikes':np.array([10,20])*1e-3, 'type':'hsr', 'cf':1000, 'duration':100e-3},
        {'spikes':np.array([30,40])*1e-3, 'type':'hsr', 'cf':1000, 'duration':100e-3},
        {'spikes':np.array([50,60])*1e-3, 'type':'hsr', 'cf':3333, 'duration':100e-3},
        {'spikes':np.array([70,80])*1e-3, 'type':'msr', 'cf':1000, 'duration':100e-3},
        {'spikes':np.array([80,90])*1e-3, 'type':'msr', 'cf':2222, 'duration':100e-3},
        {'spikes':np.array([60,50])*1e-3, 'type':'lsr', 'cf':1000, 'duration':100e-3}
    ])


    print("ANFs before")
    print(anf_trains)


    gbc.load_anf_trains(anf_trains)


    cn.run(
        duration=100e-3,
        objects=[gbc]
    )


    print()
    print("ANFs after")
    print(anf_trains)


    print()
    print("GBC spikes")
    print(gbc.get_trains())


    voltages = gbc.get_voltages()
    plt.plot(voltages)
    plt.show()




if __name__ == "__main__":
    main()
