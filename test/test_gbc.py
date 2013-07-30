#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd
import brian
from brian import second

from scipy import interpolate
import numpy.testing as npt

import cochlear_nucleus as cn


def test_synaptic_connections():

    anf_trains = pd.DataFrame([
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'msr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'msr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'lsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'lsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'lsr'},
    ])

    anfs = cn.ANFs(anf_trains)

    gbcs = cn.GBCs_RothmanManis2003(
        cfs=[100,100],
        convergences=(2,1,1),
    )

    gbcs.connect_anfs(
        anfs,
        weights=(0.3, 0.2, 0.1),
        recycle=False
    )

    for obj in gbcs.brian_objects:
        if isinstance(obj, brian.Synapses):

            weights = obj.weight.to_matrix()

            npt.assert_array_equal(
                np.nansum(weights, axis=0),
                np.array([0.9, 0.9])
            )





def _double_exp_recovery_synapse(ts, k, U, tau_fast, tau_slow):

    dts = np.diff(ts)

    Gmax = 1

    G = [Gmax]

    for dt in dts:
        Gfast = G[-1]*(1-U)*np.exp(-dt/tau_fast) + Gmax*(1-np.exp(-dt/tau_fast))
        Gslow = G[-1]*(1-U)*np.exp(-dt/tau_slow) + Gmax*(1-np.exp(-dt/tau_slow))
        G.append( k*Gfast + (1-k)*Gslow )

    return np.array(G)


def test_yang2009():
    tmax = 0.1

    ts = np.arange(0, tmax, 10e-3)

    ws = _double_exp_recovery_synapse(
        ts=ts,
        k=0.6,
        U=0.47,
        tau_fast=26e-3,
        tau_slow=1
    )



    anf_trains = pd.DataFrame([
        {'cf': 100, 'duration': tmax, 'spikes': ts, 'type': 'hsr'},
        {'cf': 100, 'duration': tmax, 'spikes': ts+2e-3, 'type': 'hsr'},
    ])
    anfs = cn.ANFs(anf_trains)


    gbcs = cn.GBCs_RothmanManis2003(
        cfs=[100],
        convergences=(2,0,0),
        endbulb_class='yang2009'
    )

    gbcs.connect_anfs(
        anfs,
        weights=(1,0,0)
    )



    for obj in gbcs.brian_objects:
        if isinstance(obj, brian.Synapses):
            synapses = obj


    g_syn = brian.StateMonitor(
        synapses,
        'g_syn',
        record=[1]              # `1' because the 2nd synapse
                                # (index=1) gets the first ANF while
                                # (randomly) connection ANF
    )
    g_syn_tot = brian.StateMonitor(
        gbcs.group,
        'g_syn_tot',
        record=True
    )

    net = brian.Network(
        gbcs.brian_objects,
        anfs.brian_objects,
        g_syn,
        g_syn_tot
    )
    net.run(
        tmax*second,
        report='text',
        report_period=1
    )


    g_syn_interp = interpolate.interp1d(
        g_syn.times,
        g_syn.values[0]
    )
    g_syn_tot_interp = interpolate.interp1d(
        g_syn_tot.times,
        g_syn_tot.values[0]
    )


    npt.assert_array_almost_equal(
        g_syn_interp(ts),
        ws
    )
    npt.assert_array_almost_equal(
        g_syn_tot_interp(ts),
        ws
    )

    # import matplotlib.pyplot as plt
    # g_syn.plot()
    # g_syn_tot.plot()
    # plt.plot(ts,ws)
    # plt.show()




def main():
    pass

if __name__ == "__main__":
    main()
