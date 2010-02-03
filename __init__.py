#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-02-03 14:14:45 marek>

# Description:

from __future__ import division
import numpy as np
from collections import namedtuple

import os
import platform
from neuron import h

lib_dir = os.path.dirname(__file__)
h.nrn_load_dll(lib_dir + '/' + platform.machine() + '/.libs/libnrnmech.so')


EndbulbPars = namedtuple('EndbulbPars', ['e', 'tau', 'tau_fast',
                                         'tau_slow', 'U', 'k',
                                         'threshold', 'delay', 'weight'])

synapse_type = [('syn', object), ('con', object), ('spikes', np.ndarray)]


class GBC_Point(object):
    def __init__(self, threshold=-20, q10=2, na_type='rothman93'):


        # ANF synapses
        self.anf_synapse_list = np.array([], dtype=synapse_type)


        # Soma: parameters from (Rothman & Manis 2003)
        self.soma = h.Section()
        totcap = 12                             # pF
        soma_area = totcap * 1e-6 / 1.0         # cm2
        Lstd = 1e4 * np.sqrt(soma_area / np.pi) # um

        self.soma.L = Lstd
        self.soma.Ra = 150.0

        if na_type == 'rothman93':
            self.soma.insert('na_rothman93')
        elif na_type == 'orig':
            self.soma.insert('na')
        self.soma.insert('klt')
        self.soma.insert('kht')
        self.soma.insert('ih')
        self.soma.insert('pas')

        # Setup mechanisms parameters
        self.soma.diam = Lstd
        self.soma.cm = 1
        self.soma.ek = -77
        self.soma.ena = 50

        for seg in self.soma:
            if na_type == 'rothman93':
                seg.na_rothman93.gnabar = self._Tf(q10) * 0.35
            elif na_type == 'orig':
                seg.na.gnabar = self._Tf(q10) * 0.35
            seg.klt.gkltbar = self._Tf(q10) * self._nstomho(200, soma_area)
            seg.kht.gkhtbar = self._Tf(q10) * self._nstomho(150, soma_area)
            seg.ih.ghbar = self._Tf(q10) * self._nstomho(20, soma_area)
            seg.pas.g = self._Tf(q10) * self._nstomho(2, soma_area)


        # Netcon for recording spikes
        self.soma.push()
        self._probe = h.NetCon(self.soma(0.5)._ref_v, None, threshold, 0, 0)
        h.pop_section()
        self.spikes = h.Vector()
        self._probe.record(self.spikes)


    def _nstomho(self, ns, area):
        """
        Rothman/Manis helper function.
        """
        return 1e-9 * ns / area


    def _Tf(self, q10, ref_temp=22):
        return q10 ** ((h.celsius - ref_temp)/10.0)



    def set_endbulb_pars(self, pars, idx=-1):
        """
        pars: synaptic parameters
        """
        anf = self.anf_synapse_list[idx]

        anf['syn'].e = pars.e
        anf['syn'].tau = pars.tau
        anf['syn'].tau_fast = pars.tau_fast
        anf['syn'].tau_slow = pars.tau_slow
        anf['syn'].U = pars.U
        anf['syn'].k = pars.k

        anf['con'].weight[0] = pars.weight


    def load_anf_train(self, train, pars):
        syn = h.Recov2Exp(self.soma(0.5))
        con = h.NetCon(None, syn)

        tmp = self.anf_synapse_list.tolist()
        tmp.append((syn, con, train))
        self.anf_synapse_list = np.array(tmp, dtype=synapse_type)
        self.set_endbulb_pars(pars)

    def clear_synapse_list(self):
        self.anf_synapse_list = np.array([], dtype=synapse_type)


    def init(self):
        for synapse in self.anf_synapse_list:
            for sp in synapse['spikes']:
                synapse['con'].event(float(sp))



def main():
    import matplotlib.pyplot as plt
    import neuron
    import thorns.waves as wv

    h.celsius = 37

    gbc = GBC_Point()
    pars = EndbulbPars(e=0, tau=0.2, tau_fast=15, tau_slow=1000,
                       U=0.5, k=0.5, threshold=0, delay=0.5,
                       weight=0.04)
    train = [8, 12]

    gbc.load_anf_train(train, pars)

    v = h.Vector()
    v.record(gbc.soma(0.5)._ref_v)

    neuron.init()
    gbc.init()
    neuron.run(20)

    plt.plot(wv.t(1/h.dt,v), v)
    plt.show()


if __name__ == "__main__":
    main()
