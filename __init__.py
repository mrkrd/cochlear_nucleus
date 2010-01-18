#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-01-18 20:05:14 marek>

# Description:

from __future__ import division
import numpy as np
from collections import namedtuple

import os
import platform
lib_dir = os.path.dirname(__file__)
from neuron import h
h.nrn_load_dll(lib_dir + '/' + platform.machine() + '/.libs/libnrnmech.so')


EndbulbPars = namedtuple('EndbulbPars', ['e', 'tau_1', 'tau_rec',
                                         'tau_facil', 'U',
                                         'threshold', 'delay', 'weight'])

InputSynapse = namedtuple('InputSynapse', 'syn con')


class GBC_Point(object):
    def __init__(self, threshold=-20, q10=2, na_type='rothman93'):


        # ANF synapses
        self.anf_synapse_list = []
        self.anf_trains = []


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



    def set_endbulb_pars(self, pars):
        """
        pars: synaptic parameters
        """
        for anf in self.anf_synapse_list:
            anf.syn.e = pars.e
            anf.syn.tau_1 = pars.tau_1
            anf.syn.tau_rec = pars.tau_rec
            anf.syn.tau_facil = pars.tau_facil
            anf.syn.U = pars.U

            anf.con.weight[0] = pars.weight


    def load_anf_trains(self, trains):
        for train in trains:
            syn = h.tmgsyn(self.soma(0.5))
            con = h.NetCon(None, syn)

            self.anf_synapse_list.append( InputSynapse(syn, con) )
            self.anf_trains.append(train)

    def clear_synapse_list(self):
        self.anf_synapse_list = []
        self.anf_trains = []


    def init(self):
        for train,synapse in zip(self.anf_trains, self.anf_synapse_list):
            for sp in train:
                synapse.con.event(float(sp))




def main():
    pass

if __name__ == "__main__":
    main()
