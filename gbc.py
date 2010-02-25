#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-02-25 16:18:17 marek>

# Description:

from __future__ import division
import numpy as np
import os

import neuron
from neuron import h
from neuron import nrn

lib_dir = os.path.dirname(__file__)
neuron.load_mechanisms(lib_dir)


class GBC_Point(object):
    def __init__(self, anf_input=None, endbulb_type=h.ExpSyn, threshold=-20):


        # ANF synapses
        self.anf_synapse_list = np.array([], dtype=synapse_type)


        # Soma parameters from (Rothman & Manis 2003)
        self.soma = h.Section()
        totcap = 12                             # pF
        soma_area = totcap * 1e-6 / 1.0         # cm2
        Lstd = 1e4 * np.sqrt(soma_area / np.pi) # um

        self.soma.L = Lstd
        self.soma.Ra = 150.0
        self.soma.diam = Lstd
        self.soma.cm = 1
        self.soma.ek = -77
        self.soma.ena = 50

        if na_type == 'rothman93':
            self.soma.insert('na_rothman93')
        elif na_type == 'orig':
            self.soma.insert('na')
        self.soma.insert('klt')
        self.soma.insert('kht')
        self.soma.insert('ih')
        self.soma.insert('pas')

        q10 = 1
        for seg in self.soma:
            seg.na_rothman93.gnabar = self._Tf(q10) * 0.35
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


    _endbulb_type  = [('anf_type', 'S3'),
                      ('syn', object),
                      ('con', object),
                      ('spikes', object)]

    def make_endbulbs(anf_num, endbulb_type=h.ExpSyn):
        """ anf_num: (hsr, msr, lsr) """
        assert isinstance(anf_num, tuple)

        hsr_num, msr_num, lsr_num = anf_num
        anf_sum = np.sum(anf_num)

        syns = [endbulb_type(self.soma(0.5)) for each in range(anf_sum)]
        cons = [h.NetCon(None, syn) for each in range(anf_sum)]
        types = ['hsr' for each in range(hsr_num)] + \
            ['msr' for each in range(msr_num)] + \
            ['lsr' for each in range(lsr_num)]
        spikes = [None for each in range(anf_sum)]

        self._endbulbs = np.rec.fromarrays([types, syns, cons, spikes],
                                           names='anf_type,syn,con,spikes')



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




class GBC_Soma(nrn.Section):
    def __init__(self, threshold=-10):
        nrn.Section.__init__(self)

        totcap = 12                             # pF
        soma_area = totcap * 1e-6 / 1.0         # cm2
        Lstd = 1e4 * np.sqrt(soma_area / np.pi) # um
        q10 = 1

        self.L = Lstd
        self.Ra = 150.0

        self.insert('na_rothman93')
        self.insert('klt')
        self.insert('kht')
        self.insert('ih')
        self.insert('pas')

        # Setup mechanisms parameters
        self.diam = Lstd
        self.cm = 1
        self.ek = -77
        self.ena = 50

        for seg in self:
            seg.na_rothman93.gnabar = self._Tf(q10) * 0.35
            seg.klt.gkltbar = self._Tf(q10) * self._nstomho(200, soma_area)
            seg.kht.gkhtbar = self._Tf(q10) * self._nstomho(150, soma_area)
            seg.ih.ghbar = self._Tf(q10) * self._nstomho(20, soma_area)
            seg.pas.g = self._Tf(q10) * self._nstomho(2, soma_area)


        # Netcon for recording spikes
        self.push()
        self._probe = h.NetCon(self(0.5)._ref_v, None, threshold, 0, 0)
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


def main():
    import matplotlib.pyplot as plt

    h.celsius = 37

    soma = GBC_Soma()
    ic = h.IClamp(soma(0.5))
    ic.delay = 10
    ic.dur = 100
    ic.amp = 0.5

    v = h.Vector()
    v.record(soma(0.5)._ref_v)

    neuron.init()
    neuron.run(120)

    print np.asarray(soma.spikes)
    plt.plot(v)
    plt.show()


if __name__ == "__main__":
    main()
