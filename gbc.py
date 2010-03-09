#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-03-09 10:24:56 marek>

# Description:

from __future__ import division
import numpy as np
import os

import neuron
from neuron import h


class GBC_Point(object):
    def __init__(self, anf_num, cf, endbulb_class=h.ExpSyn, endbulb_pars=None,
                 threshold=-20):


        # Soma parameters from (Rothman & Manis 2003)
        self.soma = h.Section()
        totcap = 12                             # pF
        soma_area = totcap * 1e-6 / 1.0         # cm2
        Lstd = 1e4 * np.sqrt(soma_area / np.pi) # um

        self.soma.insert('na_rothman93')
        self.soma.insert('klt')
        self.soma.insert('kht')
        self.soma.insert('ih')
        self.soma.insert('pas')

        self.soma.L = Lstd
        self.soma.Ra = 150.0
        self.soma.diam = Lstd
        self.soma.cm = 1
        self.soma.ek = -77
        self.soma.ena = 50

        q10 = 2.0
        for seg in self.soma:
            seg.na_rothman93.gnabar = self._Tf(q10) * 0.35
            seg.klt.gkltbar = self._Tf(q10) * self._nstomho(200, soma_area)
            seg.kht.gkhtbar = self._Tf(q10) * self._nstomho(150, soma_area)
            seg.ih.ghbar = self._Tf(q10) * self._nstomho(20, soma_area)
            seg.pas.g = self._Tf(q10) * self._nstomho(2, soma_area)


        # Seting up synapses
        self._make_endbulbs(anf_num, endbulb_class, endbulb_pars)
        self.cf = float(cf)


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


    _endbulb_type  = [('typ', 'S3'),
                      ('syn', object),
                      ('con', object),
                      ('spikes', object)]

    def _make_endbulbs(self, anf_num, endbulb_class=h.ExpSyn, endbulb_pars=None):
        """ anf_num: (hsr, msr, lsr) """
        assert isinstance(anf_num, tuple)

        hsr_num, msr_num, lsr_num = anf_num

        types = ['hsr' for each in range(hsr_num)] + \
            ['msr' for each in range(msr_num)] + \
            ['lsr' for each in range(lsr_num)]

        endbulbs = []

        for typ in types:
            syn = endbulb_class(self.soma(0.5))
            con = h.NetCon(None, syn)

            endbulbs.append((typ, syn, con, None))

        self.endbulbs = np.rec.array(endbulbs, dtype=self._endbulb_type)

        if endbulb_pars is not None:
            self.set_endbulb_pars(endbulb_pars)


    def set_endbulb_pars(self, endbulb_pars):
        assert isinstance(endbulb_pars, dict)

        for key in endbulb_pars:
            [setattr(syn, key, endbulb_pars[key]) for syn in self.endbulbs['syn']]



    def set_endbulb_weights(self, w):
        if isinstance(w, float) or isinstance(w, int):
            for con in self.endbulbs['con']:
                con.weight[0] = w

        elif isinstance(w, dict):
            for typ in w:
                q = (self.endbulbs['typ'] == typ)
                for con in self.endbulbs[q]['con']:
                    con.weight[0] = w[typ]

        elif isinstance(w, tuple):
            for typ,wi in zip(('hsr','msr','lsr'),w):
                q = (self.endbulbs['typ'] == typ)
                for con in self.endbulbs[q]['con']:
                    con.weight[0] = wi


        elif isinstance(w, list):
            assert len(w) == len(self.endbulbs)
            for wi,con in zip(w, self.endbulbs['con']):
                con.weight[0] = wi

        else:
            assert False




    def load_anf_trains(self, anf):

        # Feed each synapse with a proper ANF train
        for bulb in self.endbulbs:
            # Find the first matching ANF train
            try:
                idx = np.where((anf.typ==bulb.typ) & (anf.cf==self.cf))[0][0]
            except IndexError:
                print("***** Probably not enough ANF spike trains. *****")
                raise

            # Copy ANF train to the endbulb
            bulb.spikes = anf[idx].spikes

            # Mark the train as `deleted'
            anf[idx].typ = 'del'



    def init(self):
        for endbulb in self.endbulbs:
            for sp in endbulb.spikes:
                endbulb.con.event(float(sp))




def main():
    import matplotlib.pyplot as plt

    h.celsius = 37

    gbc = GBC_Point((2,1,1), cf=1000)
    print gbc.endbulbs

    pars = {'e':0, 'tau':0.5}
    gbc.set_endbulb_pars(pars)
    for syn in gbc.endbulbs['syn']:
        print "tau:", syn.tau

    weights = (0.006, 0.005, 0.003)
    gbc.set_endbulb_weights(weights)
    for bulb in gbc.endbulbs:
        print "weight:", bulb['typ'], bulb['con'].weight[0]


    anf_type = [('typ', 'S3'), ('cf', float), ('id', int), ('spikes', object)]
    anf = [('hsr', 1000, 0, np.array([10,20])),
           ('hsr', 1000, 1, np.array([30,40])),
           ('hsr', 3333, 0, np.array([50,60])),
           ('msr', 1000, 0, np.array([70,80])),
           ('msr', 2222, 0, np.array([90,00])),
           ('lsr', 1000, 0, np.array([60,50]))]
    anf = np.rec.array(anf, anf_type)

    gbc.load_anf_trains(anf)

    v = h.Vector()
    v.record(gbc.soma(0.5)._ref_v)

    neuron.init()
    gbc.init()
    neuron.run(100)

    print np.asarray(gbc.spikes)
    plt.plot(v)
    plt.show()


if __name__ == "__main__":
    main()
