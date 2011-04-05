#!/usr/bin/env python

from __future__ import division
__author__ = "Marek Rudnicki"

import numpy as np

import thorns as th

import neuron
from neuron import h


class GBC_Template(object):

    def _Tf(self, q10, ref_temp=22):
        return q10 ** ((h.celsius - ref_temp)/10.0)


    _endbulb_type  = [('typ', 'S3'),
                      ('syn', object),
                      ('con', object),
                      ('spikes', object)]


    def get_spikes(self):
        return np.array(self.spikes)


    def _nstomho(self, ns, area):
        """
        Rothman/Manis helper function.
        """
        return 1e-9 * ns / area



class GBC_Point(GBC_Template):
    _default_weights = {'non-depressing'+str((17,3,3)): (0.0057, 0.0125, 0.0355),
                        'non-depressing'+str((27,4,3)): (0.0048, 0.0095, 0.0221),
                        'non-depressing'+str((36,5,4)): (0.0041, 0.0083, 0.0176),
                        'yang2009impact'+str((17,4,3)): (0.0105, 0.0067, 0.2975),
                        'yang2009impact'+str((27,4,3)): (0.0086, 0.0062, 0.2439),
                        'yang2009impact'+str((36,5,4)): (0.0077, 0.0083, 0.1965),
                        }


    def __init__(self, convergence=(0,0,0), cf=1000,
                 endbulb_class="expsyn", endbulb_pars=None,
                 threshold=-20):

        print "GBC temperature:", h.celsius, "C"

        # soma_area = 2500        # um2
        # Lstd = np.sqrt(soma_area/np.pi)
        Lstd = 20

        self.soma = h.Section()

        self.soma.insert('na_rothman93')
        self.soma.insert('kht')
        self.soma.insert('klt')
        self.soma.insert('ih')
        self.soma.insert('pas')

        self.soma.L = Lstd
        self.soma.Ra = 150.0
        self.soma.diam = Lstd
        self.soma.cm = 0.9
        self.soma.ek = -77
        self.soma.ena = 50

        q10 = 1.4
        for seg in self.soma:
            seg.na_rothman93.gnabar = self._Tf(q10) * 0.35
            seg.kht.gkhtbar = self._Tf(q10) * 0.0125 #self._nstomho(150, soma_area)
            seg.klt.gkltbar = self._Tf(q10) * 0.0167 #self._nstomho(200, soma_area)
            seg.ih.ghbar = self._Tf(q10) * 0.00167 #self._nstomho(20, soma_area)
            seg.pas.g = self._Tf(q10) * 0.000167 #self._nstomho(2, soma_area)
            seg.pas.e = -65



        # Seting up synapses
        self._are_weights_set = False
        self._make_endbulbs(convergence, endbulb_class, endbulb_pars)
        self.cf = float(cf)


        # Netcon for recording spikes
        self._probe = h.NetCon(self.soma(0.5)._ref_v, None, threshold, 0, 0,
                               sec=self.soma)
        self.spikes = h.Vector()
        self._probe.record(self.spikes)





    def _make_endbulbs(self, convergence, endbulb_class, endbulb_pars=None):
        """ convergence: (hsr, msr, lsr) """
        assert isinstance(convergence, tuple)

        hsr_num, msr_num, lsr_num = convergence


        if endbulb_class in ("expsyn", "non-depressing"):
            endbulb_class = "non-depressing"
            EndbulbClass = h.ExpSyn
            if endbulb_pars is None:
                endbulb_pars = {'e': 0, 'tau': 0.2}
        elif endbulb_class in ("recov2exp", "yang2009impact"):
            endbulb_class = "yang2009impact"
            EndbulbClass = h.Recov2Exp
            if endbulb_pars is None:
                endbulb_pars = {'e': 0,
                                'tau': 0.2,
                                'tau_fast': 27,
                                'tau_slow': 1000,
                                'U': 0.47,
                                'k': 0.6}
        elif endbulb_class == "little-depressing":
            EndbulbClass = h.RecovExp
            if endbulb_pars is None:
                # tau_rec, U: calclated analytically for 10%
                # depression @ 300Hz
                endbulb_pars = {"e": 0,
                                "tau": 0.2,
                                "tau_rec": 5.7858390699913,
                                "U": 0.086568968290663}

        types = (['hsr' for each in range(hsr_num)] +
                 ['msr' for each in range(msr_num)] +
                 ['lsr' for each in range(lsr_num)])

        endbulbs = []

        for typ in types:
            syn = EndbulbClass(self.soma(0.5))
            con = h.NetCon(None, syn)
            endbulbs.append((typ, syn, con, None))

        self._endbulbs = np.array(endbulbs, dtype=self._endbulb_type)

        if endbulb_pars is not None:
            self.set_endbulb_pars(endbulb_pars)

        key = endbulb_class+str(convergence)
        if key in self._default_weights:
            self.set_endbulb_weights(self._default_weights[key])


    def set_endbulb_pars(self, endbulb_pars):
        assert isinstance(endbulb_pars, dict)

        for key in endbulb_pars:
            [setattr(syn, key, endbulb_pars[key]) for syn in self._endbulbs['syn']]



    def set_endbulb_weights(self, w):
        self._are_weights_set = True

        if isinstance(w, float) or isinstance(w, int):
            for con in self._endbulbs['con']:
                con.weight[0] = w

        elif isinstance(w, dict):
            for typ in w:
                q = (self._endbulbs['typ'] == typ)
                for con in self._endbulbs[q]['con']:
                    con.weight[0] = w[typ]

        elif isinstance(w, tuple):
            for typ,wi in zip(('hsr','msr','lsr'),w):
                q = (self._endbulbs['typ'] == typ)
                for con in self._endbulbs[q]['con']:
                    con.weight[0] = wi


        elif isinstance(w, list):
            assert len(w) == len(self._endbulbs)
            for wi,con in zip(w, self._endbulbs['con']):
                con.weight[0] = wi

        else:
            assert False



    def load_anf_trains(self, anf, random=False):

        hsr = anf.where(typ='hsr', cf=self.cf)
        msr = anf.where(typ='msr', cf=self.cf)
        lsr = anf.where(typ='lsr', cf=self.cf)

        # Feed each synapse with a proper ANF train
        for bulb in self._endbulbs:
            if bulb['typ'] == 'hsr':
                bulb['spikes'] = hsr.pop(random)
            if bulb['typ'] == 'msr':
                bulb['spikes'] = msr.pop(random)
            if bulb['typ'] == 'lsr':
                bulb['spikes'] = lsr.pop(random)



    def init(self):
        assert self._are_weights_set, "Synaptic weights not set, use gbc.set_endbulb_weights()"

        for endbulb in self._endbulbs:
            try:
                assert endbulb['spikes'] is not None
            except AssertionError:
                print("***** Not all endbulbs loaded with ANF spikes! *****")
                raise
            for sp in endbulb['spikes']:
                endbulb['con'].event(float(sp))









def main():
    import biggles

    h.celsius = 37

    gbc = GBC_Point((2,1,1), cf=1000)

    pars = {'e':0, 'tau':0.2}
    gbc.set_endbulb_pars(pars)


    weights = (0.06, 0.005, 0.003)
    gbc.set_endbulb_weights(weights)


    anf = th.SpikeTrains()
    anf.append(np.array([10,20]), typ='hsr', cf=1000)
    anf.append(np.array([30,40]), typ='hsr', cf=1000)
    anf.append(np.array([50,60]), typ='hsr', cf=3333)
    anf.append(np.array([70,80]), typ='msr', cf=1000)
    anf.append(np.array([90,00]), typ='msr', cf=2222)
    anf.append(np.array([60,50]), typ='lsr', cf=1000)

    print
    print "ANFs before"
    print anf

    gbc.load_anf_trains(anf)

    v = h.Vector()
    v.record(gbc.soma(0.5)._ref_v)

    neuron.init()
    gbc.init()
    neuron.run(100)

    print
    print "ANFs after"
    print anf

    biggles.plot(v)
    print
    print gbc.soma(0.5).v

if __name__ == "__main__":
    main()
