#!/usr/bin/env python

from __future__ import division
__author__ = "Marek Rudnicki"

import numpy as np

import thorns as th

import neuron
from neuron import h



class GBC_Point(object):
    _default_weights = (

        {('tonic', (16, 2, 2)): (0.0079352317824073794,
                                 0.014460717071929686,
                                 0.028795609225048407),
         ('tonic', (24, 3, 3)): (0.0065226792723531704,
                                 0.011928806133164414,
                                 0.025883112863915193),
         ('tonic', (32, 4, 4)): (0.0055830196681398575,
                                 0.011717093367539564,
                                 0.021686717286279321),
         ('tonic', (40, 5, 5)): (0.0049558232524809977,
                                 0.010417816654962499,
                                 0.017900221108361182)}


    )



    def __init__(self,
                 convergence=(0,0,0),
                 cf=1000,
                 endbulb_class="tonic",
                 endbulb_pars=None,
                 threshold=-20,
                 debug=True):

        if debug:
            print "GBC temperature:", h.celsius, "C"


        totcap = 12
        soma_area = totcap * 1e-6 / 1
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

        q10 = 1.5
        for seg in self.soma:
            seg.na_rothman93.gnabar = self._Tf(q10) * self._nstomho(2500, soma_area)
            seg.kht.gkhtbar = self._Tf(q10) * self._nstomho(150, soma_area)
            seg.klt.gkltbar = self._Tf(q10) * self._nstomho(200, soma_area)
            seg.ih.ghbar = self._Tf(q10) * self._nstomho(20, soma_area)
            seg.pas.g = self._Tf(q10) * self._nstomho(2, soma_area)
            seg.pas.e = -65



        # Seting up synapses
        self._are_weights_set = False
        self._make_endbulbs(convergence, endbulb_class, endbulb_pars)
        self.cf = float(cf)


        # Netcon for recording spikes
        self._probe = h.NetCon(self.soma(0.5)._ref_v, None, threshold, 0, 0,
                               sec=self.soma)
        self._spikes = h.Vector()
        self._probe.record(self._spikes)




    def _Tf(self, q10, ref_temp=22):
        return q10 ** ((h.celsius - ref_temp)/10.0)


    def get_spikes(self):
        assert h.t != 0, "Time is 0 (did you run the simulation already?)"

        train = np.array( [(np.array(self._spikes)*1e-3, h.t*1e-3, self.cf, 'gbc')],
                          dtype=[('spikes', np.ndarray),
                                 ('duration', float),
                                 ('cf', float),
                                 ('type', '|S3')]
        )
        return train



    def _nstomho(self, ns, area):
        """
        Rothman/Manis helper function.
        """
        mho = 1e-9 * ns / area
        # print ns, mho
        return mho



    def _make_endbulbs(self, convergence, endbulb_class, endbulb_pars=None):
        """ convergence: (hsr, msr, lsr) """
        assert isinstance(convergence, tuple)

        hsr_num, msr_num, lsr_num = convergence

        # Endbulbs will be stored here
        self._endbulbs = []


        if endbulb_class in ("expsyn", "non-depressing", "tonic"):
            endbulb_class = "tonic"
            EndbulbClass = h.ExpSyn
            if endbulb_pars is None:
                endbulb_pars = {'e': 0, 'tau': 0.2}
        elif endbulb_class in ("recov2exp", "yang2009impact"):
            endbulb_class = "yang2009impact"
            EndbulbClass = h.Recov2Exp
            if endbulb_pars is None:
                endbulb_pars = {'e': 0,
                                'tau': 0.2,
                                'tau_fast': 26,
                                'tau_slow': 1000,
                                'U': 0.47,
                                'k': 0.6}
        elif endbulb_class in ("little-depressing", "10%-depressing"):
            endbulb_class = '10%-depressing'
            EndbulbClass = h.RecovExp
            if endbulb_pars is None:
                # tau_rec, U: calclated analytically for 10%
                # depression @ 300Hz
                endbulb_pars = {"e": 0,
                                "tau": 0.2,
                                "tau_rec": 5.7858390699913,
                                "U": 0.086568968290663}
        elif endbulb_class == "20%-depressing":
            EndbulbClass = h.RecovExp
            if endbulb_pars is None:
                # tau_rec, U: calclated analytically for 20%
                # depression @ 300Hz
                endbulb_pars = {"e": 0,
                                "tau": 0.2,
                                "tau_rec": 6.7600326478197,
                                "U": 0.15934371552475}
        else:
            assert False, "Synapse \"%s\" not implemented"%endbulb_class

        anf_types = (['hsr' for each in range(hsr_num)] +
                     ['msr' for each in range(msr_num)] +
                     ['lsr' for each in range(lsr_num)])


        for typ in anf_types:
            syn = EndbulbClass(self.soma(0.5))

            self._endbulbs.append(
                {'type': typ,
                 'syn': syn,
                 'con': h.NetCon(None, syn)}
            )


        self.set_endbulb_pars(endbulb_pars)

        weight_key = (endbulb_class, convergence)
        if weight_key in self._default_weights:
            self.set_endbulb_weights(
                self._default_weights[weight_key]
            )


    def set_endbulb_pars(self, endbulb_pars):
        assert isinstance(endbulb_pars, dict)

        for bulb in self._endbulbs:
            for par,val in endbulb_pars.items():
                setattr(bulb['syn'], par, val)



    def set_endbulb_weights(self, w):
        self._are_weights_set = True

        if isinstance(w, float) or isinstance(w, int):
            for bulb in self._endbulbs:
                bulb['con'].weight[0] = w

        elif isinstance(w, tuple):
            wh, wm, wl = w

            for bulb in self._endbulbs:
                if bulb['type'] == 'hsr':
                    bulb['con'].weight[0] = wh
                elif bulb['type'] == 'msr':
                    bulb['con'].weight[0] = wm
                elif bulb['type'] == 'lsr':
                    bulb['con'].weight[0] = wl

        elif isinstance(w, list):
            assert len(w) == len(self._endbulbs)

            for wi,bulb in zip(w, self._endbulbs):
                bulb['con'].weight[0] = wi

        else:
            assert False



    def load_anf_trains(self, anf):

        hsr_idx = np.where(
            (anf['type']=='hsr') & (anf['cf']==self.cf)
        )[0].tolist()

        msr_idx = np.where(
            (anf['type']=='msr') & (anf['cf']==self.cf)
        )[0].tolist()

        lsr_idx = np.where(
            (anf['type']=='lsr') & (anf['cf']==self.cf)
        )[0].tolist()


        # Feed each synapse with a proper ANF train
        for bulb in self._endbulbs:
            if bulb['type'] == 'hsr':
                i = hsr_idx.pop(
                    np.random.randint(0, len(hsr_idx))
                )
                bulb['spikes'] = anf[i]['spikes']

            elif bulb['type'] == 'msr':
                i = msr_idx.pop(
                    np.random.randint(0, len(msr_idx))
                )
                bulb['spikes'] = anf[i]['spikes']

            elif bulb['type'] == 'lsr':
                i = lsr_idx.pop(
                    np.random.randint(0, len(lsr_idx))
                )
                bulb['spikes'] = anf[i]['spikes']





    def init(self):
        assert self._are_weights_set, "Synaptic weights not set, use gbc.set_endbulb_weights()"

        for bulb in self._endbulbs:
            assert bulb.has_key('spikes'), "***** Not all endbulbs loaded with ANF spikes! *****"

            for sp in bulb['spikes']:
                bulb['con'].event(float(sp)*1e3)









def main():
    import biggles

    h.celsius = 37

    gbc = GBC_Point((2,1,1), cf=1000)

    pars = {'e':0, 'tau':0.2}
    gbc.set_endbulb_pars(pars)


    weights = (0.06, 0.005, 0.003)
    gbc.set_endbulb_weights(weights)


    anf = [
        (np.array([10,20])*1e-3, 'hsr', 1000, 100e-3),
        (np.array([30,40])*1e-3, 'hsr', 1000, 100e-3),
        (np.array([50,60])*1e-3, 'hsr', 3333, 100e-3),
        (np.array([70,80])*1e-3, 'msr', 1000, 100e-3),
        (np.array([90,100])*1e-3, 'msr', 2222, 100e-3),
        (np.array([60,50])*1e-3, 'lsr', 1000, 100e-3)
    ]
    anf = np.rec.array(anf, dtype=[('spikes', np.ndarray),
                                   ('type', '|S3'),
                                   ('cf', float),
                                   ('duration', float)])

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
    print "GBC voltage", gbc.soma(0.5).v
    print
    print "Output spike trains"
    print gbc.get_spikes()
    print
    print "Rate:", th.stats.rate(gbc.get_spikes())
    print



    a = gbc.get_spikes()
    b = gbc.get_spikes()

    print np.concatenate([a, b]).dtype


if __name__ == "__main__":
    main()
