#!/usr/bin/env python

from __future__ import division
__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd

import neuron
from neuron import h


import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class GBC_Point(object):
    _default_weights = (

        {('10%-depressing', (16, 2, 2)): (0.0072790147819572215,
                                          0.01300363406907902,
                                          0.025077761219112135),
         ('10%-depressing', (24, 3, 3)): (0.0060803036590587464,
                                          0.011064608209339638,
                                          0.022960611280205795),
         ('10%-depressing', (32, 4, 4)): (0.0052627911610534182,
                                          0.009997284283591602,
                                          0.019772102754783479),
         ('10%-depressing', (40, 5, 5)): (0.0047530380948505235,
                                          0.0093045639569898642,
                                          0.018217731766975283),
         ('tonic', (0, 0, 20)): (0.0, 0.0, 0.070062207003387347),
         ('tonic', (0, 0, 40)): (0.0, 0.0, 0.084179665808960011),
         ('tonic', (16, 2, 2)): (0.007038794817791418,
                                 0.01266342935321116,
                                 0.02541172424059597),
         ('tonic', (20, 0, 0)): (0.0066033045079881593, 0.0, 0.0),
         ('tonic', (24, 3, 3)): (0.0058733536098521466,
                                 0.010682710448933506,
                                 0.021856493947204871),
         ('tonic', (32, 4, 4)): (0.0051942288176696858,
                                 0.009887290059422231,
                                 0.019580587912241685),
         ('tonic', (40, 0, 0)): (0.0047561806622803005, 0.0, 0.0),
         ('tonic', (40, 5, 5)): (0.0046037072220965133,
                                 0.0093309748057562245,
                                 0.017105117399478547),
         ('yang2009', (16, 2, 2)): (0.014024066512624741,
                                          0.035801613002810206,
                                          0.21464383648564361),
         ('yang2009', (24, 3, 3)): (0.014151826854560337,
                                          0.013762257387782693,
                                          0.10069232021044561),
         ('yang2009', (32, 4, 4)): (0.012441810052544041,
                                          0.013691620281564799,
                                          0.086407868314042346),
         ('yang2009', (40, 5, 5)): (0.011215341103431862,
                                          0.011607518306086639,
                                          0.089115665231745828)}


    )



    def __init__(
            self,
            convergence=(0,0,0),
            cf=1000,
            endbulb_class="tonic",
            threshold=-20,
            record_voltages=False
    ):

        log.info("GBC temperature: {} C".format(h.celsius))

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
        self.soma.e_pas = -65


        q10 = 1.5
        for seg in self.soma:
            seg.na_rothman93.gnabar = calc_tf(q10) * calc_conductivity_cm2(2500e-9, 12e-12)
            seg.kht.gkhtbar = calc_tf(q10) * calc_conductivity_cm2(150e-9, 12e-12)
            seg.klt.gkltbar = calc_tf(q10) * calc_conductivity_cm2(200e-9, 12e-12)
            seg.ih.ghbar = calc_tf(q10) * calc_conductivity_cm2(20e-9, 12e-12)
            seg.pas.g = calc_tf(q10) * calc_conductivity_cm2(2e-9, 12e-12)


        ### Seting up synapses
        self._are_weights_set = False
        self._make_endbulbs(convergence, endbulb_class)
        self.cf = float(cf)


        ### Netcon for recording spikes
        self._probe = h.NetCon(self.soma(0.5)._ref_v, None, threshold, 0, 0,
                               sec=self.soma)
        self._spikes = h.Vector()
        self._probe.record(self._spikes)



        ### Optional membrane potential recording
        if record_voltages:
            log.info("Recording voltages is ON")

            voltages = h.Vector()

            voltages.record(self.soma(0.5)._ref_v)
            self._voltages = voltages

        else:
            self._voltages = None




    def get_trains(self):
        assert h.t != 0, "Time is 0 (did you run the simulation already?)"

        trains = pd.DataFrame([
            {
                'spikes': np.array(self._spikes)*1e-3,
                'duration': h.t*1e-3,
                'type': 'gbc',
                'cf': self.cf
            }
        ])

        return trains


    def get_voltages(self):
        voltages = np.array(self._voltages) * 1e-3
        return voltages



    def _make_endbulbs(self, convergence, endbulb_class):
        """ convergence: (hsr, msr, lsr) """
        assert isinstance(convergence, tuple)

        hsr_num, msr_num, lsr_num = convergence

        # Endbulbs will be stored here
        self._endbulbs = []


        if endbulb_class in ("expsyn", "non-depressing", "tonic"):
            endbulb_class = "tonic"
            EndbulbClass = h.ExpSyn
            endbulb_pars = {'e': 0, 'tau': 0.2}
        elif endbulb_class in ("recov2exp", "yang2009"):
            endbulb_class = "yang2009"
            EndbulbClass = h.Recov2Exp
            endbulb_pars = {
                'e': 0,
                'tau': 0.2,
                'tau_fast': 26,
                'tau_slow': 1000,
                'U': 0.47,
                'k': 0.6
            }
        elif endbulb_class in ("little-depressing", "10%-depressing"):
            endbulb_class = '10%-depressing'
            EndbulbClass = h.RecovExp
            # tau_rec, U: calclated analytically for 10%
            # depression @ 300Hz
            endbulb_pars = {
                "e": 0,
                "tau": 0.2,
                "tau_rec": 5.7858390699913,
                "U": 0.086568968290663
            }
        elif endbulb_class == "20%-depressing":
            EndbulbClass = h.RecovExp
            # tau_rec, U: calclated analytically for 20%
            # depression @ 300Hz
            endbulb_pars = {
                "e": 0,
                "tau": 0.2,
                "tau_rec": 6.7600326478197,
                "U": 0.15934371552475
            }
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



    def set_endbulb_weights(self, weights):
        self._are_weights_set = True

        wh, wm, wl = weights

        for bulb in self._endbulbs:
            if bulb['type'] == 'hsr':
                bulb['con'].weight[0] = wh
            elif bulb['type'] == 'msr':
                bulb['con'].weight[0] = wm
            elif bulb['type'] == 'lsr':
                bulb['con'].weight[0] = wl




    def load_anf_trains(self, anf):

        hsr_idx = np.where(
            (anf['type'] == 'hsr') & (anf['cf'] == self.cf)
        )[0].tolist()

        msr_idx = np.where(
            (anf['type'] == 'msr') & (anf['cf'] == self.cf)
        )[0].tolist()

        lsr_idx = np.where(
            (anf['type'] == 'lsr') & (anf['cf'] == self.cf)
        )[0].tolist()


        # Feed each synapse with a proper ANF train
        for bulb in self._endbulbs:
            if bulb['type'] == 'hsr':
                i = hsr_idx.pop(
                    np.random.randint(0, len(hsr_idx))
                )
                bulb['spikes'] = anf['spikes'][i]

            elif bulb['type'] == 'msr':
                i = msr_idx.pop(
                    np.random.randint(0, len(msr_idx))
                )
                bulb['spikes'] = anf['spikes'][i]

            elif bulb['type'] == 'lsr':
                i = lsr_idx.pop(
                    np.random.randint(0, len(lsr_idx))
                )
                bulb['spikes'] = anf['spikes'][i]





    def init(self):
        assert self._are_weights_set, "Synaptic weights not set, use gbc.set_endbulb_weights()"

        for bulb in self._endbulbs:
            assert bulb.has_key('spikes'), "***** Not all endbulbs loaded with ANF spikes! *****"

            for sp in bulb['spikes']:
                bulb['con'].event(float(sp)*1e3)




def calc_tf(q10, ref_temp=22):
    tf = q10 ** ((h.celsius - ref_temp)/10.0)
    return tf


def calc_conductivity_cm2(conductance, capacity):
    cm = 0.9e-6                 # [F/cm2]
    area = capacity / cm        # [cm2]

    conductivity = conductance / area # [S/cm2]
    return conductivity
