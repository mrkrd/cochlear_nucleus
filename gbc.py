#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import random

import brian
from brian import mV, pF, ms, nS, nA, amp, uS, uohm, second
from brian.library import synapses

from scipy.sparse import lil_matrix


class GBCs_RothmanManis2003(object):
    def __init__(self, cfs, convergences, group=None, endbulb_classes='tonic'):

        # TODO: implement self.meta

        assert len(cfs) == len(convergences)

        self.cfs = cfs
        self.convergences = convergences

        if isinstance(endbulb_classes, str):
            self.endbulb_classes = [endbulb_classes for i in self.cfs]

        if group is None:
            self.group = self._make_gbcs(len(cfs))
        else:
            assert False, 'not implemented'
            self.group = group


        self._spike_monitor = brian.SpikeMonitor(self.group)

        self.brian_objects = [self.group, self._spike_monitor]


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
         ('yang2009impact', (16, 2, 2)): (0.014024066512624741,
                                          0.035801613002810206,
                                          0.21464383648564361),
         ('yang2009impact', (24, 3, 3)): (0.014151826854560337,
                                          0.013762257387782693,
                                          0.10069232021044561),
         ('yang2009impact', (32, 4, 4)): (0.012441810052544041,
                                          0.013691620281564799,
                                          0.086407868314042346),
         ('yang2009impact', (40, 5, 5)): (0.011215341103431862,
                                          0.011607518306086639,
                                          0.089115665231745828)}

    )



    def _make_gbcs(self, num):
        C=12*pF
        Eh=-43*mV
        EK=-77*mV # -70mV in orig py file, but -77*mV in mod file
        El=-65*mV
        ENa=50*mV

        nf = 0.85 # proportion of n vs p kinetics
        zss = 0.5 # steady state inactivation of glt
        celsius = 37. # temperature
        q10 = 3.**((celsius - 22)/10.)
        T10 = 10.**((celsius - 22)/10.)

        def Tf(q10, ref_temp=22):
            return q10 ** ((celsius - ref_temp)/10.0)


        q10_gbar = 1.5

        gnabar = Tf(q10_gbar) * 2500 * nS
        gkhtbar = Tf(q10_gbar) * 150 * nS
        gkltbar = Tf(q10_gbar) * 200 * nS
        ghbar = Tf(q10_gbar) * 20 * nS
        gl = Tf(q10_gbar) * 2 * nS


        # Rothman 1993 Na channel
        eqs_na="""
        ina = gnabar*m**3*h*(ENa-vm) : amp

        dm/dt = malpha * (1. - m) - mbeta * m : 1
        dh/dt = halpha * (1. - h) - hbeta * h : 1

        malpha = (0.36 * q10 * (vu+49.)) / (1. - exp(-(vu+49.)/3.)) /ms : 1/ms
        mbeta = (-0.4 * q10 * (vu+58.)) / (1. - exp((vu+58)/20.)) /ms : 1/ms

        halpha = 2.4*q10 / (1. + exp((vu+68.)/3.)) /ms  +  0.8*T10 / (1. + exp(vu + 61.3)) /ms : 1/ms
        hbeta = 3.6*q10 / (1. + exp(-(vu+21.)/10.)) /ms : 1/ms
        """


        # KHT channel (delayed-rectifier K+)
        eqs_kht="""
        ikht = gkhtbar*(nf*n**2 + (1-nf)*p)*(EK-vm) : amp
        dn/dt=q10*(ninf-n)/ntau : 1
        dp/dt=q10*(pinf-p)/ptau : 1
        ninf =   (1 + exp(-(vu + 15) / 5.))**-0.5 : 1
        pinf =  1. / (1 + exp(-(vu + 23) / 6.)) : 1
        ntau =  ((100. / (11*exp((vu+60) / 24.) + 21*exp(-(vu+60) / 23.))) + 0.7)*ms : ms
        ptau = ((100. / (4*exp((vu+60) / 32.) + 5*exp(-(vu+60) / 22.))) + 5)*ms : ms
        """

        # Ih channel (subthreshold adaptive, non-inactivating)
        eqs_ih="""
        ih = ghbar*r*(Eh-vm) : amp
        dr/dt=q10*(rinf-r)/rtau : 1
        rinf = 1. / (1+exp((vu + 76.) / 7.)) : 1
        rtau = ((100000. / (237.*exp((vu+60.) / 12.) + 17.*exp(-(vu+60.) / 14.))) + 25.)*ms : ms
        """

        # KLT channel (low threshold K+)
        eqs_klt="""
        iklt = gkltbar*w**4*z*(EK-vm) : amp
        dw/dt=q10*(winf-w)/wtau : 1
        dz/dt=q10*(zinf-z)/wtau : 1
        winf = (1. / (1 + exp(-(vu + 48.) / 6.)))**0.25 : 1
        zinf = zss + ((1.-zss) / (1 + exp((vu + 71.) / 10.))) : 1
        wtau = ((100. / (6.*exp((vu+60.) / 6.) + 16.*exp(-(vu+60.) / 45.))) + 1.5)*ms : ms
        ztau = ((1000. / (exp((vu+60.) / 20.) + exp(-(vu+60.) / 8.))) + 50)*ms : ms
        """

        # Leak
        eqs_leak="""
        ileak = gl*(El-vm) : amp
        """


        eqs="""
        dvm/dt = (ileak + ina + ikht + iklt + ih + ge_current) / C : volt
        vu = vm/mV : 1 # unitless v
        """
        eqs += eqs_leak + eqs_na + eqs_ih + eqs_klt + eqs_kht


        ### Excitatory synapse
        syn = synapses.exp_conductance(input='ge', E=0*mV, tau=0.2*ms)
        eqs += syn


        group = brian.NeuronGroup(N=num,
                                  model=eqs,
                                  threshold=brian.EmpiricalThreshold(threshold=-20*mV, refractory=0.5*ms),
                                  implicit=True)

        ### Set initial conditions
        group.vm = El
        group.r = 1. / (1+ np.exp((El/mV + 76.) / 7.))
        group.m = group.malpha / (group.malpha + group.mbeta)
        group.h = group.halpha / (group.halpha + group.halpha)
        group.w = (1. / (1 + np.exp(-(El/mV + 48.) / 6.)))**0.25
        group.z = zss + ((1.-zss) / (1 + np.exp((El/mV + 71.) / 10.)))
        group.n = (1 + np.exp(-(El/mV + 15) / 5.))**-0.5
        group.p = 1. / (1 + np.exp(-(El/mV + 23) / 6.))

        return group


    def connect_anfs(self, anfs, weights=None):

        types = ('hsr', 'msr', 'lsr')

        convergences = []
        for c in self.convergences:
            convergences.append( dict( zip( types, c ) ) )

        ws = np.zeros( (len(anfs.group), len(self.group)) )

        for cf,convergence,endbulb_class,col in zip(self.cfs,
                                                    convergences,
                                                    self.endbulb_classes,
                                                    ws.T):
            for typ in types:

                # Indexes of all available ANFs for a given CF nad TYPE
                idxs = np.where(
                    (anfs.meta['cf'] == cf) &
                    (anfs.meta['type'] == typ)
                )[0]

                idx = random.sample( idxs, convergence[typ] )

                col[idx] = self._calc_synaptic_weight(
                    endbulb_class=endbulb_class,
                    convergence=convergence,
                    anf_type=typ,
                    weights=weights
                )


        ws_sparse = lil_matrix( ws ) * uS
        connection = brian.Connection(anfs.group, self.group, 'ge')
        connection.connect( anfs.group, self.group, ws_sparse )

        self.brian_objects.append(connection)

        print self.brian_objects


    def _calc_synaptic_weight(self, endbulb_class, convergence, anf_type, weights):

        anf_type_idx = {'hsr': 0, 'msr': 1, 'lsr': 2}[anf_type]
        convergence = (convergence['hsr'], convergence['msr'], convergence['lsr'])

        ### Use precalculated weights
        if weights is None:
            ws = self._default_weights[ (endbulb_class, convergence) ]
            w = ws[ anf_type_idx ]

        elif isinstance(weights, float) or isinstance(weights, int):
            w = weights

        elif isinstance(weights, tuple):
            assert len(weights) == 3
            w = weights[anf_type_idx]

        return w



    def get_spikes(self):
        spiketimes = self._spike_monitor.spiketimes

        trains = []
        for i,spikes in spiketimes.items():
            trains.append( (spikes, self.group.clock.t, self.cfs[i], 'gbc') )

        trains = np.array(trains, dtype=[('spikes', np.ndarray),
                                         ('duration', float),
                                         ('cf', float),
                                         ('type', '|S3')])
        return trains





def main():
    import pycat
    from anf import ANFs

    brian.defaultclock.dt = 0.025*ms

    tmax = 0.05                 # [s]

    fs = 100e3
    t = np.arange(0, tmax, 1/fs)
    s = np.sin(2 * np.pi * t * 1000)
    s = pycat.set_dbspl(s, 30)

    ear = pycat.Zilany2009((3,2,1), cf=(80, 8000, 2))
    anf_raw = ear.run(s, fs)


    anfs = ANFs(anf_raw)
    cfs = np.unique(anfs.cfs)
    gbcs = GBCs_RothmanManis2003(cfs=cfs, convergences=[(3,2,1), (3,2,1)])

    gbcs.connect_anfs( anfs, weights=(0.05, 0.05, 0.05))


    M = brian.StateMonitor(gbcs.group, 'vm', record=True)

    net = brian.Network(gbcs.brian_objects, anfs.brian_objects, M)

    net.run(tmax*second, report='text', report_period=1) # Go to rest

    print gbcs.get_spikes()

    M.plot()
    brian.show()


if __name__ == "__main__":
    main()
