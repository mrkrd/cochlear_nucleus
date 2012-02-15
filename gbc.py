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


    _default_weights = {
        ('tonic', (10,2,1)): ( 0.0079 , 0.0147 , 0.0332 , ),
        ('tonic', (17,3,3)): ( 0.0064 , 0.0126 , 0.0306 , ),
        ('tonic', (23,0,0)): ( 0.0059 , 0.0000 , 0.0000 , ),
        ('tonic', (27,4,3)): ( 0.0052 , 0.0101 , 0.0222 , ),
        ('tonic', (36,5,4)): ( 0.0045 , 0.0097 , 0.0179 , ),
        ('tonic', (47,0,0)): ( 0.0041 , 0.0000 , 0.0000 , ),
        ('tonic', (55,8,6)): ( 0.0035 , 0.0073 , 0.0131 , ),

        ('yang2009impact', (36,5,4)): ( 0.0111 , 0.0128 , 0.0614 , ),

        ('10%-depressing', (17,3,3)): ( 0.0066 , 0.0123 , 0.0305 , ),
        ('10%-depressing', (27,4,3)): ( 0.0053 , 0.0105 , 0.0216 , ),
        ('10%-depressing', (36,5,4)): ( 0.0046 , 0.0096 , 0.0183 , ),
        ('10%-depressing', (55,8,6)): ( 0.0036 , 0.0079 , 0.0139 , ),

        ('20%-depressing', (17,3,3)): ( 0.0068 , 0.0127 , 0.0325 , ),
        ('20%-depressing', (27,4,3)): ( 0.0055 , 0.0106 , 0.0238 , ),
        ('20%-depressing', (36,5,4)): ( 0.0047 , 0.0099 , 0.0205 , ),
        ('20%-depressing', (55,8,6)): ( 0.0038 , 0.0085 , 0.0155 , ),
    }




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


        q10_gbar = 1

        gnabar = Tf(q10_gbar) * 3000 * nS
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
                                  threshold=brian.EmpiricalThreshold(threshold=-10*mV, refractory=0.5*ms),
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
