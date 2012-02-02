#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np

import brian
from brian import mV, pF, ms, nS, nA, amp, uS, uohm
from brian.library import synapses

class GBCs(object):
    def __init__(self, group=None, cfs=None, convergences=None):
        assert cfs is not None, "Must give CF list"
        self.cfs = cfs

        assert convergences is not None, "Must provide ANF convergence"
        self.convergences = convergences

        if group is None:
            self.group = self._make_gbcs(len(cfs))
        else:
            assert False, 'not implemented'
            self.group = group




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

        def Tf(q10, ref_temp=22):
            return q10 ** ((celsius - ref_temp)/10.0)

        gnabar = Tf(1.4) * 1000 * nS
        gkhtbar = Tf(1.4) * 150 * nS
        gkltbar = Tf(1.4) * 200 * nS
        ghbar = Tf(1.4) * 20 * nS
        gl = Tf(1.4) * 2 * nS


        # Classical Na channel
        eqs_na="""
        ina = gnabar*m**3*h*(ENa-vm) : amp
        dm/dt=q10*(minf-m)/mtau : 1
        dh/dt=q10*(hinf-h)/htau : 1
        minf = 1./(1+exp(-(vu + 38.) / 7.)) : 1
        hinf = 1./(1+exp((vu + 65.) / 6.)) : 1
        mtau =  ((10. / (5*exp((vu+60.) / 18.) + 36.*exp(-(vu+60.) / 25.))) + 0.04)*ms : ms
        htau =  ((100. / (7*exp((vu+60.) / 11.) + 10.*exp(-(vu+60.) / 25.))) + 0.6)*ms : ms
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


        group = brian.NeuronGroup(num, eqs, implicit=True)

        ### Set initial conditions
        group.vm = El
        group.r = 1. / (1+ np.exp((El/mV + 76.) / 7.))
        group.m = 1./(1+np.exp(-(El/mV + 38.) / 7.))
        group.h = 1./(1+np.exp((El/mV + 65.) / 6.))
        group.w = (1. / (1 + np.exp(-(El/mV + 48.) / 6.)))**0.25
        group.z = zss + ((1.-zss) / (1 + np.exp((El/mV + 71.) / 10.)))
        group.n = (1 + np.exp(-(El/mV + 15) / 5.))**-0.5
        group.p = 1. / (1 + np.exp(-(El/mV + 23) / 6.))

        return group



def main():
    brian.defaultclock.dt = 0.025*ms

    gbcs = GBCs(cfs=[100], convergences=[(3,2,1), (3,2,1)])


    generator = brian.SpikeGeneratorGroup(1, [(0,30*ms),(0,35*ms),(0,40*ms)])
    connection = brian.Connection(generator, gbcs.group, 'ge', weight=0.5*uS)

    M = brian.StateMonitor(gbcs.group, 'vm', record=True)

    net = brian.Network(gbcs.group, generator, connection, M)
    net.run(50*ms, report='text', report_period=1) # Go to rest

    M.plot()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
