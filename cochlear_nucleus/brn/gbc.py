#!/usr/bin/env python

from __future__ import division

import numpy as np
import random
import pandas as pd

import brian
from brian import mV, pF, ms, siemens, nA, amp, nS, uohm, second



def make_gbc_group(num, celsius=37):

    C = 12*pF
    Eh = -43*mV
    EK = -77*mV # -70mV in orig py file, but -77*mV in mod file
    El = -65*mV
    ENa = 50*mV

    nf = 0.85 # proportion of n vs p kinetics
    zss = 0.5 # steady state inactivation of glt

    q10 = 3.**((celsius - 22)/10.)
    T10 = 10.**((celsius - 22)/10.)



    q10_gbar = 1.5

    gnabar = calc_tf(q10_gbar, celsius) * 2500 * nS
    gkhtbar = calc_tf(q10_gbar, celsius) * 150 * nS
    gkltbar = calc_tf(q10_gbar, celsius) * 200 * nS
    ghbar = calc_tf(q10_gbar, celsius) * 20 * nS
    gl = calc_tf(q10_gbar, celsius) * 2 * nS


    eqs = """
    dvm/dt = (ileak + ina + ikht + iklt + ih + i_syn) / C : volt
    vu = vm/mV : 1 # unitless v
    """


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
    eqs += eqs_na


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
    eqs += eqs_kht


    # Ih channel (subthreshold adaptive, non-inactivating)
    eqs_ih="""
    ih = ghbar*r*(Eh-vm) : amp
    dr/dt=q10*(rinf-r)/rtau : 1
    rinf = 1. / (1+exp((vu + 76.) / 7.)) : 1
    rtau = ((100000. / (237.*exp((vu+60.) / 12.) + 17.*exp(-(vu+60.) / 14.))) + 25.)*ms : ms
    """
    eqs += eqs_ih


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
    eqs += eqs_klt


    # Leak
    eqs_leak="ileak = gl*(El-vm) : amp"
    eqs += eqs_leak



    ### Excitatory synapse
    # Q10 for synaptic decay calculated from \cite{Postlethwaite2007}
    Tf = calc_tf(q10=0.75, celsius=celsius, ref_temp=37)
    taue_syn = 0.2 * Tf * ms
    taui_syn = 9.03 * calc_tf(q10=0.75, celsius=celsius, ref_temp=34) * ms # \cite{Xie2013}
    eqs_syn = """
    i_syn = ge_syn*(0*mV - vm) + gi_syn*(-77*mV - vm): amp
    dge_syn/dt = -ge_syn/taue_syn : siemens
    dgi_syn/dt = -gi_syn/taui_syn : siemens
    """
    eqs += eqs_syn



    if celsius < 37:
        refractory = 0.7*ms
    else:
        refractory = 0.5*ms


    group = brian.NeuronGroup(
        N=num,
        model=eqs,
        threshold=brian.EmpiricalThreshold(threshold=-20*mV, refractory=refractory),
        implicit=True,
    )


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



make_gbcs = make_gbc_group



def synaptic_weight(
        pre,
        post,
        convergence,
        synapse='tonic'
):
    """Calculate synaptic weight for the give types of neuron gruops and
    convergence patterns.

    """
    weight = np.nan

    if (pre, post, synapse) == ('anf', 'gbc', 'tonic'):
        if convergence in (20, (20, 0, 0)):
            # derived for sposnt rate of 7.5 spikes/s
            weight = 8.043e-9 * siemens

    if np.isnan(weight):
        raise RuntimeError("Unknown synaptic weight.")

    return weight


def _calc_synaptic_weight(endbulb_class, convergence, anf_type, weights, celsius):

    anf_type_idx = {'hsr': 0, 'msr': 1, 'lsr': 2}[anf_type]

    ### Use precalculated weights
    if weights is None:
        assert celsius == 37 # default weights were calculated at 37C
        ws = _default_weights[ (endbulb_class, convergence) ]
        w = ws[ anf_type_idx ]

    elif isinstance(weights, float) or isinstance(weights, int):
        w = weights

    elif isinstance(weights, tuple):
        assert len(weights) == 3
        w = weights[anf_type_idx]

    else:
        raise RuntimeError("Unknown weight format.")

    return w * siemens





def calc_tf(q10, celsius, ref_temp=22):
    tf = q10 ** ((celsius - ref_temp)/10.0)
    return tf
