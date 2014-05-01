#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

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



def get_weight(
        pre,
        post,
        convergence,
        synapse='tonic'
):

    if (pre,post,synapse,convergence) == ('hsr','gbc','tonic',(40,0,0)):
        weight = 5e-9 * siemens

    else:
        raise RuntimeError("Unknown synaptic parameters")

    return weight




_default_weights = {
    ('10%-depressing', (16, 2, 2)): (0.0072790147819572215*1e-6,
                                     0.01300363406907902*1e-6,
                                     0.025077761219112135*1e-6),
    ('10%-depressing', (24, 3, 3)): (0.0060803036590587464*1e-6,
                                     0.011064608209339638*1e-6,
                                     0.022960611280205795*1e-6),
    ('10%-depressing', (32, 4, 4)): (0.0052627911610534182*1e-6,
                                     0.009997284283591602*1e-6,
                                     0.019772102754783479*1e-6),
    ('10%-depressing', (40, 5, 5)): (0.0047530380948505235*1e-6,
                                     0.0093045639569898642*1e-6,
                                     0.018217731766975283*1e-6),
    ('tonic', (0, 0, 20)): (0.0, 0.0, 0.070062207003387347*1e-6),
    ('tonic', (0, 0, 40)): (0.0, 0.0, 0.084179665808960011*1e-6),
    ('tonic', (16, 2, 2)): (0.007038794817791418*1e-6,
                            0.01266342935321116*1e-6,
                            0.02541172424059597*1e-6),
    ('tonic', (20, 0, 0)): (0.0066033045079881593*1e-6, 0.0, 0.0),
    ('tonic', (24, 3, 3)): (0.0058733536098521466*1e-6,
                            0.010682710448933506*1e-6,
                            0.021856493947204871*1e-6),
    ('tonic', (32, 4, 4)): (0.0051942288176696858*1e-6,
                            0.009887290059422231*1e-6,
                            0.019580587912241685*1e-6),
    ('tonic', (40, 0, 0)): (0.0047561806622803005*1e-6, 0.0, 0.0),
    ('tonic', (40, 5, 5)): (0.0046037072220965133*1e-6,
                            0.0093309748057562245*1e-6,
                            0.017105117399478547*1e-6),
    ('yang2009', (16, 2, 2)): (0.014024066512624741*1e-6,
                               0.035801613002810206*1e-6,
                               0.21464383648564361*1e-6),
    ('yang2009', (24, 3, 3)): (0.014151826854560337*1e-6,
                               0.013762257387782693*1e-6,
                               0.10069232021044561*1e-6),
    ('yang2009', (32, 4, 4)): (0.012441810052544041*1e-6,
                               0.013691620281564799*1e-6,
                               0.086407868314042346*1e-6),
    ('yang2009', (40, 5, 5)): (0.011215341103431862*1e-6,
                               0.011607518306086639*1e-6,
                               0.089115665231745828*1e-6),
    ('electric', (20, 0, 0)): (0.00249200829105*1e-6, 0.0, 0.0)
}



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
