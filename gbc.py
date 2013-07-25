#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import random
import pandas as pd

import brian
from brian import mV, pF, ms, siemens, nA, amp, nS, uohm, second


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
    ('yang2009impact', (16, 2, 2)): (0.014024066512624741*1e-6,
                                     0.035801613002810206*1e-6,
                                     0.21464383648564361*1e-6),
    ('yang2009impact', (24, 3, 3)): (0.014151826854560337*1e-6,
                                     0.013762257387782693*1e-6,
                                     0.10069232021044561*1e-6),
    ('yang2009impact', (32, 4, 4)): (0.012441810052544041*1e-6,
                                     0.013691620281564799*1e-6,
                                     0.086407868314042346*1e-6),
    ('yang2009impact', (40, 5, 5)): (0.011215341103431862*1e-6,
                                     0.011607518306086639*1e-6,
                                     0.089115665231745828*1e-6),
    ('electric', (20, 0, 0)): (0.00249200829105*1e-6, 0.0, 0.0)
}



def _calc_synaptic_weight(endbulb_class, convergence, anf_type, weights, celsius):

    assert endbulb_class == 'tonic', "Only tonic synapse is implemented."

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
        raise RuntimeError, "Unknown weight format."

    return w * siemens





def calc_tf(q10, celsius, ref_temp=22):
    tf = q10 ** ((celsius - ref_temp)/10.0)
    return tf



class GBCs_RothmanManis2003(object):
    def __init__(
            self,
            cfs,
            convergences,
            endbulb_class='tonic',
            celsius=37.,
            group=None
    ):


        self._cfs = np.array(cfs)
        self._celsius = celsius


        if isinstance(convergences, tuple):
            assert len(convergences) == 3
            self._convergences = [convergences] * len(self._cfs)
        else:
            assert len(convergences) == len(self._cfs)
            self._convergences = convergences



        if isinstance(endbulb_class, str):
            self._endbulb_classes = [endbulb_class] * len(self._cfs)
        else:
            assert len(endbulb_class) == len(self._cfs)
            self._endbulb_classes = endbulb_class





        if group is None:
            self.group = self._make_gbcs(len(cfs))
        else:
            raise RuntimeError, 'not implemented'
            self.group = group





        self._spike_monitor = brian.SpikeMonitor(self.group)

        self.brian_objects = [self.group, self._spike_monitor]





    def _make_gbcs(self, num):
        C = 12*pF
        Eh = -43*mV
        EK = -77*mV # -70mV in orig py file, but -77*mV in mod file
        El = -65*mV
        ENa = 50*mV

        nf = 0.85 # proportion of n vs p kinetics
        zss = 0.5 # steady state inactivation of glt

        q10 = 3.**((self._celsius - 22)/10.)
        T10 = 10.**((self._celsius - 22)/10.)



        q10_gbar = 1.5

        gnabar = calc_tf(q10_gbar, self._celsius) * 2500 * nS
        gkhtbar = calc_tf(q10_gbar, self._celsius) * 150 * nS
        gkltbar = calc_tf(q10_gbar, self._celsius) * 200 * nS
        ghbar = calc_tf(q10_gbar, self._celsius) * 20 * nS
        gl = calc_tf(q10_gbar, self._celsius) * 2 * nS


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
        Tf = calc_tf(q10=0.75, celsius=self._celsius, ref_temp=37)
        e_syn = 0*mV
        tau_syn = 0.2 * Tf * ms
        eqs_syn = """
        i_syn = g_syn * (e_syn - vm) : amp
        dg_syn/dt = -g_syn/tau_syn : siemens
        """
        eqs += eqs_syn



        if self._celsius < 37:
            refractory = 0.7*ms
        else:
            refractory = 0.5*ms


        group = brian.NeuronGroup(
            N=num,
            model=eqs,
            threshold=brian.EmpiricalThreshold(threshold=-20*mV, refractory=refractory),
            implicit=True
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



    def connect_anfs(self, anfs, weights=None, recycle=True):

        random.seed(0)

        anf_types = {'hsr':0, 'msr':1, 'lsr':2}


        if isinstance(weights, tuple):
            assert len(weights) == 3
            weights = [weights] * len(self.group)
        elif weights is None:
            weights = [None] * len(self.group)
        else:
            assert len(weights) == len(self.group)


        active_anfs = np.ones(len(anfs.group), dtype=bool)


        synapses = brian.Synapses(
            anfs.group,
            self.group,
            model='weight:1',
            pre='g_syn+=weight',
        )


        cfs = np.array(anfs.meta['cf'])
        types = np.array(anfs.meta['type'])
        for gbc_idx in range(len(self.group)):
            for typ,typ_idx in anf_types.items():

                # Indexes of all active ANFs for a given CF and TYPE
                anf_idxs = np.where(
                    (cfs == self._cfs[gbc_idx]) &
                    (types == typ) &
                    active_anfs
                )[0]

                anf_idx = random.sample(
                    anf_idxs,
                    self._convergences[gbc_idx][typ_idx]
                )

                if not recycle:
                    active_anfs[anf_idx] = False

                for i in anf_idx:
                    synapses[i,gbc_idx] = True
                    weight = _calc_synaptic_weight(
                        endbulb_class=self._endbulb_classes[gbc_idx],
                        convergence=self._convergences[gbc_idx],
                        anf_type=typ,
                        weights=weights[gbc_idx],
                        celsius=self._celsius
                    )
                    synapses.weight[i,gbc_idx] = weight

        self.brian_objects.append(synapses)






    def get_trains(self):
        spiketimes = self._spike_monitor.spiketimes

        trains = []
        for i,spikes in spiketimes.items():
            trains.append({
                'spikes': spikes,
                'duration': self.group.clock.t,
                'cf': self._cfs[i],
                'type': 'gbc'
            })

        trains = pd.DataFrame(trains)

        return trains

    get_spikes = get_trains



def main():
    import cochlea
    from anf import ANFs

    brian.defaultclock.dt = 0.025*ms

    tmax = 0.05                 # [s]

    fs = 100e3
    t = np.arange(0, tmax, 1/fs)
    s = np.sin(2 * np.pi * t * 1000)
    s = cochlea.set_dbspl(s, 30)

    anf_trains = cochlea.run_zilany2009(
        sound=s,
        fs=fs,
        anf_num=(3,2,1),
        cf=(80, 8000, 2),
        seed=0
    )
    print(anf_trains)

    anfs = ANFs(anf_trains)
    cfs = np.unique(anfs.cfs)

    gbcs = GBCs_RothmanManis2003(
        cfs=cfs,
        convergences=(3,2,1),
    )

    gbcs.connect_anfs(
        anfs,
        weights=(0.05*1e-6, 0.05*1e-6, 0.05*1e-6),
        recycle=False
    )


    M = brian.StateMonitor(
        gbcs.group,
        'vm',
        record=True
    )

    net = brian.Network(
        gbcs.brian_objects,
        anfs.brian_objects,
        M
    )

    net.run(
        tmax*second,
        report='text',
        report_period=1
    )

    print gbcs.get_spikes()

    M.plot()
    brian.show()


if __name__ == "__main__":
    main()
