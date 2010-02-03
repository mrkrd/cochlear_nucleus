COMMENT
Time-stamp: <2010-02-03 11:59:40 marek>
Author: Marek Rudnicki

Description:
Implementation of the endbulb of held synaps as described in

``Synaptic depression in the localization of sound''
by: Daniel L. Cook, Peter C. Schwindt, Lucinda A. Grande, William J. Spain
Nature, Vol. 421, No. 6918. (02 January 2003), pp. 66-70.

ENDCOMMENT


NEURON {
    POINT_PROCESS Recov2Exp
    RANGE tau, e, i
    RANGE k, tau_fast, tau_slow, U
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (umho) = (micromho)
}

PARAMETER {
    e = 0 (mV)
    tau = 0.2 (ms) < 1e-9, 1e9 >
    k = 0.5
    tau_fast = 10 (ms)
    tau_slow = 100 (ms)
    U = 0.7
}

ASSIGNED {
    v (mV)
    i (nA)
    Gfast (umho)
    Gslow (umho)
}

STATE {
    g (umho)
}

INITIAL {
    g = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g*(v - e)
}

DERIVATIVE state {
    g' = -g/tau
}

NET_RECEIVE(weight (umho), G (umho), t0 (ms)) {
    INITIAL {
	t0 = 0
	G = weight / (1-U)
    }

    Gfast = G*(1-U)*exp(-(t-t0)/tau_fast) + weight*(1-exp(-(t-t0)/tau_fast))
    Gslow = G*(1-U)*exp(-(t-t0)/tau_slow) + weight*(1-exp(-(t-t0)/tau_slow))
    G = k*Gfast + (1-k)*Gslow

    state_discontinuity(g, g + G)

    t0 = t

    printf("%5.1f\t%8.5g\t%g\n", t, G, g)
}
