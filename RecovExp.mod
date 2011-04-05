COMMENT
Time-stamp: <2011-04-05 16:09:44 marek>
Author: Marek Rudnicki

Description:
Single exponential recovery synapse.

ENDCOMMENT


NEURON {
    POINT_PROCESS RecovExp
    RANGE tau, e, i
    RANGE k, tau_rec, U
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
    tau_rec = 10 (ms)
    U = 0.7
}

ASSIGNED {
    v (mV)
    i (nA)
    G (umho)
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

    G = G*(1-U)*exp(-(t-t0)/tau_rec) + weight*(1-exp(-(t-t0)/tau_rec))

    state_discontinuity(g, g + G)

    t0 = t

    : printf("%5.1f\t%8.5g\t%g\n", t, G, g)
}
