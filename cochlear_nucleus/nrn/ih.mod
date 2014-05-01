TITLE jsr.mod  VCN conductances

COMMENT
Ih for VCN neurons - average from several studies in auditory neurons


Implementation by Paul B. Manis, April (JHU) and Sept, (UNC)1999.
revised 2/28/04 pbm

pmanis@med.unc.edu

ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
        (nA) = (nanoamp)
}

NEURON {
THREADSAFE
        SUFFIX ih
        NONSPECIFIC_CURRENT i
        RANGE ghbar, gh, ih
        GLOBAL rinf, rtau
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {
        v (mV)
        celsius = 22 (degC)
        dt (ms)
        ghbar = 0.00318 (mho/cm2) <0,1e9>
        eh = -43 (mV)
}

STATE {
        r
}

ASSIGNED {
	gh (mho/cm2)
	i (mA/cm2)
	rinf
    rtau (ms)
}

LOCAL rexp

BREAKPOINT {
	SOLVE states
    
	gh = ghbar*r
    i = gh*(v - eh)
    }

UNITSOFF

INITIAL {
    trates(v)
    r = rinf
}

PROCEDURE states() {  :Computes state variables m, h, and n
	trates(v)      :             at the current v and dt.
	r = r + rexp*(rinf-r)
VERBATIM
	return 0;
ENDVERBATIM
}

LOCAL q10
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

	q10 = 3^((celsius - 22)/10)
    rinf = 1 / (1+exp((v + 76) / 7))
    rtau = (100000 / (237*exp((v+60) / 12) + 17*exp(-(v+60) / 14))) + 25

}

PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL tinc
	TABLE rinf, rexp
	DEPEND dt, celsius FROM -200 TO 150 WITH 350

    rates(v)    : not consistently executed from here if usetable_hh == 1
        : so don't expect the tau values to be tracking along with
        : the inf values in hoc

	tinc = -dt * q10
	rexp = 1 - exp(tinc/rtau)
}

FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}

UNITSON
