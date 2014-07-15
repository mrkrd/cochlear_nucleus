
NEURON {
    SUFFIX na_rothman93
    USEION na READ ena WRITE ina
    RANGE gnabar, ina
    GLOBAL minf, hinf, mtau, htau, q10
    THREADSAFE
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    v (mV)
    celsius (degC)
    gnabar=.120 (mho/cm2) <0,1e9>
    q10 : 3.0
}

STATE {
    m h
}

ASSIGNED {
    ena (mV)
    ina (mA/cm2)
    minf hinf
    mtau (ms)
    htau (ms)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar*m*m*h*(v - ena)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
}

FUNCTION malpha(v(mV)) (/ms) {
    LOCAL Tf
    Tf = q10^((celsius - 22(degC))/10(degC))
    malpha = 0.36 * Tf * expM1( -(v+49), 3 )
    :malpha = 0.36 * Tf * (v + 49) / (1 - exp(-(v+49)/3))
}

FUNCTION mbeta(v(mV)) (/ms) {
    LOCAL Tf
    Tf = q10^((celsius - 22(degC))/10(degC))
    mbeta = 0.4 * Tf * expM1( (v+58), 20 )
    :mbeta = -0.4 * Tf * (v + 58) / (1 - exp((v+58)/20))
}

FUNCTION halpha(v(mV)) (/ms) {
    LOCAL Tf, T10
    Tf = q10^((celsius - 22(degC))/10(degC))
    T10 = 10^((celsius - 22(degC))/10(degC))
    halpha = 2.4 * Tf / (1 + exp((v+68)/3)) +  0.8 * T10 / (1 + exp(v+61.3))
}

FUNCTION hbeta(v(mV)) (/ms) {
    LOCAL Tf
    Tf = q10^((celsius - 22(degC))/10(degC))
    hbeta = 3.6 * Tf / (1 + exp(-(v+21)/10))
}

FUNCTION expM1(x,y) {
    if (fabs(x/y) < 1e-6) {
	expM1 = y*(1 - x/y/2)
    } else {
	expM1 = x/(exp(x/y) - 1)
    }
}

PROCEDURE rates(v(mV)) {
    LOCAL a, b

    TABLE minf, hinf, mtau, htau DEPEND celsius FROM -100 TO 100 WITH 200
    mtau = 1/(malpha(v) + mbeta(v))
    minf = malpha(v)/(malpha(v) + mbeta(v))

    htau = 1/(halpha(v) + hbeta(v))
    hinf = halpha(v)/(halpha(v) + hbeta(v))
}
