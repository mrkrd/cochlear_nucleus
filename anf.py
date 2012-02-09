#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np

import brian

class ANFs(object):
    def __init__(self, anf):
        ### cfs
        self.cfs = anf['cf']


        ### anf_type
        self.neuron_type = anf['type']


        ### spikes
        times = []
        indices = []
        for i,spikes in enumerate(anf['spikes']):
            times.append( spikes / 1e3)
            indices.append( np.ones(len(spikes)) * i )

        times = np.concatenate( times )
        indices = np.concatenate( indices )
        spiketimes = np.vstack( (indices, times) ).T


        self.group = brian.SpikeGeneratorGroup(
            len(anf),
            spiketimes=spiketimes
        )



        self.magic




def main():
    import pycat

    fs = 100e3
    t = np.arange(0, 0.1, 1/fs)
    s = np.sin(2 * np.pi * t * 1000)
    s = pycat.set_dbspl(s, 30)

    ear = pycat.Zilany2009((3,2,1), cf=(80, 8000, 10))
    anf_raw = ear.run(s, fs)

    print anf_raw.dtype

    anfs = ANFs(anf_raw)



if __name__ == "__main__":
    main()
