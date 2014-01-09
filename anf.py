#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np

import brian
from brian import second

def make_anf_group(anf_trains):

    times = []
    indices = []
    for i,spikes in enumerate(anf_trains['spikes']):
        times.append( spikes )
        indices.append( np.ones(len(spikes)) * i )


    indices = np.concatenate( indices )
    times = np.concatenate( times ) * second

    group = brian.SpikeGeneratorGroup(
        len(anf_trains),
        spiketimes=(indices, times)
    )

    return group


make_anfs = make_anf_group



class ANFs(object):
    def __init__(self, anf_trains):

        self.meta = anf_trains.drop('spikes', axis=1)


        ### cfs
        self.cfs = self.meta['cf']


        ### anf_type
        self.neuron_types = self.meta['type']


        ### spikes
        times = []
        indices = []
        for i,spikes in enumerate(anf_trains['spikes']):
            times.append( spikes )
            indices.append( np.ones(len(spikes)) * i )


        indices = np.concatenate( indices )
        times = np.concatenate( times ) * second

        self.group = brian.SpikeGeneratorGroup(
            len(anf_trains),
            spiketimes=(indices, times)
        )


        self.brian_objects = [self.group]



def main():
    import cochlea

    fs = 100e3
    t = np.arange(0, 0.1, 1/fs)
    s = np.sin(2 * np.pi * t * 1000)
    s = pycat.set_dbspl(s, 30)

    ear = cochlea.Zilany2009((3,2,1), cf=(80, 8000, 10))
    anf_raw = ear.run(s, fs)

    print anf_raw.dtype

    anfs = ANFs(anf_raw)



if __name__ == "__main__":
    main()
