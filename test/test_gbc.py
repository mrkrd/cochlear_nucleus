#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd
import brian

import cochlear_nucleus as cn

def test_synaptic_connections():

    anf_trains = pd.DataFrame([
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'hsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'msr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'msr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'lsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'lsr'},
        {'cf': 100, 'duration': 0.01, 'spikes': [], 'type': 'lsr'},
    ])

    anfs = cn.ANFs(anf_trains)

    gbcs = cn.GBCs_RothmanManis2003(
        cfs=[100,100],
        convergences=(2,1,1),
    )

    gbcs.connect_anfs(
        anfs,
        weights=(0.3, 0.2, 0.1),
        recycle=False
    )

    for obj in gbcs.brian_objects:
        if isinstance(obj, brian.Synapses):

            weights = obj.weight.to_matrix()

            assert np.all(
                np.nansum(weights, axis=0) == np.array([0.9, 0.9])
            )




def main():
    pass

if __name__ == "__main__":
    main()
