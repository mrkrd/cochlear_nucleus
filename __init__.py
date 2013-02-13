from gbc import GBCs_RothmanManis2003
from anf import ANFs

import brian
from brian import second

import marlib.thorns as th


brian.set_global_preferences(
    usecodegen=True
)


def set_fs(fs):
    brian.defaultclock.dt = (1/fs) * second


def run(groups, duration=None):

    brian.defaultclock.t = 0 * second

    brian_objects = []

    for group in groups:

        if (duration is None) and isinstance(group, ANFs):
            duration = th.get_duration(group.meta)

        brian_objects.append(group.brian_objects)


    net = brian.Network(brian_objects)

    net.run(
        duration*second,
        report='text',
        report_period=1
    )
