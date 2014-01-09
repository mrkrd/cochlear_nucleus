from gbc import (
    GBCs_RothmanManis2003,
    make_gbc_group,
    make_gbcs,
    get_weight,
)

from anf import (
    ANFs,
    make_anf_group,
    make_anfs
)

import brian
from brian import (
    second,
    defaultclock
)


import mrlib.thorns as th


brian.set_global_preferences(
    usecodegen=True
)


def set_fs(fs):
    brian.defaultclock.dt = (1/fs) * second

def reset_defaultclock():
    brian.defaultclock.t = 0*second

reset = reset_defaultclock

def run(groups, duration=None, **kwargs):

    brian.defaultclock.t = 0 * second

    brian_objects = []

    for group in groups:

        if (duration is None) and isinstance(group, ANFs):
            duration = th.get_duration(group.meta)

        brian_objects.append(group.brian_objects)


    net = brian.Network(brian_objects)

    kwargs.setdefault('report', 'text')
    kwargs.setdefault('report_period', 1)
    net.run(
        duration*second,
        **kwargs
    )
