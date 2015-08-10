from __future__ import print_function, absolute_import, division


__author__ = "Marek Rudnicki"
__version__ = "0.1"



from . gbc import (
    make_gbc_group,
    make_gbcs,
    synaptic_weight,
)

from . anf import (
    make_anf_group,
    make_anfs
)

import brian
from brian import (
    second,
    defaultclock
)




def set_fs(fs):
    brian.defaultclock.dt = (1/fs) * second

def reset_defaultclock():
    brian.defaultclock.t = 0 * second

reset = reset_defaultclock


def run(duration, objects, **kwargs):
    """Run Brian simulation

    Parameters
    ----------
    duration : float
        Duration of the simulation in seconds.
    objects : list
        A collection of Brian objects to be simulated.

    """

    brian.defaultclock.t = 0 * second

    net = brian.Network(objects)

    kwargs.setdefault('report', 'text')
    kwargs.setdefault('report_period', 1)
    net.run(
        duration*second,
        **kwargs
    )
