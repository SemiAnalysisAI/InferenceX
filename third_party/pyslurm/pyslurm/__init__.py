"""pyslurm package

pyslurm is a wrapper around the Slurm C-API.
"""
import os
import sys

sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL | os.RTLD_DEEPBIND)

from .version import __version__

from pyslurm import db
from pyslurm import utils
from pyslurm import constants

from pyslurm.enums import *

from pyslurm.core.job import (
    Job,
    Jobs,
    JobStep,
    JobSteps,
    JobSubmitDescription,
)
from pyslurm.core.node import Node, Nodes
from pyslurm.core.partition import Partition, Partitions
from pyslurm.core.reservation import (
    Reservation,
    Reservations,
    ReservationFlags,
    ReservationReoccurrence,
)
from pyslurm.core import error
from pyslurm.core.error import (
    PyslurmError,
    RPCError,
)
from pyslurm.core import slurmctld

# The old API in deprecated.pyx
from pyslurm.deprecated import *

# Initialize slurm api
from pyslurm.api import slurm_init, slurm_fini

_auto_init_disabled = bool(os.environ.get("PYSLURM_DISABLE_AUTO_INIT", False))
if not _auto_init_disabled:
    slurm_init()
