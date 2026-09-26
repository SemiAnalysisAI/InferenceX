#########################################################################
# slurmctld/enums.pyx - pyslurm slurmctld enums
#########################################################################
# Copyright (C) 2025 Toni Harzendorf <toni.harzendorf@gmail.com>
#
# This file is part of PySlurm
#
# PySlurm is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# PySlurm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with PySlurm; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# cython: c_string_type=unicode, c_string_encoding=default
# cython: language_level=3

from enum import IntEnum

class ShutdownMode(IntEnum):
    """Mode of operation for shutdown action."""
    ALL = 0
    CORE_FILE = 1
    CONTROLLER_ONLY = 2


# A bit hacky, but it works for now. Putting the docstring under the enum value
# does not work unfortunately.
ShutdownMode.ALL.__doc__ = "Shutdown all daemons (`slurmctld` and `slurmd`)"
ShutdownMode.CORE_FILE.__doc__ = "Shutdown only `slurmctld`, and create a coredump"
ShutdownMode.CONTROLLER_ONLY.__doc__ = "Shutdown only `slurmctld`, without a coredump"
