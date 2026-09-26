#########################################################################
# slurmctld/base.pxd - pyslurm slurmctld api functions
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

from pyslurm cimport slurm
from pyslurm.slurm cimport (
    slurm_conf_t,
    slurm_reconfigure,
    slurm_shutdown,
    slurm_ping,
    slurm_takeover,
    slurm_set_debugflags,
    slurm_set_debug_level,
    slurm_set_schedlog_level,
    slurm_set_fs_dampeningfactor,
)
from libc.stdint cimport uint16_t, uint64_t
from pyslurm.utils.uint cimport u16_parse
from pyslurm.utils cimport cstr


cdef class PingResponse:
    """Slurm Controller Ping response information

    Attributes:
        is_primary (bool):
            Whether this Slurm Controller is the primary Server.
        is_responding (bool):
            Whether this Slurm Controller actually responds to the ping.
        index (int):
            The index in the slurm.conf. For example, 0 means primary.
        hostname (str):
            Hostname of the Controller
        latency (float):
            The latency which the Controller responds with. This is in
            milliseconds.
    """
    cdef public:
        is_primary
        is_responding
        index
        hostname
        latency
