#########################################################################
# helpers.pxd - basic helper functions
#########################################################################
# Copyright (C) 2023 Toni Harzendorf <toni.harzendorf@gmail.com>
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
from pyslurm.slurm cimport xfree, try_xmalloc, xmalloc, slurm_step_id_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from pyslurm.utils cimport cstr
from libc.stdlib cimport free
from libc.string cimport memset

cpdef uid_to_name(uint32_t uid, err_on_invalid=*, dict lookup=*)
cpdef gid_to_name(uint32_t gid, err_on_invalid=*, dict lookup=*)
cpdef gres_from_tres_dict(dict tres_dict)
cdef slurm_step_id_t init_step_id()
