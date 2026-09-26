#########################################################################
# qos.pxd - pyslurm slurmdbd qos api
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
from pyslurm.slurm cimport (
    slurmdb_qos_rec_t,
    slurmdb_qos_cond_t,
    slurmdb_destroy_qos_rec,
    slurmdb_destroy_qos_cond,
    slurmdb_qos_get,
    slurm_preempt_mode_num,
    list_t,
    try_xmalloc,
)
from pyslurm.db.util cimport (
    SlurmList,
    SlurmListItem,
    make_char_list,
)
from pyslurm.db.connection cimport Connection
from pyslurm.utils cimport cstr
from pyslurm.utils.uint cimport u16_set_bool_flag

cdef _set_qos_list(list_t **in_list, vals, QualitiesOfService data)


cdef class QualitiesOfService(dict):
    pass


cdef class QualityOfServiceFilter:
   cdef slurmdb_qos_cond_t *ptr

   cdef public:
       names
       ids
       descriptions
       preempt_modes
       with_deleted


cdef class QualityOfService:
    cdef slurmdb_qos_rec_t *ptr

    @staticmethod
    cdef QualityOfService from_ptr(slurmdb_qos_rec_t *in_ptr)
