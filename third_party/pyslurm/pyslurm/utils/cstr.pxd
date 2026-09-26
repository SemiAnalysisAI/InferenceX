#########################################################################
# common/cstr.pxd - slurm string functions
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
from pyslurm.slurm cimport xfree, try_xmalloc, xmalloc
from libc.string cimport memcpy, strlen

cdef char *from_unicode(s)
cdef to_unicode(char *s, default=*)
cdef fmalloc(char **old, val)
cdef fmalloc2(char **p1, char **p2, val)
cdef free_array(char **arr, count)
cpdef list to_list(char *str_list, default=*, delim=*)
cdef list to_list_free(char **str_list)
cdef list to_list_with_count(char **str_list, cnt)
cdef from_list(char **old, vals, delim=*)
cdef from_list2(char **p1, char **p2, vals, delim=*)
cpdef dict to_dict(char *str_dict, str delim1=*, str delim2=*, def_value=*)
cdef from_dict(char **old, vals, prepend=*, str delim1=*, str delim2=*)
cpdef dict to_gres_dict(char *gres)
