#########################################################################
# common/uint.pxd - functions dealing with parsing uint types
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
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cpdef u8(val, inf=*, noval=*, on_noval=*, zero_is_noval=*)
cpdef u16(val, inf=*, noval=*, on_noval=*, zero_is_noval=*)
cpdef u32(val, inf=*, noval=*, on_noval=*, zero_is_noval=*)
cpdef u64(val, inf=*, noval=*, on_noval=*, zero_is_noval=*)
cpdef u8_parse(uint8_t val, on_inf=*, on_noval=*, noval=*, zero_is_noval=*)
cpdef u16_parse(uint16_t val, on_inf=*, on_noval=*, noval=*, zero_is_noval=*)
cpdef u32_parse(uint32_t val, on_inf=*, on_noval=*, noval=*, zero_is_noval=*)
cpdef u64_parse(uint64_t val, on_inf=*, on_noval=*, noval=*, zero_is_noval=*)
cpdef u8_bool(val)
cpdef u16_bool(val)
cdef uint_set_bool_flag(flags, boolean, true_flag, false_flag=*)
cdef uint_parse_bool_flag(flags, flag, no_val)
cdef uint_parse_bool(val, no_val)
cdef uint_bool(val, no_val)
cdef u8_parse_bool(uint8_t val)
cdef u16_parse_bool(uint16_t val)
cdef u64_parse_bool_flag(uint64_t flags, flag)
cdef u64_set_bool_flag(uint64_t *flags, boolean, true_flag, false_flag=*)
cdef u32_parse_bool_flag(uint32_t flags, flag)
cdef u32_set_bool_flag(uint32_t *flags, boolean, true_flag, false_flag=*)
cdef u16_parse_bool_flag(uint16_t flags, flag)
cdef u16_set_bool_flag(uint16_t *flags, boolean, true_flag, false_flag=*)
cdef u8_parse_bool_flag(uint8_t flags, flag)
cdef u8_set_bool_flag(uint8_t *flags, boolean, true_flag, false_flag=*)
