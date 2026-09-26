#########################################################################
# enums.pyx - pyslurm enums for various types
#########################################################################
# Copyright (C) 2026 Toni Harzendorf <toni.harzendorf@gmail.com>
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


from enum import auto
from pyslurm.utils.enums import SlurmEnum
from pyslurm cimport slurm


# TODO: Move everything enum related here.


class SchedulerType(SlurmEnum):
    SUBMIT = auto(), slurm.SLURMDB_JOB_FLAG_SUBMIT
    MAIN = auto(), slurm.SLURMDB_JOB_FLAG_SCHED
    BACKFILL = auto(), slurm.SLURMDB_JOB_FLAG_BACKFILL
    UNKNOWN = auto()


SchedulerType.SUBMIT.__doc__ = "Scheduled immediately on submit"
SchedulerType.MAIN.__doc__ = "Scheduled by the Main Scheduler"
SchedulerType.SUBMIT.__doc__ = "Scheduled by the Backfill Scheduler"


__all__ = [
    "SchedulerType",
]
