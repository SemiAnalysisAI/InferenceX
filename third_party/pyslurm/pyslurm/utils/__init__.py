#########################################################################
# utils.py - pyslurm utility functions
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
"""pyslurm utility functions"""

# Utility time functions
from .ctime import (
    timestr_to_secs,
    timestr_to_mins,
    secs_to_timestr,
    mins_to_timestr,
    date_to_timestamp,
    timestamp_to_date,
)

# General utility functions
from .helpers import (
    uid_to_name,
    gid_to_name,
    user_to_uid,
    group_to_gid,
    expand_range_str,
    humanize,
    dehumanize,
    nodelist_from_range_str,
    nodelist_to_range_str,
)
