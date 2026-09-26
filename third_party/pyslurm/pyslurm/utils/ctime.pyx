#########################################################################
# ctime.pyx - wrappers around slurm time functions
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

import datetime
from pyslurm.constants import UNLIMITED


def timestr_to_secs(timestr):
    """Convert Slurm Timestring to seconds

    Args:
        timestr (str):
            A Timestring compatible with Slurms time functions.

    Returns:
        (int): Amount of time in seconds
    """
    cdef:
        char *tmp = NULL
        uint32_t secs

    if timestr is None:
        return slurm.NO_VAL
    elif timestr == UNLIMITED or timestr.casefold() == "unlimited":
        return slurm.INFINITE

    if str(timestr).isdigit():
        timestr = "00:00:{}".format(timestr)

    tmp = cstr.from_unicode(timestr)
    secs = slurm.slurm_time_str2secs(tmp)

    if secs == slurm.NO_VAL:
        raise ValueError(f"Invalid Time Specification: {timestr}.")

    return secs


def timestr_to_mins(timestr):
    """Convert Slurm Timestring to minutes

    Args:
        timestr (str):
            A Timestring compatible with Slurms time functions.

    Returns:
        (int): Amount of time in minutes
    """
    cdef:
        char *tmp = NULL
        uint32_t mins

    if timestr is None:
        return slurm.NO_VAL
    elif str(timestr).isdigit():
        return timestr
    elif timestr == UNLIMITED or timestr.casefold() == "unlimited":
        return slurm.INFINITE

    tmp = cstr.from_unicode(timestr)
    mins = slurm.slurm_time_str2mins(tmp)

    if mins == slurm.NO_VAL:
        raise ValueError(f"Invalid Time Specification: {timestr}.")

    return mins


def secs_to_timestr(secs, default=None):
    """Parse time in seconds to Slurm Timestring

    Args:
        secs (int):
            Amount of seconds to convert 

    Returns:
        (str): A Slurm timestring
    """
    cdef char time_line[32]

    if secs == slurm.NO_VAL or secs is None:
        return default
    elif secs != slurm.INFINITE:
        slurm.slurm_secs2time_str(
            <time_t>secs,
            time_line,
            sizeof(time_line)
            )

        tmp = cstr.to_unicode(time_line)
        if tmp == "00:00:00":
            return None
        else:
            return tmp
    else:
        return UNLIMITED


def mins_to_timestr(mins, default=None):
    """Parse time in minutes to Slurm Timestring

    Args:
        mins (int):
            Amount of minutes to convert

    Returns:
        (str): A Slurm timestring
    """
    cdef char time_line[32]

    if mins == slurm.NO_VAL or mins is None:
        return default
    elif mins != slurm.INFINITE:
        slurm.slurm_mins2time_str(
            <uint32_t>mins,
            time_line,
            sizeof(time_line)
            )

        tmp = cstr.to_unicode(time_line)
        if tmp == "00:00:00":
            return None
        else:
            return tmp
    else:
        return UNLIMITED


def date_to_timestamp(date, on_nodate=0):
    """Parse Date to Unix timestamp

    Args:
        date (Union[str, int, datetime.datetime]):
            A date to convert to a Unix timestamp.

    Returns:
        (int): A unix timestamp
    """
    cdef:
        time_t tmp_time
        char* tmp_char = NULL

    if not date:
        # time_t of 0, so the option will be ignored by slurmctld
        return on_nodate
    elif str(date).isdigit():
        # Allow the user to pass a timestamp directly.
        return int(date)
    elif isinstance(date, datetime.datetime):
        # Allow the user to pass a datetime.datetime object.
        return int(date.timestamp())

    tmp_char = cstr.from_unicode(date)
    tmp_time = slurm.slurm_parse_time(tmp_char, 0)

    if not tmp_time:
        raise ValueError(f"Invalid Time Specification: {date}")

    return tmp_time


def timestamp_to_date(timestamp):
    """Parse Unix timestamp to Slurm Date-string

    Args:
        timestamp (int):
            A Unix timestamp that should be converted.

    Returns:
        (str): A Slurm date timestring
    """
    cdef:
        char time_str[32]
        time_t _time = <time_t>timestamp

    if _time == <time_t>slurm.NO_VAL:
        return None

    # slurm_make_time_str returns 'Unknown' if 0 or slurm.INFINITE
    slurm.slurm_make_time_str(&_time, time_str, sizeof(time_str))

    ret = cstr.to_unicode(time_str)
    if ret == "Unknown":
        return None

    return ret


def _raw_time(time, on_noval=None, on_inf=None):
    if time == slurm.NO_VAL or time == 0:
        return on_noval
    elif time == slurm.INFINITE:
        return on_inf
    else:
        return time
