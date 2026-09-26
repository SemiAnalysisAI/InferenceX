#########################################################################
# util.pyx - utility functions used to parse various job flags
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

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from pyslurm cimport slurm
from pyslurm.utils.uint import *
from pyslurm.utils.uint cimport *

# Note: Maybe consider using libslurmfull again to avoid having to reimplement
# some of these functions and keeping track for changes in new releases.

def mail_type_list_to_int(types):
    """Convert a str or list of mail types to a uint16_t."""
    cdef uint16_t flags = 0

    if not types or "None" == types:
        return slurm.NO_VAL16

    if isinstance(types, str):
        types = types.split(",")

    for typ in types:
        typ = typ.casefold()

        if "array_tasks" == typ:
            flags |= slurm.MAIL_ARRAY_TASKS
        elif "begin" == typ:
            flags |= slurm.MAIL_JOB_BEGIN
        elif "end" == typ:
            flags |= slurm.MAIL_JOB_END
        elif "fail" == typ:
            flags |= slurm.MAIL_JOB_FAIL
    #    elif "invalid_depend" == typ:
    #        flags |= slurm.MAIL_INVALID_DEPEND
        elif "requeue" == typ:
            flags |= slurm.MAIL_JOB_REQUEUE
        elif "stage_out" == typ:
            flags |= slurm.MAIL_JOB_STAGE_OUT
        elif "time_limit" == typ:
            flags |= slurm.MAIL_JOB_TIME100
        elif "time_limit_90" == typ:
            flags |= slurm.MAIL_JOB_TIME90
        elif "time_limit_80" == typ:
            flags |= slurm.MAIL_JOB_TIME80
        elif "time_limit_50" == typ:
            flags |= slurm.MAIL_JOB_TIME50
        elif "all" == typ:
            flags |= (slurm.MAIL_JOB_BEGIN
                  |   slurm.MAIL_JOB_END
                  |   slurm.MAIL_JOB_FAIL
                  |   slurm.MAIL_JOB_REQUEUE
                  |   slurm.MAIL_JOB_STAGE_OUT)
        else:
            raise ValueError("Invalid Mail type: {typ}.")

    return flags


def mail_type_int_to_list(uint16_t typ):
    """Convert uint16_t to a list of mail types."""
    types = []

    if typ == 0:
        return types

    if typ & slurm.MAIL_ARRAY_TASKS:
        types.append("ARRAY_TASKS")

#    if typ & slurm.MAIL_INVALID_DEPEND:
#        types.append("invalid_depend")

    if typ & slurm.MAIL_JOB_BEGIN:
        types.append("BEGIN")

    if typ & slurm.MAIL_JOB_END:
        types.append("END")

    if typ & slurm.MAIL_JOB_FAIL:
        types.append("FAIL")

    if typ & slurm.MAIL_JOB_REQUEUE:
        types.append("REQUEUE")

    if typ & slurm.MAIL_JOB_STAGE_OUT:
        types.append("STAGE_OUT")

    if typ & slurm.MAIL_JOB_TIME50:
        types.append("TIME_LIMIT_50")

    if typ & slurm.MAIL_JOB_TIME80:
        types.append("TIME_LIMIT_80")

    if typ & slurm.MAIL_JOB_TIME90:
        types.append("TIME_LIMIT_90")

    if typ & slurm.MAIL_JOB_TIME100:
        types.append("TIME_LIMIT_100")

    return types


def acctg_profile_list_to_int(types):
    """Convert a str or list of accounting gather profiles to uin32_t."""
    cdef uint32_t profile = 0

    if not types:
        return slurm.NO_VAL

    if isinstance(types, str):
        types = types.split(",")

    for typ in types:
        typ = typ.casefold()

        if "energy" == typ:
            profile |= slurm.ACCT_GATHER_PROFILE_ENERGY
        elif "task" == typ:
            profile |= slurm.ACCT_GATHER_PROFILE_TASK
        elif "lustre" == typ:
            profile |= slurm.ACCT_GATHER_PROFILE_LUSTRE
        elif "network" == typ:
            profile |= slurm.ACCT_GATHER_PROFILE_NETWORK
        elif "none" == typ:
            return slurm.ACCT_GATHER_PROFILE_NONE
        elif "all" == typ:
            return slurm.ACCT_GATHER_PROFILE_ALL
        else:
            raise ValueError("Invalid profile type: {typ}.")

    return profile


def acctg_profile_int_to_list(flags):
    """Convert uin32_t accounting gather profiles to a list of strings."""
    profiles = []

    if flags == 0 or flags == slurm.NO_VAL:
        return []

    if flags == slurm.ACCT_GATHER_PROFILE_ALL:
        return ["ALL"]
    elif flags == slurm.ACCT_GATHER_PROFILE_NONE:
        return []

    if flags & slurm.ACCT_GATHER_PROFILE_ENERGY:
        profiles.append("ENERGY")

    if flags & slurm.ACCT_GATHER_PROFILE_TASK:
        profiles.append("TASK")

    if flags & slurm.ACCT_GATHER_PROFILE_LUSTRE:
        profiles.append("LUSTRE")

    if flags & slurm.ACCT_GATHER_PROFILE_NETWORK:
        profiles.append("NETWORK")

    return profiles


def shared_type_str_to_int(typ):
    """Convert a job-sharing type str to its numerical representation."""
    if not typ:
        return slurm.NO_VAL16

    typ = typ.casefold()
    if typ == "oversubscribe" or typ == "yes":
        return slurm.JOB_SHARED_OK
    elif typ == "user":
        return slurm.JOB_SHARED_USER
    elif typ == "mcs":
        return slurm.JOB_SHARED_MCS
    elif typ == "no" or typ == "exclusive":
        return slurm.JOB_SHARED_NONE
    else:
        raise ValueError(f"Invalid resource_sharing type: {typ}.")


def cpu_gov_str_to_int(gov):
    """Convert a cpu governor str to is numerical representation."""
    if not gov:
        return u32(None)

    gov = gov.casefold()
    rc = 0

    if gov == "conservative":
        rc = slurm.CPU_FREQ_CONSERVATIVE
    elif gov == "ondemand":
        rc = slurm.CPU_FREQ_ONDEMAND
    elif gov == "performance":
        rc = slurm.CPU_FREQ_PERFORMANCE
    elif gov == "powersave":
        rc = slurm.CPU_FREQ_POWERSAVE
    elif gov == "userspace":
        rc = slurm.CPU_FREQ_USERSPACE
    elif gov == "schedutil":
        rc = slurm.CPU_FREQ_SCHEDUTIL
    else:
        raise ValueError("Invalid cpu gov type: {}".format(gov))

    return rc | slurm.CPU_FREQ_RANGE_FLAG


def cpu_freq_str_to_int(freq):
    """Convert a cpu-frequency str to its numerical representation."""
    if not freq:
        return slurm.NO_VAL

    if isinstance(freq, str) and not freq.isdigit():
        freq = freq.casefold()

        if freq == "low":
            return slurm.CPU_FREQ_LOW
        elif freq == "highm1":
            return slurm.CPU_FREQ_HIGHM1
        elif freq == "high":
            return slurm.CPU_FREQ_HIGH
        elif freq == "medium":
            return slurm.CPU_FREQ_MEDIUM
    else:
        fr = u32(int(freq))
        if fr != slurm.NO_VAL:
            return fr

    raise ValueError(f"Invalid cpu freq value: {freq}.")


def dependency_str_to_dict(dep):
    if not dep:
        return None

    out = {
        "after": [],
        "afterany": [],
        "afterburstbuffer": [],
        "aftercorr": [],
        "afternotok": [],
        "afterok": [],
        "singleton": False,
        "satisfy": "all",
    }

    delim = ","
    if "?" in dep:
        delim = "?"
        out["satisfy"] = "any"

    for item in dep.split(delim):
        if item == "singleton":
            out["singleton"] = True

        dep_and_job = item.split(":", 1)
        if len(dep_and_job) != 2:
            continue

        dep_name, jobs = dep_and_job[0], dep_and_job[1].split(":")
        if dep_name not in out:
            continue

        for job in jobs:
            out[dep_name].append(int(job) if job.isdigit() else job)

    return out
