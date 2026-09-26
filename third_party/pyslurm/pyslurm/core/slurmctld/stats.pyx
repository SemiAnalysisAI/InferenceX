#########################################################################
# slurmctld/stats.pyx - pyslurm slurmctld statistics api (sdiag)
#########################################################################
# Copyright (C) 2025 Toni Harzendorf <toni.harzendorf@gmail.com>

#########################################################################
# The implementation here is inspired by:
# - https://github.com/SchedMD/slurm/blob/c28fcf4f15981f891df7893099bceda21e2c5e6e/src/sdiag/sdiag.c
#
# So for completeness, the appropriate Copyright notices are also written
# below:
#
# Copyright (C) 2010-2011 Barcelona Supercomputing Center.
# Copyright (C) 2010-2022 SchedMD LLC.
#
# Please also check the Slurm DISCLAIMER at: pyslurm/slurm/SLURM_DISCLAIMER
#########################################################################
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

from pyslurm.core.error import verify_rpc, RPCError
from pyslurm.utils.ctime import _raw_time
from pyslurm.utils.helpers import (
    instance_to_dict,
    uid_to_name,
)
from pyslurm.utils import cstr
from pyslurm import xcollections


# Make sure this is in sync with the current Slurm release we are targeting.
# Check in Slurm source at src/slurmctld/slurmctld.h
BF_EXIT_COUNT = 6
SCHED_EXIT_COUNT = 6


cdef class ScheduleExitStatistics:

    def __init__(self):
        self.end_of_job_queue = 0
        self.default_queue_depth = 0
        self.max_job_start = 0
        self.blocked_on_licenses = 0
        self.max_rpc_count = 0
        self.max_time = 0

    @staticmethod
    cdef ScheduleExitStatistics from_ptr(stats_info_response_msg_t *ptr):
        if ptr.schedule_exit_cnt != SCHED_EXIT_COUNT:
            raise RPCError(msg="schedule_exit_cnt has an unexpected size. "
                           f"Got {ptr.schedule_exit_cnt}, expected {SCHED_EXIT_COUNT}.")

        out = ScheduleExitStatistics()
        out.end_of_job_queue = ptr.schedule_exit[0]
        out.default_queue_depth = ptr.schedule_exit[1]
        out.max_job_start = ptr.schedule_exit[2]
        out.blocked_on_licenses = ptr.schedule_exit[3]
        out.max_rpc_count = ptr.schedule_exit[4]
        out.max_time = ptr.schedule_exit[5]
        return out

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)


cdef class BackfillExitStatistics:

    def __init__(self):
        self.end_of_job_queue = 0
        self.max_job_start = 0
        self.max_job_test = 0
        self.max_time = 0
        self.node_space_size = 0
        self.state_changed = 0

    @staticmethod
    cdef BackfillExitStatistics from_ptr(stats_info_response_msg_t *ptr):
        if ptr.bf_exit_cnt != BF_EXIT_COUNT:
            raise RPCError(msg="bf_exit_cnt has an unexpected size. "
                           f"Got {ptr.bf_exit_cnt}, expected {BF_EXIT_COUNT}.")

        out = BackfillExitStatistics()
        out.end_of_job_queue = ptr.bf_exit[0]
        out.max_job_start = ptr.bf_exit[1]
        out.max_job_test = ptr.bf_exit[2]
        out.max_time = ptr.bf_exit[3]
        out.node_space_size = ptr.bf_exit[4]
        out.state_changed = ptr.bf_exit[5]
        return out

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)


cdef class RPCPending:

    def __init__(self):
        self.id = 0
        self.name = None
        self.count = 0

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)


cdef class RPCType:

    def __init__(self):
        self.id = 0
        self.name = None
        self.count = 0
        self.time = 0
        self.average_time = 0
        self.queued = 0
        self.dropped = 0
        self.cycle_last = 0
        self.cycle_max = 0

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)


cdef class RPCUser:

    def __init__(self):
        self.user_id = 0
        self.user_name = None
        self.count = 0
        self.time = 0
        self.average_time = 0

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)


cdef class RPCTypeStatistics(dict):

    def __init__(self):
        super().__init__()

    def to_dict(self, recursive = False):
        return xcollections.dict_recursive(self, recursive)

    @staticmethod
    cdef RPCTypeStatistics from_ptr(stats_info_response_msg_t *ptr,
                                   rpc_queue_enabled):
        out = RPCTypeStatistics()

        for i in range(ptr.rpc_type_size):
            stats = RPCType()
            stats.id = ptr.rpc_type_id[i]
            stats.name = rpc_num2string(ptr.rpc_type_id[i])
            stats.count = ptr.rpc_type_cnt[i]
            stats.time = ptr.rpc_type_time[i]

            if ptr.rpc_type_cnt[i]:
                stats.average_time = int(ptr.rpc_type_time[i] / ptr.rpc_type_cnt[i])

            if rpc_queue_enabled:
                stats.queued = ptr.rpc_type_queued[i]
                stats.dropped = ptr.rpc_type_dropped[i]
                stats.cycle_last = ptr.rpc_type_cycle_last[i]
                stats.cycle_max = ptr.rpc_type_cycle_max[i]

            out[stats.name] = stats

        return out

    @property
    def count(self):
        return xcollections.sum_property(self, RPCType.count)

    @property
    def time(self):
        return xcollections.sum_property(self, RPCType.time)

    @property
    def queued(self):
        return xcollections.sum_property(self, RPCType.queued)

    @property
    def dropped(self):
        return xcollections.sum_property(self, RPCType.dropped)


cdef class RPCUserStatistics(dict):

    def __init__(self):
        super().__init__()

    def to_dict(self, recursive = False):
        return xcollections.dict_recursive(self, recursive)

    @staticmethod
    cdef RPCUserStatistics from_ptr(stats_info_response_msg_t *ptr):
        out = RPCUserStatistics()

        for i in range(ptr.rpc_user_size):
            user_id = ptr.rpc_user_id[i]
            user = uid_to_name(user_id, err_on_invalid=False)
            stats = RPCUser()
            stats.user_id = ptr.rpc_user_id[i]
            stats.user_name = user
            stats.count = ptr.rpc_user_cnt[i]
            stats.time = ptr.rpc_user_time[i]

            if ptr.rpc_user_cnt[i]:
                stats.average_time = int(ptr.rpc_user_time[i] / ptr.rpc_user_cnt[i])

            key = user if user is not None else str(user_id)
            out[key] = stats

        return out

    @property
    def count(self):
        return xcollections.sum_property(self, RPCUser.count)

    @property
    def time(self):
        return xcollections.sum_property(self, RPCUser.time)


cdef class RPCPendingStatistics(dict):

    def __init__(self):
        super().__init__()

    def to_dict(self, recursive = False):
        return xcollections.dict_recursive(self, recursive)

    @staticmethod
    cdef RPCPendingStatistics from_ptr(stats_info_response_msg_t *ptr):
        out = RPCPendingStatistics()

        for i in range(ptr.rpc_queue_type_count):
            stats = RPCPending()
            stats.id = ptr.rpc_queue_type_id[i]
            stats.name = rpc_num2string(ptr.rpc_queue_type_id[i])
            stats.count = ptr.rpc_queue_count[i]
            out[stats.name] = stats

        return out

    @property
    def count(self):
        return xcollections.sum_property(self, RPCPendingStatistics.count)


cdef class Statistics:

    def __init__(self):
        self.schedule_cycle_mean = 0
        self.schedule_cycle_mean_depth = 0
        self.schedule_cycles_per_minute = 0
        self.backfill_cycle_mean = 0
        self.backfill_cycle_sum = 0
        self.backfill_mean_depth = 0
        self.backfill_mean_depth_try = 0
        self.backfill_queue_length_mean = 0
        self.backfill_table_size_mean = 0
        self.backfill_queue_length_sum = 0
        self.backfill_table_size_sum = 0

    @staticmethod
    def load():
        """Load the Statistics of the `slurmctld`.

        Returns:
            (pyslurm.slurmctld.Statistics): The Controller statistics.

        Raises:
            (pyslurm.RPCError): When fetching the Statistics failed.

        Examples:
            >>> from pyslurm import slurmctld
            >>> stats = slurmctld.Statistics.load()
            >>> print(stats.jobs_completed, stats.schedule_cycle_counter)
            10 20
        """
        cdef:
            stats_info_request_msg_t req
            stats_info_response_msg_t *resp = NULL
            Statistics out = None

        req.command_id = slurm.STAT_COMMAND_GET
        verify_rpc(slurm_get_statistics(&resp, &req))

        try:
            out = parse_response(resp)
        except Exception as e:
            raise e
        finally:
            slurm_free_stats_response_msg(resp)

        return out

    @staticmethod
    def reset():
        """Reset the Statistics of the `slurmctld`.

        Raises:
            (pyslurm.RPCError): When resetting the Statistics failed.

        Examples:
            >>> from pyslurm import slurmctld
            >>> slurmctld.Statistics.reset()
        """
        cdef stats_info_request_msg_t req
        req.command_id = slurm.STAT_COMMAND_RESET
        verify_rpc(slurm_reset_statistics(&req))

    def to_dict(self, recursive = False):
        """Convert the statistics to a dictionary.

        Returns:
            (dict): Statistics as a dict.

        Examples:
            >>> from pyslurm import slurmctld
            >>> stats = slurmctld.Statistics.load()
            >>> stats_dict = stats.to_dict()
        """
        return instance_to_dict(self, recursive)


def diag():
    """Load the Statistics of the `slurmctld`.

    This is a shortcut for [pyslurm.slurmctld.Statistics.load][]

    Returns:
        (pyslurm.slurmctld.Statistics): The Controller statistics.

    Raises:
        (pyslurm.RPCError): When fetching the Statistics failed.

    Examples:
        >>> from pyslurm import slurmctld
        >>> stats = slurmctld.Statistics.load()
        >>> print(stats.jobs_completed, stats.schedule_cycle_counter)
        10 20
    """
    return Statistics.load()


cdef parse_response(stats_info_response_msg_t *ptr):
    cdef Statistics out = Statistics()

    cycle_count = ptr.schedule_cycle_counter
    bf_cycle_count = ptr.bf_cycle_counter

    out.request_time = ptr.req_time
    out.data_since = ptr.req_time_start
    out.server_thread_count = ptr.server_thread_count
    out.rpc_queue_enabled = True if ptr.rpc_queue_enabled else False
    out.agent_queue_size = ptr.agent_queue_size
    out.agent_count = ptr.agent_count
    out.agent_thread_count = ptr.agent_thread_count
    out.dbd_agent_queue_size = ptr.dbd_agent_queue_size
    out.jobs_submitted = ptr.jobs_submitted
    out.jobs_started = ptr.jobs_started
    out.jobs_completed = ptr.jobs_completed
    out.jobs_canceled = ptr.jobs_canceled
    out.jobs_failed = ptr.jobs_failed
    out.jobs_pending = ptr.jobs_pending
    out.jobs_running = ptr.jobs_running
    out.schedule_cycle_last = int(ptr.schedule_cycle_last)
    out.schedule_cycle_max = int(ptr.schedule_cycle_max)
    out.schedule_cycle_counter = int(cycle_count)
    out.schedule_queue_length = int(ptr.schedule_queue_len)
    out.schedule_cycle_sum = int(ptr.schedule_cycle_sum)

    if cycle_count > 0:
        out.schedule_cycle_mean = int(ptr.schedule_cycle_sum / cycle_count)
        out.schedule_cycle_mean_depth = int(ptr.schedule_cycle_depth / cycle_count)

    ts = ptr.req_time - ptr.req_time_start
    if ts > 60:
        out.schedule_cycles_per_minute = int(cycle_count / (ts / 60))

    out.backfill_active = bool(ptr.bf_active)
    out.backfilled_jobs = ptr.bf_backfilled_jobs
    out.last_backfilled_jobs = ptr.bf_last_backfilled_jobs
    out.backfilled_het_jobs = ptr.bf_backfilled_het_jobs
    out.backfill_cycle_last_when = ptr.bf_when_last_cycle
    out.backfill_cycle_last = ptr.bf_cycle_last
    out.backfill_cycle_max = ptr.bf_cycle_max
    out.backfill_cycle_counter = bf_cycle_count
    out.backfill_cycle_sum = ptr.bf_cycle_sum
    out.backfill_last_depth = ptr.bf_last_depth
    out.backfill_last_depth_try = ptr.bf_last_depth_try
    out.backfill_queue_length = ptr.bf_queue_len
    out.backfill_queue_length_sum = ptr.bf_queue_len_sum
    out.backfill_table_size = ptr.bf_table_size
    out.backfill_table_size_sum = ptr.bf_table_size_sum
    out.backfill_depth_sum = ptr.bf_depth_sum
    out.backfill_depth_try_sum = ptr.bf_depth_try_sum

    if bf_cycle_count > 0:
        out.backfill_cycle_mean = int(ptr.bf_cycle_sum / bf_cycle_count)
        out.backfill_mean_depth = int(ptr.bf_depth_sum / bf_cycle_count)
        out.backfill_mean_depth_try = int(ptr.bf_depth_try_sum / bf_cycle_count)
        out.backfill_queue_length_mean = int(ptr.bf_queue_len_sum / bf_cycle_count)
        out.backfill_table_size_mean = int(ptr.bf_table_size_sum / bf_cycle_count)

    out.gettimeofday_latency = ptr.gettimeofday_latency

    out.rpcs_by_type = RPCTypeStatistics.from_ptr(ptr, out.rpc_queue_enabled)
    out.rpcs_by_user = RPCUserStatistics.from_ptr(ptr)
    out.rpcs_pending = RPCPendingStatistics.from_ptr(ptr)
    out.schedule_exit = ScheduleExitStatistics.from_ptr(ptr)
    out.backfill_exit = BackfillExitStatistics.from_ptr(ptr)

    return out


# Prepare some test data
def _parse_test_data():
    import datetime

    cdef stats_info_response_msg_t stats
    memset(&stats, 0, sizeof(stats))

    stats.req_time = int(datetime.datetime.now().timestamp())
    stats.req_time_start = int(datetime.datetime.now().timestamp()) - 200
    stats.jobs_submitted = 20
    stats.jobs_running = 3
    stats.schedule_cycle_counter = 10
    stats.schedule_cycle_last = 40
    stats.schedule_cycle_sum = 45

    stats.bf_cycle_counter = 100
    stats.bf_active = 0
    stats.bf_backfilled_jobs = 10
    stats.bf_cycle_sum = 200
    stats.bf_depth_try_sum = 300
    stats.bf_queue_len_sum = 600
    stats.bf_table_size_sum = 200

    stats.rpc_type_size = 3
    stats.rpc_type_id = <uint16_t*>xmalloc(sizeof(uint16_t) * stats.rpc_type_size)
    stats.rpc_type_cnt = <uint32_t*>xmalloc(sizeof(uint32_t) * stats.rpc_type_size)
    stats.rpc_type_time = <uint64_t*>xmalloc(sizeof(uint64_t) * stats.rpc_type_size)

    for i in range(stats.rpc_type_size):
        stats.rpc_type_id[i] = 2000+i
        stats.rpc_type_cnt[i] = i+1
        stats.rpc_type_time[i] = i+2

    stats.rpc_user_size = 1
    stats.rpc_user_id = <uint32_t*>xmalloc(sizeof(uint32_t) * stats.rpc_user_size)
    stats.rpc_user_cnt = <uint32_t*>xmalloc(sizeof(uint32_t) * stats.rpc_user_size)
    stats.rpc_user_time = <uint64_t*>xmalloc(sizeof(uint64_t) * stats.rpc_user_size)

    for i in range(stats.rpc_user_size):
        stats.rpc_user_id[i] = i
        stats.rpc_user_cnt[i] = i+1
        stats.rpc_user_time[i] = i+2

    stats.bf_exit_cnt = BF_EXIT_COUNT
    stats.bf_exit = <uint32_t*>xmalloc(sizeof(uint32_t) * BF_EXIT_COUNT)
    for i in range(stats.bf_exit_cnt):
        stats.bf_exit[i] = i+1

    stats.schedule_exit_cnt = SCHED_EXIT_COUNT
    stats.schedule_exit = <uint32_t*>xmalloc(sizeof(uint32_t) * SCHED_EXIT_COUNT)

    for i in range(stats.schedule_exit_cnt):
        stats.schedule_exit[i] = i+1

    stats.rpc_queue_type_count = 5
    stats.rpc_queue_count = <uint32_t*>xmalloc(sizeof(uint32_t) * stats.rpc_queue_type_count)
    stats.rpc_queue_type_id = <uint32_t*>xmalloc(sizeof(uint32_t) * stats.rpc_queue_type_count)

    for i in range(stats.rpc_queue_type_count):
        stats.rpc_queue_count[i] = i+1
        stats.rpc_queue_type_id[i] = 2000+i

    return parse_response(&stats)
