#########################################################################
# slurmctld/stats.pxd - pyslurm slurmctld statistics api (sdiag)
#########################################################################
# Copyright (C) 2025 Toni Harzendorf <toni.harzendorf@gmail.com>
#
#########################################################################
# Much of the documentation here (with some modifications) has been taken from:
# - https://slurm.schedmd.com/sdiag.html
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

from libc.string cimport memset
from pyslurm cimport slurm
from pyslurm.slurm cimport (
    stats_info_response_msg_t,
    stats_info_request_msg_t,
    slurm_get_statistics,
    slurm_reset_statistics,
    slurm_free_stats_response_msg,
    xfree,
    xmalloc,
)
from pyslurm.utils cimport cstr
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from pyslurm.utils.uint cimport (
    u16_parse,
    u32_parse,
    u64_parse,
    u16_parse_bool,
)

cdef extern const char *rpc_num2string(uint16_t msg_type)

cdef parse_response(stats_info_response_msg_t *ptr)


cdef class ScheduleExitStatistics:
    """Conditions reached at the end of a scheduling run

    Each attribute is simply a counter that describes how many times a specific
    condition was met during the main scheduling run.

    Attributes:
        end_of_job_queue (int):
            Times the end of the job queue was reached.
        default_queue_depth (int):
            Reached the number of jobs allowed to be tested limit
        max_job_start (int):
            Reached the number of jobs allowed to start limit
        blocked_on_licenses (int):
            Times the scheduler blocked on licenses.
        max_rpc_count (int):
            Reached RPC Limit.
        max_time (int):
            Reached maximum allowed scheduler time for a cycle.
    """
    cdef public:
        end_of_job_queue
        default_queue_depth
        max_job_start
        blocked_on_licenses
        max_rpc_count
        max_time

    @staticmethod
    cdef ScheduleExitStatistics from_ptr(stats_info_response_msg_t *ptr)


cdef class BackfillExitStatistics:
    """Conditions reached at the end of a Backfill scheduling run.

    Each attribute is simply a counter that describes how many times a specific
    condition was met during the Backfill scheduling run.

    Attributes:
        end_of_job_queue (int):
            Times the end of the job queue was reached.
        max_job_start (int):
            Reached the number of jobs allowed to start limit.
        max_job_test (int):
            Reached the number of jobs allowed to attempt backfill scheduling
            for.
        max_time (int):
            Reached maximum allowed scheduler time for a cycle.
        node_space_size (int):
            Reached the node_space table size limit.
        state_changed (int):
            System state changes.
    """
    cdef public:
        end_of_job_queue
        max_job_start
        max_job_test
        max_time
        node_space_size
        state_changed

    @staticmethod
    cdef BackfillExitStatistics from_ptr(stats_info_response_msg_t *ptr)


cdef class RPCPending:
    """Statistics for a pending RPC.

    Attributes:
        id (int):
            The numeric ID of the RPC type.
        name (str):
            The string representation of the RPC.
        count (int):
            How many RPCs are pending of this type.
    """
    cdef public:
        id
        name
        count


cdef class RPCType:
    """Statistics for a specific RPC Type.

    Attributes:
        id (int):
            The numeric ID of the RPC Type
        name (str):
            The string representation of the RPC
        count (int):
            How many times this RPC was issued since the last time the
            statistics were cleared.
        time (int):
            How much total time it has taken to process this RPC. The unit is
            microseconds
        average_time (int):
            How much time on average it has taken to process this RPC. The unit
            is microseconds.
        queued (int):
            How many of these RPCs are still queued.
        dropped (int):
            How many of these RPCs have been dropped.
        cycle_last (int):
            Count of RPCs processed within the last RPC queue cycle.
        cycle_max (int):
            Maximum count of RPCs that have been processed within a RPC queue
            cycle.
    """
    cdef public:
        id
        name
        count
        time
        average_time
        queued
        dropped
        cycle_last
        cycle_max


cdef class RPCUser:
    """RPC Statistics for a specific User.

    Attributes:
        user_id (int):
            The numeric ID of the User.
        user_name (str):
            The name of the User.
        count (int):
            How many times the User issued RPCs since the last time the
            statistics were cleared.
        time (int):
            How much total time it has taken to process RPCs by this User. The
            unit is microseconds
        average_time (int):
            How much time on average it has taken to process RPCs by this User.
            The unit is microseconds.
    """
    cdef public:
        user_id
        user_name
        count
        time
        average_time


cdef class RPCTypeStatistics(dict):
    """Collection of [pyslurm.slurmctld.RPCType][] objects.

    Attributes:
        count (int):
            Total amount of RPCs made to the `slurmctld` since last reset.
        time (int):
            Total amount of time it has taken to process all RPCs made yet.
        queued (int):
            Total amount of RPCs queued.
        dropped (int):
            Total amount of RPCs dropped.
    """
    @staticmethod
    cdef RPCTypeStatistics from_ptr(stats_info_response_msg_t *ptr, rpc_queue_enabled)


cdef class RPCUserStatistics(dict):
    """Collection of [pyslurm.slurmctld.RPCUser][] objects.

    Attributes:
        count (int):
            Total amount of RPCs made to the `slurmctld` since last reset.
        time (int):
            Total amount of time it has taken to process all RPCs made yet.
    """
    @staticmethod
    cdef RPCUserStatistics from_ptr(stats_info_response_msg_t *ptr)


cdef class RPCPendingStatistics(dict):
    """Collection of [pyslurm.slurmctld.RPCPending][] objects.

    Attributes:
        count (int):
            Total amount of RPCs currently pending.
    """
    @staticmethod
    cdef RPCPendingStatistics from_ptr(stats_info_response_msg_t *ptr)


cdef class Statistics:
    """Statistics for the `slurmctld`.

    For more information, also check out the Slurm [sdiag documentation](https://slurm.schedmd.com/sdiag.html).

    Attributes:
        request_time (int):
            Time when the data was requested. This is a unix timestamp.
        data_since (int):
            The date when `slurmctld` started gathering statistics. This is a
            unix timestamp.
        server_thread_count (int):
            The number of current active `slurmctld` threads.
        rpc_queue_enabled (bool):
            Whether RPC queuing is enabled.
        agent_queue_size (int):
            Count of enqueued outgoing RPC requests in an internal retry list.
        agent_count (int):
            Number of agent threads.
        agent_thread_count (int):
            Total count of active threads created by all the agent threads.
        dbd_agent_queue_size (int):
            Number of messages intended for the `slurmdbd`. If the `slurmdbd`
            goes down, then this number starts going up.
        jobs_submitted (int):
            Number of jobs submitted since last reset
        jobs_started (int):
            Number of jobs started since last reset. This includes backfilled
            jobs.
        jobs_completed (int):
            Number of jobs completed since last reset.
        jobs_canceled (int):
            Number of jobs canceled since last reset.
        jobs_failed (int):
            Number of jobs failed due to `slurmd` or other internal issues since
            last reset.
        jobs_pending (int):
            Number of jobs pending.
        jobs_running (int):
            Number of jobs running.
        schedule_cycle_last (int):
            Time in microseconds for last scheduling cycle.
        schedule_cycle_max (int):
            Maximum time in microseconds for any scheduling cycle since last
            reset.
        schedule_cycle_counter (int):
            Total amount of scheduling cycles ran since last reset.
        schedule_cycle_mean (int):
            Mean time in microseconds for all scheduling cycles since last
            reset.
        schedule_cycle_mean_depth (int):
            Mean of cycle depth. Depth means number of jobs processed in a
            scheduling cycle.
        schedule_cycle_sum (int):
            Total run time in microseconds for all scheduling cycles since last
            reset.
        schedule_cycles_per_minute (int):
            Counter of scheduling executions per minute.
        schedule_queue_length (int):
            Length of jobs pending queue.
        backfill_active (bool):
            Whether these statistics have been gathered during backfilling
            operation.
        backfilled_jobs (int):
            Number of jobs started thanks to backfilling since last slurm
            start.
        last_backfilled_jobs (int):
            Number of jobs started thanks to backfilling since last time stats
            where reset. (which is midnight UTC time in this case)
        backfilled_het_jobs (int):
            Number of heterogeneous job components started thanks to
            backfilling since last Slurm start.
        backfill_cycle_counter (int):
            Number of backfill scheduling cycles since last reset.
        backfill_cycle_last_when (int):
            Time when last backfill scheduling cycle happened. This is a unix
            timestamp.
        backfill_cycle_last (int):
            Time in microseconds of last backfill scheduling cycle.
        backfill_cycle_max (int):
            Time in microseconds of maximum backfill scheduling cycle execution
            since last reset.
        backfill_cycle_mean (int):
            Mean time in microseconds of backfilling scheduling cycles since
            last reset.
        backfill_cycle_sum (int):
            Total time in microseconds of backfilling scheduling cycles since
            last reset.
        backfill_last_depth (int):
            Number of processed jobs during last backfilling scheduling cycle.
            It counts every job even if that job can not be started due to
            dependencies or limits.
        backfill_depth_sum (int):
            Total number of jobs processed during all backfilling scheduling
            cycles since last reset.
        backfill_last_depth_try (int):
            Number of processed jobs during last backfilling scheduling cycle.
            It counts only jobs with a chance to start using available
            resources.
        backfill_depth_try_sum (int):
            Subset of `backfill_depth_sum` that the backfill scheduler
            attempted to schedule.
        backfill_mean_depth (int):
            Mean count of jobs processed during all backfilling scheduling
            cycles since last reset. Jobs which are found to be ineligible to
            run when examined by the backfill scheduler are not counted.
        backfill_mean_depth_try (int):
            The subset of `backfill_mean_depth` that the backfill
            scheduler attempted to schedule.
        backfill_queue_length (int):
            Number of jobs pending to be processed by backfilling algorithm. A
            job is counted once for each partition it is queued to use.
        backfill_queue_length_sum (int):
            Total number of jobs pending to be processed by backfilling
            algorithm since last reset.
        backfill_queue_length_mean (int):
            Mean count of jobs pending to be processed by backfilling
            algorithm.
        backfill_table_size (int):
            Count of different time slots tested by the backfill scheduler in
            its last iteration.
        backfill_table_size_sum (int):
            Total number of different time slots tested by the backfill
            scheduler.
        backfill_table_size_mean (int):
            Mean count of different time slots tested by the backfill
            scheduler. Larger counts increase the time required for the
            backfill operation.
        gettimeofday_latency (int):
            Latency of 1000 calls to the gettimeofday() syscall in
            microseconds, as measured at controller startup.
        rpcs_by_type (pyslurm.slurmctld.RPCTypeStatistics):
            RPC Statistics organized by Type.
        rpcs_by_user (pyslurm.slurmctld.RPCUserStatistics):
            RPC Statistics organized by User.
        rpcs_pending (pyslurm.slurmctld.RPCPendingStatistics):
            Statistics for pending RPCs.
    """
    cdef public:
        request_time
        data_since
        server_thread_count
        rpc_queue_enabled
        agent_queue_size
        agent_count
        agent_thread_count
        dbd_agent_queue_size

        jobs_submitted
        jobs_started
        jobs_completed
        jobs_canceled
        jobs_failed
        jobs_pending
        jobs_running

        schedule_cycle_last
        schedule_cycle_max
        schedule_cycle_counter
        schedule_cycle_mean
        schedule_cycle_mean_depth
        schedule_cycle_sum
        schedule_cycles_per_minute
        schedule_queue_length
        schedule_exit

        backfill_active
        backfilled_jobs
        last_backfilled_jobs
        backfilled_het_jobs
        backfill_cycle_counter
        backfill_cycle_last_when
        backfill_cycle_last
        backfill_cycle_max
        backfill_cycle_mean
        backfill_cycle_sum
        backfill_last_depth
        backfill_depth_sum
        backfill_last_depth_try
        backfill_depth_try_sum
        backfill_mean_depth
        backfill_mean_depth_try
        backfill_queue_length
        backfill_queue_length_sum
        backfill_queue_length_mean
        backfill_table_size
        backfill_table_size_sum
        backfill_table_size_mean
        backfill_exit

        gettimeofday_latency

        rpcs_by_type
        rpcs_by_user
        rpcs_pending
