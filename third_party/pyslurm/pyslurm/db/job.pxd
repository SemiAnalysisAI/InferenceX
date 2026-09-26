#########################################################################
# job.pxd - pyslurm slurmdbd job api
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
    slurmdb_job_rec_t,
    slurmdb_job_cond_t,
    slurmdb_step_rec_t,
    slurmdb_jobs_get,
    slurmdb_destroy_job_cond,
    slurmdb_destroy_job_rec,
    slurmdb_destroy_step_rec,
    slurm_destroy_selected_step,
    slurm_selected_step_t,
    slurm_list_create,
    slurm_list_append,
    try_xmalloc,
    slurmdb_job_cond_def_start_end,
    slurm_job_state_string,
    slurm_job_state_reason_string,
    slurmdb_create_job_rec,
    slurmdb_job_modify,
    xfree,
)
from pyslurm.db.util cimport (
    SlurmList,
    SlurmListItem,
    make_char_list,
)
from pyslurm.db.step cimport JobStep, JobSteps
from pyslurm.db.stats cimport JobStatistics
from pyslurm.db.connection cimport Connection
from pyslurm.utils cimport cstr
from pyslurm.db.qos cimport QualitiesOfService
from pyslurm.db.tres cimport (
    TrackableResources,
    TrackableResource,
    GPU,
)
from pyslurm.xcollections cimport MultiClusterMap
from pyslurm.utils.uint cimport u32_parse_bool_flag
from libc.stdint cimport uint32_t


cdef class JobFilter:
    """Query-Conditions for Jobs in the Slurm Database.

    Args:
        **kwargs (Any, optional=None):
            Any valid attribute of the object.

    Attributes:
        ids (list[int]):
            A list of Job ids to search for.
        start_time (Union[str, int, datetime.datetime]):
            Search for Jobs which started after this time.
        end_time (Union[str, int, datetime.datetime]):
            Search for Jobs which ended before this time.
        accounts (list[str]):
            Search for Jobs with these account names.
        association_ids (list[int]):
            Search for Jobs with these association ids.
        clusters (list[str]):
            Search for Jobs running in these clusters.
        constraints (list[str]):
            Search for Jobs with these constraints.
        cpus (int):
            Search for Jobs with exactly this many CPUs.
            Note: If you also specify `max_cpus`, then this value will act as
            the minimum.
        max_cpus (int):
            Search for Jobs with no more than this amount of CPUs.
            Note: This value has no effect without also setting `cpus`.
        nodes (int):
            Search for Jobs with exactly this many nodes.
            Note: If you also specify `max_nodes`, then this value will act as
            the minimum.
        max_nodes (int):
            Search for Jobs with no more than this amount of nodes.
            Note: This value has no effect without also setting `nodes`.
        qos (list[str]):
            Search for Jobs with these Qualities of Service.
        names (list[str]):
            Search for Jobs with these job names.
        partitions (list[str]):
            Search for Jobs with these partition names.
        groups (list[str]):
            Search for Jobs with these group names. Alternatively, you can
            also specify the GIDs directly.
        timelimit (Union[str, int]):
            Search for Jobs with exactly this timelimit.
            Note: If you also specify `max_timelimit`, then this value will act
            as the minimum.
        max_timelimit (Union[str, int]):
            Search for Jobs which run no longer than this timelimit
            Note: This value has no effect without also setting `timelimit`
        users (list[str]):
            Search for Jobs with these user names. Alternatively, you can also
            specify the UIDs directly.
        wckeys (list[str]):
            Search for Jobs with these WCKeys
        nodelist (list[str]):
            Search for Jobs that ran on any of these Nodes
        with_script (bool):
            Instruct the slurmdbd to also send the job script(s)
            Note: This requires specifying explictiy job ids, and is mutually
            exclusive with `with_env`
        with_env (bool):
            Instruct the slurmdbd to also send the job environment(s)
            Note: This requires specifying explictiy job ids, and is mutually
            exclusive with `with_script`
        truncate_time (bool):
            Truncate start and end time.
            For example, when a Job has actually started before the requested
            `start_time`, the time will be truncated to `start_time`. Same
            logic applies for `end_time`. This is like the `-T` / `--truncate`
            option from `sacct`.
    """
    cdef slurmdb_job_cond_t *ptr

    cdef public:
        ids
        start_time
        end_time
        accounts
        association_ids
        clusters
        constraints
        cpus
        max_cpus
        nodes
        max_nodes
        qos
        names
        partitions
        groups
        timelimit
        max_timelimit
        users
        wckeys
        nodelist
        with_script
        with_env
        truncate_time


cdef class Jobs(MultiClusterMap):
    """A [`Multi Cluster`][pyslurm.xcollections.MultiClusterMap] collection of [pyslurm.db.Job][] objects.

    Args:
        jobs (Union[list[int], dict[int, pyslurm.db.Job], str], optional=None):
            Jobs to initialize this collection with.

    Attributes:
        stats (pyslurm.db.JobStatistics):
            Utilization statistics of this Job Collection
        cpus (int):
            Total amount of cpus requested.
        nodes (int):
            Total amount of nodes requested.
        memory (int):
            Total amount of requested memory in Mebibytes.
    """
    cdef public:
        stats
        cpus
        nodes
        memory


cdef class Job:
    """A Slurm Database Job.

    Args:
        job_id (int, optional=0):
            An Integer representing a Job-ID.
        cluster (str, optional=None):
            Name of the Cluster for this Job. Default is the name of the local
            Cluster.

    Other Parameters:
        admin_comment (str):
            Admin comment for the Job.
        comment (str):
            Comment for the Job
        wckey (str):
            Name of the WCKey for this Job
        derived_exit_code (int):
            Highest exit code of all the Job steps
        extra (str):
            Arbitrary string that can be stored with a Job.

    Attributes:
        steps (pyslurm.db.JobSteps):
            Steps this Job has
        stats (pyslurm.db.JobStatistics):
            Utilization statistics of this Job
        account (str):
            Account of the Job.
        admin_comment (str):
            Admin comment for the Job.
        num_nodes (int):
            Amount of nodes this Job has allocated (if it is running) or
            requested (if it is still pending).
        array_id (int):
            The master Array-Job ID.
        array_tasks_parallel (int):
            Max number of array tasks allowed to run simultaneously.
        array_task_id (int):
            Array Task ID of this Job if it is an Array-Job.
        array_tasks_waiting (str):
            Array Tasks that are still waiting.
        association_id (int):
            ID of the Association this job runs in.
        block_id (str):
            Name of the block used (for BlueGene Systems)
        cluster (str):
            Cluster this Job belongs to
        constraints (str):
            Constraints of the Job
        container (str):
            Path to OCI Container bundle
        db_index (int):
            Unique database index of the Job in the job table
        derived_exit_code (int):
            Highest exit code of all the Job steps
        derived_exit_code_signal (int):
            Signal of the derived exit code
        comment (str):
            Comment for the Job
        elapsed_time (int):
            Amount of seconds elapsed for the Job
        eligible_time (int):
            When the Job became eligible to run, as a unix timestamp
        end_time (int):
            When the Job ended, as a unix timestamp
        extra (str):
            Arbitrary string that can be stored with a Job.
        exit_code (int):
            Exit code of the job script or salloc.
        exit_code_signal (int):
            Signal of the exit code for this Job.
        failed_node (str):
            Name of the failed node that caused the job to get killed.
        group_id (int):
            ID of the group for this Job
        group_name (str):
            Name of the group for this Job
        id (int):
            ID of the Job
        name (str):
            Name of the Job
        mcs_label (str):
            MCS Label of the Job
        nodelist (str):
            Nodes this Job is using
        partition (str):
            Name of the Partition for this Job
        priority (int):
            Priority for the Job
        qos (str):
            Name of the Quality of Service for the Job
        cpus (int):
            Amount of CPUs the Job has/had allocated, or, if the Job is still
            pending, this will reflect the amount requested.
        memory (int):
            Amount of memory the Job requested in total, in Mebibytes
        reservation (str):
            Name of the Reservation for this Job
        script (str):
            The batch script for this Job.
            Note: Only available if the "with_script" condition was given
        start_time (int):
            Time when the Job started, as a unix timestamp
        state (str):
            State of the Job
        state_reason (str):
            Last reason a Job was blocked from running
        cancelled_by (str):
            Name of the User who cancelled this Job
        submit_time (int):
            Time the Job was submitted, as a unix timestamp
        submit_command (str):
            Full command issued to submit the Job
        suspended_time (int):
            Amount of seconds the Job was suspended
        system_comment (str):
            Arbitrary System comment for the Job
        time_limit (int):
            Time limit of the Job in minutes
        user_id (int):
            UID of the User this Job belongs to
        user_name (str):
            Name of the User this Job belongs to
        wckey (str):
            Name of the WCKey for this Job
        working_directory (str):
            Working directory of the Job
        heterogeneous_id (int):
            Heterogeneous job id.
        heterogeneous_offset (int):
            Heterogeneous job offset.
        requeue_count (int):
            Amount of times the Job has been requeued.
        requested_reservations (list[str]):
            The list of Reservations this Job requests.
        reservation_id (int):
            ID of the Reservation in use.
        wckey_id (int):
            ID of the WCKey used.
        lineage (str):
            Association Lineage of the Job.
        licenses (list[str]):
            Licenses for the Job.
        standard_input (str):
            The path to the file for the standard input stream.
        standard_output (str):
            The path to the log file for the standard output stream.
        standard_error (str):
            The path to the log file for the standard error stream.
        segment_size (int):
            When a block topology is used, this defines the size of the
            segments that have been used to create the job allocation.
        scheduler (pyslurm.SchedulerType):
            The scheduler which started this Job.
        start_rpc_received (bool):
            Whether the Job received the Start RPC.
        gpus (dict[GPU]):
            A mapping of GPUs the Job has requested or allocated.
        gres (dict):
            The Generic Resources the Job has either requested or allocated.
        tres (pyslurm.db.TrackableResources):
            The TRES the Job has either requested or allocated.
        allocated_tres (pyslurm.db.TrackableResources):
            TRES the Job has allocated when already running.
            Will return `None` if it is still pending.
        requested_tres (pyslurm.db.TrackableResources):
            TRES the Job has requested.
    """
    cdef:
        slurmdb_job_rec_t *ptr
        QualitiesOfService qos_data
        TrackableResources tres_data

    cdef public:
        JobSteps steps
        JobStatistics stats

    @staticmethod
    cdef Job from_ptr(slurmdb_job_rec_t *in_ptr)

    cdef _get_stdio(self, char *path)
