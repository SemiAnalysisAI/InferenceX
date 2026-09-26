#########################################################################
# job.pyx - interface to retrieve slurm job information
#########################################################################
# Copyright (C) 2023 Toni Harzendorf <toni.harzendorf@gmail.com>
#
# This file is part of PySlurm
#
# PySlurm is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
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

from pyslurm.utils cimport cstr, ctime
from pyslurm.utils.uint cimport *
from pyslurm.utils.ctime cimport time_t
from pyslurm.db.stats cimport JobStatistics
from pyslurm.db.tres cimport TrackableResources, GPU
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from libc.stdlib cimport free
from pyslurm.core.job.submission cimport JobSubmitDescription
from pyslurm.core.job.step cimport JobSteps, JobStep
from pyslurm.xcollections cimport MultiClusterMap
from pyslurm cimport slurm
from pyslurm.slurm cimport (
    working_cluster_rec,
    slurm_msg_t,
    job_id_msg_t,
    slurm_msg_t_init,
    return_code_msg_t,
    slurm_send_recv_controller_msg,
    slurm_free_return_code_msg,
    slurm_free_job_info_msg,
    slurm_free_job_info,
    slurm_load_job,
    slurm_load_jobs,
    job_info_msg_t,
    slurm_job_info_t,
    slurm_job_state_string,
    slurm_job_state_reason_string,
    slurm_job_share_string,
    slurm_job_batch_script,
    slurm_get_job_stdin,
    slurm_get_job_stdout,
    slurm_get_job_stderr,
    slurm_signal_job,
    slurm_kill_job,
    slurm_resume,
    slurm_suspend,
    slurm_update_job,
    slurm_notify_job,
    slurm_requeue,
    xfree,
    try_xmalloc,
)


cdef class Jobs(MultiClusterMap):
    """A [`Multi Cluster`][pyslurm.xcollections.MultiClusterMap] collection of [pyslurm.Job][] objects.

    Args:
        jobs (Union[list[int], dict[int, pyslurm.Job], str], optional=None):
            Jobs to initialize this collection with.
        frozen (bool, optional=False):
            Control whether this collection is `frozen` when reloading Job
            information.

    Attributes:
        memory (int):
            Total amount of memory requested for all Jobs in this collection,
            in Mebibytes
        cpus (int):
            Total amount of cpus requested for all Jobs in this collection.
        ntasks (int):
            Total amount of tasks requested for all Jobs in this collection.
        elapsed_cpu_time (int):
            Total amount of CPU-Time used by all the Jobs in the collection.
            This is the result of multiplying the run_time with the amount of
            cpus requested for each job.
        frozen (bool):
            If this is set to True and the `reload()` method is called, then
            *ONLY* Jobs that already exist in this collection will be
            reloaded. New Jobs that are discovered will not be added to this
            collection, but old Jobs which have already been purged from the
            Slurm controllers memory will not be removed either.
            The default is False, so old jobs will be removed, and new Jobs
            will be added - basically the same behaviour as doing Jobs.load().
        stats (pyslurm.db.JobStatistics):
            Real-time statistics of all Jobs in this collection.
            Before you can access the stats data for this, you have to call
            the `load_stats` method on this collection.
    """
    cdef:
        job_info_msg_t *info
        slurm_job_info_t tmp_info

    cdef public:
        frozen
        JobStatistics stats


cdef class Job:
    """A Slurm Job.

    All attributes in this class are read-only.

    Args:
        job_id (int):
            An Integer representing a Job-ID.

    Attributes:
        steps (pyslurm.JobSteps):
            Steps this Job has.
            Before you can access the Steps data for a Job, you have to call
            the `reload()` method of a Job instance or the `load_steps()`
            method of a Jobs collection.
        stats (pyslurm.db.JobStatistics):
            Real-time statistics of a Job.
            Before you can access the stats data for a Job, you have to call
            the `load_stats` method of a Job instance or the Jobs collection.
        pids (dict[str, list]):
            Current Process-IDs of the Job, organized by node name.
            Before you can access the pids data for a Job, you have to call
            the `load_stats` method of a Job instance or the Jobs collection.
        name (str):
            Name of the Job
        id (int):
            Unique ID of the Job.
        association_id (int):
            ID of the Association this Job runs with.
        account (str):
            Name of the Account this Job is runs with.
        user_id (int):
            UID of the User who submitted the Job.
        user_name (str):
            Name of the User who submitted the Job.
        group_id (int):
            GID of the Group that Job runs under.
        group_name (str):
            Name of the Group this Job runs under.
        priority (int):
            Priority of the Job.
        nice (int):
            Nice Value of the Job.
        qos (str):
            QOS Name of the Job.
        min_cpus_per_node (int):
            Minimum Amount of CPUs per Node the Job requested.
        state (str):
            State this Job is currently in.
        state_reason (str):
            A Reason explaining why the Job is in its current state.
        is_requeueable (bool):
            Whether the Job is requeuable or not.
        requeue_count (int):
            Amount of times the Job has been requeued.
        is_batch_job (bool):
            Whether the Job is a batch job or not.
        node_reboot_required (bool):
            Whether the Job requires the Nodes to be rebooted first.
        dependencies (dict):
            Dependencies the Job has to other Jobs.
        time_limit (int):
            Time-Limit, in minutes, for this Job.
        time_limit_min (int):
            Minimum Time-Limit in minutes for this Job.
        submit_time (int):
            Time the Job was submitted, as unix timestamp.
        eligible_time (int):
            Time the Job is eligible to start, as unix timestamp.
        accrue_time (int):
            Job accrue time, as unix timestamp
        start_time (int):
            Time this Job has started execution, as unix timestamp.
        resize_time (int):
            Time the job was resized, as unix timestamp.
        deadline (int):
            Time when a pending Job will be cancelled, as unix timestamp.
        preempt_eligible_time (int):
            Time the Job is eligible for preemption, as unix timestamp.
        preempt_time (int):
            Time the Job was signaled for preemption, as unix timestamp.
        suspend_time (int):
            Last Time the Job was suspended, as unix timestamp.
        last_sched_evaluation_time (int):
            Last time evaluated for Scheduling, as unix timestamp.
        pre_suspension_time (int):
            Amount of seconds the Job ran prior to suspension, as unix
            timestamp
        mcs_label (str):
            MCS Label for the Job
        partition (str):
            Name of the Partition the Job runs in.
        submit_host (str):
            Name of the Host this Job was submitted from.
        submit_session_id (int):
            ID of the session when the Job was submitted.
        batch_host (str):
            Name of the Host where the Batch-Script is executed.
        num_nodes (int):
            Amount of Nodes the Job has requested or allocated.
        max_nodes (int):
            Maximum amount of Nodes the Job has requested.
        allocated_nodes (str):
            Nodes the Job is currently using.
            This is only valid when the Job is running. If the Job is pending,
            it will always return None.
        required_nodes (str):
            Nodes the Job is explicitly requiring to run on.
        excluded_nodes (str):
            Nodes that are explicitly excluded for execution.
        scheduled_nodes (str):
            Nodes the Job is scheduled on by the slurm controller.
        derived_exit_code (int):
            The derived exit code for the Job.
        derived_exit_code_signal (int):
            Signal for the derived exit code.
        exit_code (int):
            Code with which the Job has exited.
        exit_code_signal (int):
            The signal which has led to the exit code of the Job.
        batch_constraints (list):
            Features that node(s) should have for the batch script.
            Controls where it is possible to execute the batch-script of the
            job. Also see 'constraints'
        federation_origin (str):
            Federation Origin
        federation_siblings_active (list):
            Federation siblings active
        federation_siblings_viable (list):
            Federation siblings viable
        cpus (int):
            Total amount of CPUs the Job is using.
            If the Job is still pending, this will be the amount of requested
            CPUs.
        cpus_per_task (int):
            Number of CPUs per Task used.
        cpus_per_gpu (int):
            Number of CPUs per GPU used.
        boards_per_node (int):
            Number of boards per Node.
        sockets_per_board (int):
            Number of sockets per board.
        sockets_per_node (int):
            Number of sockets per node.
        cores_per_socket (int):
            Number of cores per socket.
        threads_per_core (int):
            Number of threads per core.
        ntasks (int):
            Number of parallel processes.
        ntasks_per_node (int):
            Number of parallel processes per node.
        ntasks_per_board (int):
            Number of parallel processes per board.
        ntasks_per_socket (int):
            Number of parallel processes per socket.
        ntasks_per_core (int):
            Number of parallel processes per core.
        ntasks_per_gpu (int):
            Number of parallel processes per GPU.
        delay_boot_time (int):
            https://slurm.schedmd.com/sbatch.html#OPT_delay-boot, in minutes
        constraints (list):
            A list of features the Job requires nodes to have.
            In contrast, the 'batch_constraints' option only focuses on the
            initial batch-script placement. This option however means features
            to restrict the list of nodes a job is able to execute on in
            general beyond the initial batch-script.
        cluster (str):
            Name of the cluster the job is executing on.
        cluster_constraints (list):
            A List of features that a cluster should have.
        reservation (str):
            Name of the reservation this Job uses.
        resource_sharing (str):
            Mode controlling how a job shares resources with others.
        requires_contiguous_nodes (bool):
            Whether the Job has allocated a set of contiguous nodes.
        licenses (list):
            List of licenses the Job needs.
        allocated_licenses (list):
            List of licenses the Job has allocated
        network (str):
            Network specification for the Job.
        command (str):
            The command that is executed for the Job.
        working_directory (str):
            Path to the working directory for this Job.
        admin_comment (str):
            An arbitrary comment set by an administrator for the Job.
        system_comment (str):
            An arbitrary comment set by the slurmctld for the Job.
        container (str):
            The container this Job uses.
        container_id (str):
            The OCI ID of the Container this Job uses.
        comment (str):
            An arbitrary comment set for the Job.
        standard_input (str):
            The path to the file for the standard input stream.
        standard_output (str):
            The path to the log file for the standard output stream.
        standard_error (str):
            The path to the log file for the standard error stream.
        required_switches (int):
            Number of switches required.
        max_wait_time_switches (int):
            Amount of seconds to wait for the switches.
        burst_buffer (str):
            Burst buffer specification
        burst_buffer_state (str):
            Burst buffer state
        cpu_frequency_min (Union[str, int]):
            Minimum CPU-Frequency requested.
        cpu_frequency_max (Union[str, int]):
            Maximum CPU-Frequency requested.
        cpu_frequency_governor (Union[str, int]):
            CPU-Frequency Governor requested.
        billable_tres (float):
            Amount of billable trackable resources.
        wckey (str):
            Name of the WCKey this Job uses.
        mail_user (list):
            Users that should receive Mails for this Job.
        mail_types (list):
            Mail Flags specified by the User.
        heterogeneous_id (int):
            Heterogeneous job id.
        heterogeneous_offset (int):
            Heterogeneous job offset.
        temporary_disk_per_node (int):
            Temporary disk space in Mebibytes available per Node.
        array_id (int):
            The master Array-Job ID.
        array_tasks_parallel (int):
            Max number of array tasks allowed to run simultaneously.
        array_task_id (int):
            Array Task ID of this Job if it is an Array-Job.
        array_tasks_waiting (str):
            Array Tasks that are still waiting.
        end_time (int):
            Time at which this Job will end, as unix timestamp.
        run_time (int):
            Amount of seconds the Job has been running.
        cores_reserved_for_system (int):
            Amount of cores reserved for System use only.
        threads_reserved_for_system (int):
            Amount of Threads reserved for System use only.
        memory (int):
            Total Amount of Memory this Job has, in Mebibytes
        memory_per_cpu (int):
            Amount of Memory per CPU this Job has, in Mebibytes
        memory_per_node (int):
            Amount of Memory per Node this Job has, in Mebibytes
        memory_per_gpu (int):
            Amount of Memory per GPU this Job has, in Mebibytes
        profile_types (list):
            Types for which detailed accounting data is collected.
        gres_binding (str):
            Binding Enforcement of a Generic Resource (e.g. GPU).
        gres_tasks_per_sharing (str):
            Task Sharing of a Generic Resource (e.g. GPU).
        kill_on_invalid_dependency (bool):
            Whether the Job should be killed on an invalid dependency.
        kill_step_when_oom (bool):
            Whether the whole Step should be killed when a single task is Out
            of Memory.
        spreads_over_nodes (bool):
            Whether the Job should be spread over as many nodes as possible.
        is_cronjob (bool):
            Whether this Job is a cronjob.
        cronjob_time (str):
            The time specification for the Cronjob.
        elapsed_cpu_time (int):
            Amount of CPU-Time used by the Job so far.
            This is the result of multiplying the run_time with the amount of
            cpus requested.
        submit_line (str):
            The sbatch command that was used to submit the Job.
        run_time_remaining (int):
            The amount of seconds the job has still left until hitting the
            `time_limit`.
        extra (str):
            Some additional information to the Job.
        failed_node (str):
            Name of the node that caused the Job to fail.
        preferred_features (list):
            A list of preferred features/constraints
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
        tres_per_node (pyslurm.db.TrackableResources):
            The TRES Per Node the Job has allocated/requested
        tres_per_task (pyslurm.db.TrackableResources):
            The TRES Per Task the Job has allocated/requested
        tres_per_job (pyslurm.db.TrackableResources):
            The TRES Per Job allocated/requested
        tres_per_socket (pyslurm.db.TrackableResources):
            The TRES Per Socket the Job has allocated/requested
        reserved_ports (int):
            Number of Ports reserved for communications.
        selinux_context (str):
            The SELinux Context for the Job.
        segment_size (int):
            When a block topology is used, this defines the size of the
            segments that have been used to create the job allocation.
        spread_segments (bool):
            Whether segments should be spread.
        consolidate_segments (bool):
            Whether the segments should be consolidated.
        slurm_protocol_version (int):
            The Slurm Protocol Version used.
    """
    cdef:
        slurm_job_info_t *ptr
        dict passwd
        dict groups

    cdef public:
        JobSteps steps
        JobStatistics stats
        dict pids

    cdef _calc_run_time(self)

    @staticmethod
    cdef _swap_data(Job dst, Job src)

    @staticmethod
    cdef Job from_ptr(slurm_job_info_t *in_ptr)

