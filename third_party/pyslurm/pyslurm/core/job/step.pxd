#########################################################################
# job/step.pxd - interface to retrieve slurm job step information
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
from .job cimport Job
from libc.string cimport memcpy, memset
from pyslurm cimport slurm
from pyslurm.slurm cimport (
    slurm_step_id_t,
    job_step_info_t,
    slurm_get_job_steps,
    job_step_info_response_msg_t,
    step_update_request_msg_t,
    slurm_free_job_step_info_response_msg,
    slurm_init_update_step_msg,
    slurm_free_update_step_msg,
    slurm_free_job_step_info_response_msg,
    slurm_free_job_step_info_members,
    slurm_update_step,
    slurm_signal_job_step,
    slurm_kill_job_step,
    slurm_job_state_string,
    xfree,
    try_xmalloc,
)
from pyslurm.utils cimport cstr, ctime
from pyslurm.utils.uint cimport *
from pyslurm.utils.ctime cimport time_t
from pyslurm.core.job.task_dist cimport TaskDistribution
from pyslurm.db.stats cimport JobStepStatistics
from pyslurm.db.tres cimport TrackableResources, GPU
from pyslurm.core.job cimport stats
from pyslurm.utils.helpers cimport init_step_id


cdef class JobSteps(dict):
    """A [dict][] of [pyslurm.JobStep][] objects for a given Job.

    Raises:
        (pyslurm.RPCError): When getting the Job steps from the slurmctld
            failed.
    """

    cdef:
        job_step_info_response_msg_t *info
        job_step_info_t tmp_info
        _job_id

    @staticmethod
    cdef JobSteps _load_single(Job job)
    cdef dict _load_data(self, uint32_t job_id, int flags)


cdef class JobStep:
    """A Slurm Jobstep

    Args:
        job_id (Union[Job, int], optional=0):
            The Job this Step belongs to.
        step_id (Union[int, str], optional=0):
            Step-ID for this JobStep object.

    Other Parameters:
        time_limit (int):
            Time limit in Minutes for this step.

    Attributes:
        stats (pyslurm.db.JobStepStatistics):
            Real-time statistics of a Step.
            Before you can access the stats data for a Step, you have to call
            the `load_stats` method of a Step instance or the Jobs collection.
        pids (dict[str, list]):
            Current Process-IDs of the Step, organized by node name. Before you
            can access the pids data, you have to call the `load_stats` method
            of a Srep instance or the Jobs collection.
        id (Union[str, int]):
            The id for this step.
        job_id (int):
            The id for the Job this step belongs to.
        name (str):
            Name of the step.
        user_id (int):
            User ID who owns this step.
        user_name (str):
            Name of the User who owns this step.
        time_limit (int):
            Time limit in Minutes for this step.
        network (str):
            Network specification for the step.
        cpu_frequency_min (Union[str, int]):
            Minimum CPU-Frequency requested.
        cpu_frequency_max (Union[str, int]):
            Maximum CPU-Frequency requested.
        cpu_frequency_governor (Union[str, int]):
            CPU-Frequency Governor requested.
        reserved_ports (str):
            Reserved ports for the step.
        cluster (str):
            Name of the cluster this step runs on.
        srun_host (str):
            Name of the host srun was executed on.
        srun_process_id (int):
            Process ID of the srun command.
        container (str):
            Path to the container OCI.
        container_id (str):
            The ID of the OCI Container.
        allocated_nodes (str):
            Nodes the Job is using.
        start_time (int):
            Time this step started, as unix timestamp.
        run_time (int):
            Seconds this step has been running for.
        run_time_remaining (int):
            The amount of seconds the step has still left until hitting the
            `time_limit`.
        elapsed_cpu_time (int):
            Amount of CPU-Time used by the step so far.
            This is the result of multiplying the `run_time` with the amount of
            `cpus` allocated.
        partition (str):
            Name of the partition this step runs in.
        state (str):
            State the step is in.
        cpus (int):
            Number of CPUs this step uses in total.
        ntasks (int):
            Number of tasks this step uses.
        distribution (dict):
            Task distribution specification for the step.
        command (str):
            Command that was specified with srun.
        slurm_protocol_version (int):
            Slurm protocol version in use.
        array_id (int):
            The ID of the Array Job. Will return `None` when this Step doesn't
            belong to an Array Job.
        array_task_id (int):
            Array Task ID of this Step if it is an Array-Job. May return `None`
            if this isn't an Array Job.
        standard_input (str):
            The path to the file for the standard input stream.
        standard_output (str):
            The path to the log file for the standard output stream.
        standard_error (str):
            The path to the log file for the standard error stream.
        working_directory (str):
            Working directory of the Step
        gpus (dict[GPU]):
            A mapping of GPUs the Step has allocated.
        gres (dict):
            The Generic Resources the Step has allocated
        tres (pyslurm.db.TrackableResources):
            The TRES the Step has allocated.
        tres_per_node (pyslurm.db.TrackableResources):
            The TRES Per Node the Step has allocated
        tres_per_task (pyslurm.db.TrackableResources):
            The TRES Per Task the Step has allocated
        tres_per_step (pyslurm.db.TrackableResources):
            The TRES Per Step allocated
        tres_per_socket (pyslurm.db.TrackableResources):
            The TRES Per Socket the Step has allocated/requested
    """

    cdef:
        job_step_info_t *ptr
        step_update_request_msg_t *umsg

    cdef public:
        JobStepStatistics stats
        dict pids

    @staticmethod
    cdef JobStep from_ptr(job_step_info_t *in_ptr)

    cdef _get_stdio(self, char *path)
