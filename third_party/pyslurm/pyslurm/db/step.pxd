#########################################################################
# step.pxd - pyslurm slurmdbd step api
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
    slurmdb_job_cond_def_start_end,
    slurm_job_state_string,
    slurm_job_state_reason_string,
    try_xmalloc,
    xfree,
)
from pyslurm.db.util cimport SlurmList, SlurmListItem
from pyslurm.db.connection cimport Connection
from pyslurm.utils cimport cstr
from pyslurm.db.stats cimport JobStepStatistics
from pyslurm.db.tres cimport TrackableResources, TrackableResource, GPU


cdef class JobSteps(dict):
    """A [dict][] of [pyslurm.db.JobStep][] objects"""
    pass


cdef class JobStep:
    """A Slurm Database JobStep.

    Attributes:
        stats (pyslurm.db.JobStepStatistics):
            Utilization statistics for this Step
        num_nodes (int):
            Amount of nodes this Step has allocated
        cpus (int):
            Amount of CPUs the Step has/had allocated
        memory (int):
            Amount of memory the Step requested
        container (str):
            Path to OCI Container bundle
        elapsed_time (int):
            Amount of seconds elapsed for the Step
        end_time (int):
            When the Step ended, as a unix timestamp
        eligible_time (int):
            When the Step became eligible to run, as a unix timestamp
        start_time (int):
            Time when the Step started, as a unix timestamp
        exit_code (int):
            Exit code of the step
        ntasks (int):
            Number of tasks the Step uses
        cpu_frequency_min (str):
            Minimum CPU-Frequency requested for the Step
        cpu_frequency_max (str):
            Maximum CPU-Frequency requested for the Step
        cpu_frequency_governor (str):
            CPU-Frequency Governor requested for the Step
        nodelist (str):
            Nodes this Step is using
        id (Union[str, int]):
            ID of the Step
        job_id (int):
            ID of the Job this Step is a part of
        state (str):
            State of the Step
        cancelled_by (str):
            Name of the User who cancelled this Step
        submit_command (str):
            Full command issued to start the Step
        suspended_time (int):
            Amount of seconds the Step was suspended.
        working_directory (str):
            Working directory of the Step
        standard_input (str):
            The path to the file for the standard input stream.
        standard_output (str):
            The path to the log file for the standard output stream.
        standard_error (str):
            The path to the log file for the standard error stream.
        time_limit (int):
            Time limit in Minutes for this step.
        gpus (dict[GPU]):
            A mapping of GPUs the Step has allocated.
        gres (dict):
            The Generic Resources the Step has allocated
        tres (pyslurm.db.TrackableResources):
            The TRES the Step has allocated.
    """
    cdef:
        slurmdb_step_rec_t *ptr
        TrackableResources tres_data

    cdef public:
        JobStepStatistics stats

    @staticmethod
    cdef JobStep from_ptr(slurmdb_step_rec_t *step)

    cdef _get_stdio(self, char *path)
