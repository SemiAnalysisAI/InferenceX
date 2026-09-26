#########################################################################
# stats.pxd - pyslurm slurmdbd job stats
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
    try_xmalloc,
    slurmdb_stats_t,
    slurmdb_step_rec_t,
    slurmdb_job_rec_t,
)
from pyslurm.db.tres cimport TrackableResources
from pyslurm.db.step cimport JobStep, JobSteps
from pyslurm.db.job cimport Job
from pyslurm.utils cimport cstr

cdef class JobStatistics:
    """Statistics for a Slurm Job or Job Collection.

    Attributes:
        total_cpu_time (int):
            Sum of user_cpu_time and system_cpu_time, in seconds
        user_cpu_time (int):
            Total amount of time spent in user space, in seconds
        system_cpu_time (int):
            Total amount of time spent in kernel space, in seconds
        consumed_energy (int):
            Total amount of energy consumed, in joules
        elapsed_cpu_time (int):
            Total amount of time used(Elapsed time * cpu count) in seconds.
            This is not the real CPU-Efficiency, but rather the total amount
            of cpu-time the CPUs were occupied for.
        disk_read (int):
            Total amount of bytes read.
        disk_write (int):
            Total amount of bytes written.
        page_faults (int):
            Total amount of page faults.
        resident_memory (int):
            Total Resident Set Size (RSS) used in bytes.
        virtual_memory (int):
            Total Virtual Memory Size (VSZ) used in bytes.
    """
    cdef public:
        total_cpu_time
        user_cpu_time
        system_cpu_time
        elapsed_cpu_time
        consumed_energy
        disk_read
        disk_write
        page_faults
        resident_memory
        virtual_memory


cdef class JobStepStatistics:
    """Statistics for a Slurm JobStep.

    !!! note

        For more information also see the sacct manpage.

    Attributes:
        consumed_energy (int):
            Total amount of energy consumed, in joules
        elapsed_cpu_time (int):
            Total amount of time used(Elapsed time * cpu count) in seconds.
            This is not the real CPU-Efficiency, but rather the total amount
            of cpu-time the CPUs were occupied for
        avg_cpu_time (int):
            Average CPU-Time (System + User) in seconds of all tasks
        avg_cpu_frequency (int):
            Average weighted CPU-Frequency of all tasks, in Kilohertz
        avg_disk_read (int):
            Average number of bytes read by all tasks
        avg_disk_write (int):
            Average number of bytes written by all tasks
        avg_page_faults (int):
            Average number of page faults by all tasks
        avg_resident_memory (int):
            Average Resident Set Size (RSS) in bytes of all tasks
        avg_virtual_memory (int):
            Average Virtual Memory Size (VSZ) in bytes of all tasks
        max_disk_read (int):
            Highest peak number of bytes read by all tasks
        max_disk_read_node (int):
            Name of the Node where max_disk_read occurred
        max_disk_read_task (int):
            ID of the Task where max_disk_read occurred
        max_disk_write (int):
            Lowest peak number of bytes written by all tasks
        max_disk_write_node (int):
            Name of the Node where max_disk_write occurred
        max_disk_write_task (int):
            ID of the Task where max_disk_write occurred
        max_page_faults (int):
            Highest peak number of page faults by all tasks
        max_page_faults_node (int):
            Name of the Node where max_page_faults occurred
        max_page_faults_task (int):
            ID of the Task where max_page_faults occurred
        max_resident_memory (int):
            Highest peak Resident Set Size (RSS) in bytes by all tasks
        max_resident_memory_node (int):
            Name of the Node where max_resident_memory occurred
        max_resident_memory_task (int):
            ID of the Task where max_resident_memory occurred
        max_virtual_memory (int):
            Highest peak Virtual Memory Size (VSZ) in bytes by all tasks
        max_virtual_memory_node (int):
            Name of the Node where max_virtual_memory occurred
        max_virtual_memory_task (int):
            ID of the Task where max_virtual_memory occurred
        min_cpu_time (int):
            Lowest peak CPU-Time (System + User) in seconds of all tasks
        min_cpu_time_node (int):
            Name of the Node where min_cpu_time occurred
        min_cpu_time_task (int):
            ID of the Task where min_cpu_time occurred
        total_cpu_time (int):
            Sum of user_cpu_time and system_cpu_time, in seconds
        user_cpu_time (int):
            Amount of Time spent in user space, in seconds
        system_cpu_time (int):
            Amount of Time spent in kernel space, in seconds
    """
    cdef public:
        consumed_energy
        elapsed_cpu_time
        avg_cpu_time
        avg_cpu_frequency
        avg_disk_read
        avg_disk_write
        avg_page_faults
        avg_resident_memory
        avg_virtual_memory
        max_disk_read
        max_disk_read_node
        max_disk_read_task
        max_disk_write
        max_disk_write_node
        max_disk_write_task
        max_page_faults
        max_page_faults_node
        max_page_faults_task
        max_resident_memory
        max_resident_memory_node
        max_resident_memory_task
        max_virtual_memory
        max_virtual_memory_node
        max_virtual_memory_task
        min_cpu_time
        min_cpu_time_node
        min_cpu_time_task
        total_cpu_time
        user_cpu_time
        system_cpu_time

    @staticmethod
    cdef JobStepStatistics from_step(JobStep step)

    @staticmethod
    cdef JobStepStatistics from_ptr(slurmdb_step_rec_t *step, list nodes, cpus=*, elapsed_time=*, is_live=*)

