#########################################################################
# stats.pyx - pyslurm slurmdbd job stats
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

from pyslurm.utils.helpers import (
    nodelist_from_range_str,
    instance_to_dict,
)


cdef class JobStatistics:

    def __init__(self):
        for attr, val in instance_to_dict(self).items():
            setattr(self, attr, 0)

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)

    @staticmethod
    def from_steps(steps):
        cdef JobStatistics total_stats = JobStatistics()
        for step in steps.values():
            total_stats._sum_steps(step.stats)

        return total_stats

    def _sum_steps(self, src):
        self.consumed_energy += src.consumed_energy
        self.disk_read += src.avg_disk_read
        self.disk_write += src.avg_disk_write
        self.page_faults += src.avg_page_faults
        self.total_cpu_time += src.total_cpu_time
        self.user_cpu_time += src.user_cpu_time
        self.system_cpu_time += src.system_cpu_time

        if src.max_resident_memory > self.resident_memory:
            self.resident_memory = src.max_resident_memory

        if src.max_virtual_memory > self.resident_memory:
            self.virtual_memory = src.max_virtual_memory

    def add(self, src):
        self.consumed_energy += src.consumed_energy
        self.disk_read += src.disk_read
        self.disk_write += src.disk_write
        self.page_faults += src.page_faults
        self.total_cpu_time += src.total_cpu_time
        self.user_cpu_time += src.user_cpu_time
        self.system_cpu_time += src.system_cpu_time
        self.resident_memory += src.resident_memory
        self.virtual_memory += src.virtual_memory
        self.elapsed_cpu_time += src.elapsed_cpu_time


cdef class JobStepStatistics:

    def __init__(self):
        for attr, val in instance_to_dict(self).items():
            setattr(self, attr, 0)

        self.max_disk_read_node = None
        self.max_disk_read_task = None
        self.max_disk_write_node = None
        self.max_disk_write_task = None
        self.max_page_faults_node = None
        self.max_page_faults_task = None
        self.max_resident_memory_node = None
        self.max_resident_memory_task = None
        self.max_virtual_memory_node = None
        self.max_virtual_memory_task = None
        self.min_cpu_time_node = None
        self.min_cpu_time_task = None

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)

    @staticmethod
    cdef JobStepStatistics from_step(JobStep step):
        return JobStepStatistics.from_ptr(
            step.ptr,
            nodelist_from_range_str(cstr.to_unicode(step.ptr.nodes)),
            step.cpus if step.cpus else 0,
            step.elapsed_time if step.elapsed_time else 0,
            is_live=False,
        )

    @staticmethod
    cdef JobStepStatistics from_ptr(slurmdb_step_rec_t *step, list nodes, cpus=0, elapsed_time=0, is_live=False):
        cdef JobStepStatistics wrap = JobStepStatistics()
        if not step:
            return wrap

        cdef:
            cpu_time_adj = 1000
            slurmdb_stats_t *ptr = &step.stats

        if ptr.consumed_energy != slurm.NO_VAL64:
            wrap.consumed_energy = ptr.consumed_energy

        wrap.avg_cpu_time = int(TrackableResources.find_count_in_str(
                ptr.tres_usage_in_ave, slurm.TRES_CPU) / cpu_time_adj)

        wrap.elapsed_cpu_time = elapsed_time * cpus

        ave_freq = int(ptr.act_cpufreq)
        if ave_freq != slurm.NO_VAL:
            wrap.avg_cpu_frequency = ptr.act_cpufreq

        wrap.avg_disk_read = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_ave, slurm.TRES_FS_DISK)
        wrap.avg_disk_write = TrackableResources.find_count_in_str(
                ptr.tres_usage_out_ave, slurm.TRES_FS_DISK)
        wrap.avg_page_faults = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_ave, slurm.TRES_PAGES)
        wrap.avg_resident_memory = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_ave, slurm.TRES_MEM)
        wrap.avg_virtual_memory = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_ave, slurm.TRES_VMEM)

        wrap.max_disk_read = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max, slurm.TRES_FS_DISK)
        max_disk_read_nodeid = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max_nodeid, slurm.TRES_FS_DISK)
        wrap.max_disk_read_task = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max_taskid, slurm.TRES_FS_DISK)

        wrap.max_disk_write = TrackableResources.find_count_in_str(
                ptr.tres_usage_out_max, slurm.TRES_FS_DISK)
        max_disk_write_nodeid = TrackableResources.find_count_in_str(
                ptr.tres_usage_out_max_nodeid, slurm.TRES_FS_DISK)
        wrap.max_disk_write_task = TrackableResources.find_count_in_str(
                ptr.tres_usage_out_max_taskid, slurm.TRES_FS_DISK)

        wrap.max_resident_memory = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max, slurm.TRES_MEM)
        max_resident_memory_nodeid = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max_nodeid, slurm.TRES_MEM)
        wrap.max_resident_memory_task = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max_taskid, slurm.TRES_MEM)

        wrap.max_virtual_memory = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max, slurm.TRES_VMEM)
        max_virtual_memory_nodeid = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max_nodeid, slurm.TRES_VMEM)
        wrap.max_virtual_memory_task = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_max_taskid, slurm.TRES_VMEM)

        wrap.min_cpu_time = int(TrackableResources.find_count_in_str(
                ptr.tres_usage_in_min, slurm.TRES_CPU) / cpu_time_adj)
        min_cpu_time_nodeid = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_min_nodeid, slurm.TRES_CPU)
        wrap.min_cpu_time_task = TrackableResources.find_count_in_str(
                ptr.tres_usage_in_min_taskid, slurm.TRES_CPU)

        # The Total CPU-Time extracted here is only used for live-stats.
        # sacct does not use it from the tres_usage_in_tot string, but instead
        # the tot_cpu_sec value from the step pointer directly, so do that too.
        if is_live:
            wrap.total_cpu_time = int(TrackableResources.find_count_in_str(
                    ptr.tres_usage_in_tot, slurm.TRES_CPU) / cpu_time_adj)
        elif step.tot_cpu_sec != slurm.NO_VAL64:
            wrap.total_cpu_time += step.tot_cpu_sec

        if step.user_cpu_sec != slurm.NO_VAL64:
            wrap.user_cpu_time += step.user_cpu_sec

        if step.sys_cpu_sec != slurm.NO_VAL64:
            wrap.system_cpu_time += step.sys_cpu_sec

        if nodes:
            wrap.max_disk_write_node = nodes[max_disk_write_nodeid]
            wrap.max_disk_read_node = nodes[max_disk_read_nodeid]
            wrap.max_resident_memory_node = nodes[max_resident_memory_nodeid]
            wrap.max_virtual_memory_node = nodes[max_virtual_memory_nodeid]
            wrap.min_cpu_time_node = nodes[min_cpu_time_nodeid]

        return wrap
