#########################################################################
# step.pyx - pyslurm slurmdbd step api
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

from os import WIFSIGNALED, WIFEXITED, WTERMSIG, WEXITSTATUS
from pyslurm.core.error import RPCError
from typing import Union
from pyslurm.utils.uint import *
from pyslurm.utils.ctime import _raw_time
from pyslurm import xcollections
from pyslurm.utils.helpers import (
    gid_to_name,
    uid_to_name,
    instance_to_dict,
    _get_exit_code,
    humanize_step_id,
    cpu_freq_int_to_str,
)


cdef class JobSteps(dict):

    def __repr__(self):
        data = super().__repr__()
        return f'pyslurm.db.{self.__class__.__name__}({data})'

    def to_dict(self, recursive = False):
        return xcollections.dict_recursive(self, recursive)


cdef class JobStep:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self):
        raise RuntimeError("You can not instantiate this class directly "
                           " at the moment")

    def __dealloc__(self):
        slurmdb_destroy_step_rec(self.ptr)
        self.ptr = NULL

    @staticmethod
    cdef JobStep from_ptr(slurmdb_step_rec_t *step):
        cdef JobStep wrap = JobStep.__new__(JobStep)
        wrap.ptr = step
        wrap.stats = JobStepStatistics.from_step(wrap)
        return wrap

    def to_dict(self, recursive = False):
        """Convert Database JobStep information to a dictionary.

        Returns:
            (dict): Database JobStep information as dict
        """
        return instance_to_dict(self, recursive)

    def __repr__(self):
        return f'pyslurm.db.{self.__class__.__name__}({self.id})'

    @property
    def num_nodes(self):
        nnodes = u32_parse(self.ptr.nnodes)
        if not nnodes and self.ptr.tres_alloc_str:
            return TrackableResources.find_count_in_str(
                    self.ptr.tres_alloc_str, slurm.TRES_NODE)
        else:
            return nnodes

    @property
    def cpus(self):
        req_cpus = TrackableResources.find_count_in_str(
                self.ptr.tres_alloc_str, slurm.TRES_CPU)

        if req_cpus == slurm.INFINITE64:
            return 0

        return req_cpus
#       if req_cpus == slurm.INFINITE64 and step.job_ptr:
#           tres_alloc_str = cstr.to_unicode(step.job_ptr.tres_alloc_str)
#           req_cpus = TrackableResources.find_count_in_str(tres_alloc_str,
#                                                           slurm.TRES_CPU)
#           if not req_cpus:
#               tres_req_str = cstr.to_unicode(step.job_ptr.tres_req_str)
#               req_cpus = TrackableResources.find_count_in_str(tres_req_str,
#                                                                slurm.TRES_CPU)

    @property
    def memory(self):
        val = TrackableResources.find_count_in_str(self.ptr.tres_alloc_str,
                                                   slurm.TRES_MEM)
        return val

    @property
    def container(self):
        return cstr.to_unicode(self.ptr.container)

    @property
    def elapsed_time(self):
        # seconds
        return _raw_time(self.ptr.elapsed)

    @property
    def end_time(self):
        return _raw_time(self.ptr.end)

    @property
    def eligible_time(self):
        return _raw_time(self.ptr.start)

    @property
    def start_time(self):
        return _raw_time(self.ptr.start)

    @property
    def exit_code(self):
        ec, _ = _get_exit_code(self.ptr.exitcode)
        return ec

    @property
    def exit_code_signal(self):
        _, sig = _get_exit_code(self.ptr.exitcode)
        return sig

    @property
    def ntasks(self):
        return u32_parse(self.ptr.ntasks)

    @property
    def cpu_frequency_min(self):
        return cpu_freq_int_to_str(self.ptr.req_cpufreq_min)

    @property
    def cpu_frequency_max(self):
        return cpu_freq_int_to_str(self.ptr.req_cpufreq_max)

    @property
    def cpu_frequency_governor(self):
        return cpu_freq_int_to_str(self.ptr.req_cpufreq_gov)

    @property
    def nodelist(self):
        return cstr.to_unicode(self.ptr.nodes)

    @property
    def id(self):
        return humanize_step_id(self.ptr.step_id.step_id)

    @property
    def job_id(self):
        return self.ptr.step_id.job_id

    @property
    def name(self):
        return cstr.to_unicode(self.ptr.stepname)

# TODO
#    @property
#    def distribution(self):
#        # ptr.task_dist
#        pass

    @property
    def state(self):
        return cstr.to_unicode(slurm_job_state_string(self.ptr.state))

    @property
    def cancelled_by(self):
        return uid_to_name(self.ptr.requid)

    @property
    def submit_command(self):
        return cstr.to_unicode(self.ptr.submit_line)

    @property
    def suspended_time(self):
        return _raw_time(self.ptr.suspended)

    @property
    def working_directory(self):
        return cstr.to_unicode(self.ptr.cwd)

    cdef _get_stdio(self, char *path):
        cdef char *tmp_path = slurm.slurmdb_expand_step_stdio_fields(path, self.ptr)
        path_str = cstr.to_unicode(tmp_path)
        xfree(tmp_path)
        return path_str

    @property
    def standard_input(self):
        return self._get_stdio(self.ptr.std_in)

    @property
    def standard_output(self):
        return self._get_stdio(self.ptr.std_out)

    @property
    def standard_error(self):
        return self._get_stdio(self.ptr.std_err)

    @property
    def time_limit(self):
        return _raw_time(self.ptr.timelimit)

    @property
    def tres(self):
        return TrackableResources.from_cstr(self.ptr.tres_alloc_str, self.tres_data)

    @property
    def gres(self):
        return self.tres.gres if self.tres else {}

    @property
    def gpus(self):
        return {k: v for k, v in self.gres.items() if isinstance(v, GPU)}
