#########################################################################
# job/step.pyx - interface to retrieve slurm job step information
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

from typing import Union
from pyslurm.utils import cstr, ctime
from pyslurm.utils.uint import *
from pyslurm.core.error import RPCError, verify_rpc
from pyslurm import settings
from pyslurm import xcollections
from pyslurm.utils.helpers import (
    signal_to_num,
    instance_to_dict,
    uid_to_name,
    humanize_step_id,
    dehumanize_step_id,
    cpu_freq_int_to_str,
)
from pyslurm.utils.ctime import (
    secs_to_timestr,
    mins_to_timestr,
    timestr_to_mins,
    timestamp_to_date,
    _raw_time,
)


cdef class JobSteps(dict):

    def __dealloc__(self):
        slurm_free_job_step_info_response_msg(self.info)

    def __cinit__(self):
        self.info = NULL

    def __init__(self, steps=None):
        if isinstance(steps, dict):
            self.update(steps)
        elif steps is not None:
            raise TypeError("Invalid Type: {type(steps)}")

    def __repr__(self):
        data = super().__repr__()
        return f'pyslurm.{self.__class__.__name__}({data})'

    @staticmethod
    def load(job):
        """Load the Job Steps from the system.

        Args:
            job (Union[Job, int]):
                The Job for which the Steps should be loaded.

        Returns:
            (pyslurm.JobSteps): JobSteps of the Job

        Examples:
            >>> import pyslurm
            >>> steps = pyslurm.JobSteps.load(1)
            >>> print(steps)
            pyslurm.JobSteps({'batch': pyslurm.JobStep('batch')})
            >>> print(steps[1])
            pyslurm.JobStep('batch')
        """
        cdef:
            Job _job
            JobSteps steps

        _job = Job.load(job.id if isinstance(job, Job) else job)
        steps = JobSteps._load_single(_job)
        steps._job_id = _job.id
        return steps

    @staticmethod
    cdef JobSteps _load_single(Job job):
        cdef JobSteps steps = JobSteps()

        data = steps._load_data(job.id, slurm.SHOW_ALL)
        if not data and not slurm.IS_JOB_PENDING(job.ptr):
            msg = f"Failed to load step info for Job {job.id}."
            raise RPCError(msg=msg)

        steps.update(data[job.id])
        return steps

    cdef dict _load_data(self, uint32_t job_id, int flags):
        cdef:
            JobStep step
            uint32_t cnt = 0
            slurm_step_id_t step_id = init_step_id()
            dict steps = {}

        step_id.job_id = job_id
        rc = slurm_get_job_steps(&step_id, &self.info, flags)
        verify_rpc(rc)

        # zero-out a dummy job_step_info_t
        memset(&self.tmp_info, 0, sizeof(job_step_info_t))

        # Put each job-step pointer into its own "JobStep" instance.
        for cnt in range(self.info.job_step_count):
            step = JobStep.from_ptr(&self.info.job_steps[cnt])
            # Prevent double free if xmalloc fails mid-loop and a MemoryError
            # is raised by replacing it with a zeroed-out job_step_info_t.
            self.info.job_steps[cnt] = self.tmp_info

            job_id = step.job_id
            if not job_id in steps:
                steps[job_id] = JobSteps()
            steps[job_id][step.id] = step

        # We have extracted all pointers
        self.info.job_step_count = 0
        return steps

    @staticmethod
    def load_all():
        """Loads all the steps in the system.

        Returns:
            (dict): A dict where every JobID (key) is mapped with an instance
                of its JobSteps (value).
        """
        cdef JobSteps steps = JobSteps()
        return steps._load_data(slurm.NO_VAL, slurm.SHOW_ALL)

    def to_dict(self, recursive = False):
        return xcollections.dict_recursive(self, recursive)


cdef class JobStep:

    def __cinit__(self):
        self.ptr = NULL
        self.umsg = NULL

    def __init__(self, job_id=0, step_id=0, **kwargs):
        self._alloc_impl()
        self.job_id = job_id.id if isinstance(job_id, Job) else job_id
        self.id = step_id
        self.stats = JobStepStatistics()
        self.pids = {}
        cstr.fmalloc(&self.ptr.cluster, settings.LOCAL_CLUSTER)

        # Initialize attributes, if any were provided
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _alloc_info(self):
        if not self.ptr:
            self.ptr = <job_step_info_t*>try_xmalloc(
                    sizeof(job_step_info_t))
            if not self.ptr:
                raise MemoryError("xmalloc failed for job_step_info_t")

    def _alloc_umsg(self):
        if not self.umsg:
            self.umsg = <step_update_request_msg_t*>try_xmalloc(
                    sizeof(step_update_request_msg_t))
            if not self.ptr:
                raise MemoryError("xmalloc failed for "
                                  "step_update_request_msg_t")
            slurm_init_update_step_msg(self.umsg)

    def _alloc_impl(self):
        self._alloc_info()
        self._alloc_umsg()

    def __dealloc__(self):
        self._dealloc_impl()

    def _dealloc_impl(self):
        slurm_free_job_step_info_members(self.ptr)
        # Bug in slurm_free_job_step_info_members - submit_line is not freed
        # there.
        xfree(self.ptr.submit_line)
        xfree(self.ptr)
        slurm_free_update_step_msg(self.umsg)
        self.umsg = NULL

    def __setattr__(self, name, val):
        # When a user wants to set attributes on a instance that was created
        # by calling JobSteps.load(), the "umsg" pointer is not yet allocated.
        # We only allocate memory for it by the time the user actually wants
        # to modify something.
        self._alloc_umsg()
        # Call descriptors __set__ directly
        JobStep.__dict__[name].__set__(self, val)

    def __repr__(self):
        return f'pyslurm.{self.__class__.__name__}({self.id})'

    @staticmethod
    def load(job_id, step_id):
        """Load information for a specific job step.

        Implements the slurm_get_job_steps RPC.

        Args:
            job_id (Union[pyslurm.Job, int]):
                ID of the Job the Step belongs to.
            step_id (Union[int, str]):
                Step-ID for the Step to be loaded.

        Returns:
            (pyslurm.JobStep): Returns a new JobStep instance

        Raises:
            (pyslurm.RPCError): When retrieving Step information from the
                slurmctld was not successful.

        Examples:
            >>> import pyslurm
            >>> jobstep = pyslurm.JobStep.load(9999, 1)
        """
        cdef:
            job_step_info_response_msg_t *info = NULL
            JobStep wrap = None
            slurm_step_id_t _step_id = init_step_id()

        _step_id.job_id = job_id.id if isinstance(job_id, Job) else job_id
        _step_id.step_id = dehumanize_step_id(step_id)
        rc = slurm_get_job_steps(&_step_id, &info, slurm.SHOW_ALL)
        verify_rpc(rc)

        if info and info.job_step_count == 1:
            wrap = JobStep.from_ptr(&info.job_steps[0])
            info.job_step_count = 0
            slurm_free_job_step_info_response_msg(info)
        else:
            slurm_free_job_step_info_response_msg(info)
            msg = f"Step {step_id} of Job {job_id} not found."
            raise RPCError(msg=msg)

        return wrap

    @staticmethod
    cdef JobStep from_ptr(job_step_info_t *in_ptr):
        cdef JobStep wrap = JobStep.__new__(JobStep)
        wrap._alloc_info()
        wrap.stats = JobStepStatistics()
        wrap.pids = {}
        memcpy(wrap.ptr, in_ptr, sizeof(job_step_info_t))
        return wrap

    def load_stats(self):
        """Load realtime stats for this Step.

        Calling this function returns the live statistics of the step, and
        additionally populates the `stats` and `pids` attribute of the
        instance.

        Returns:
            (pyslurm.db.JobStepStatistics): The statistics of the Step.

        Raises:
            (pyslurm.RPCError): When retrieving the stats for the Step failed.

        Examples:
            >>> import pyslurm
            >>> step = pyslurm.JobStep.load(9999, 1)
            >>> stats = step.load_stats()
            >>>
            >>> # Print the CPU Time Used
            >>> print(stats.total_cpu_time)
            >>>
            >>> # Print the Process-IDs for the Step, organized by hostname
            >>> print(step.pids)
        """
        stats.load_single(self)
        return self.stats

    def send_signal(self, signal):
        """Send a signal to a running Job step.

        Implements the slurm_signal_job_step RPC.

        Args:
            signal (Union[str, int]):
                Any valid signal which will be sent to the Job. Can be either
                a str like `SIGUSR1`, or simply an [int][].

        Raises:
            (pyslurm.RPCError): When sending the signal was not successful.

        Examples:
            Specifying the signal as a string:

            >>> import pyslurm
            >>> pyslurm.JobStep(9999, 1).send_signal("SIGUSR1")

            or passing in a numeric signal:

            >>> pyslurm.JobStep(9999, 1).send_signal(9)
        """
        sig = signal_to_num(signal)
        verify_rpc(slurm_signal_job_step(&self.ptr.step_id, sig))

    def cancel(self):
        """Cancel a Job step.

        Implements the slurm_kill_job_step RPC.

        Raises:
            (pyslurm.RPCError): When cancelling the Job was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.JobStep(9999, 1).cancel()
        """
        verify_rpc(slurm_kill_job_step(&self.ptr.step_id, 9, 0))

    def modify(self, JobStep changes):
        """Modify a job step.

        Implements the slurm_update_step RPC.

        Args:
            changes (pyslurm.JobStep):
                Another JobStep object that contains all the changes to apply.
                Check the `Other Parameters` of the JobStep class to see which
                properties can be modified.

        Raises:
            (pyslurm.RPCError): When updating the JobStep was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> # Setting the new time-limit to 20 days
            >>> changes = pyslurm.JobStep(time_limit="20-00:00:00")
            >>> pyslurm.JobStep(9999, 1).modify(changes)
        """
        cdef JobStep js = <JobStep>changes
        js._alloc_umsg()
        js.umsg.step_id = self.ptr.step_id
        verify_rpc(slurm_update_step(js.umsg))

    def as_dict(self):
        return self.to_dict()

    def to_dict(self, recursive = False):
        """JobStep information formatted as a dictionary.

        Returns:
            (dict): JobStep information as dict
        """
        return instance_to_dict(self, recursive)

    @property
    def id(self):
        return humanize_step_id(self.ptr.step_id.step_id)

    @id.setter
    def id(self, val):
        self.ptr.step_id.step_id = dehumanize_step_id(val)

    @property
    def job_id(self):
        return self.ptr.step_id.job_id

    @job_id.setter
    def job_id(self, val):
        self.ptr.step_id.job_id = int(val)

    @property
    def name(self):
        return cstr.to_unicode(self.ptr.name)

    @property
    def user_id(self):
        return u32_parse(self.ptr.user_id, zero_is_noval=False)

    @property
    def user_name(self):
        return uid_to_name(self.ptr.user_id)

    @property
    def time_limit(self):
        return _raw_time(self.ptr.time_limit)

    @time_limit.setter
    def time_limit(self, val):
        self.umsg.time_limit=self.ptr.time_limit = timestr_to_mins(val)

    @property
    def network(self):
        return cstr.to_unicode(self.ptr.network)

    @property
    def cpu_frequency_min(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_min)

    @property
    def cpu_frequency_max(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_max)

    @property
    def cpu_frequency_governor(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_gov)

    @property
    def reserved_ports(self):
        return cstr.to_unicode(self.ptr.resv_ports)

    @property
    def cluster(self):
        return cstr.to_unicode(self.ptr.cluster)

    @property
    def srun_host(self):
        return cstr.to_unicode(self.ptr.srun_host)

    @property
    def srun_process_id(self):
        return u32_parse(self.ptr.srun_pid)

    @property
    def container(self):
        return cstr.to_unicode(self.ptr.container)

    @property
    def container_id(self):
        return cstr.to_unicode(self.ptr.container_id)

    def array_id(self):
        return u32_parse(self.ptr.array_job_id)

    @property
    def array_task_id(self):
        return u32_parse(self.ptr.array_task_id)

    @property
    def allocated_nodes(self):
        return cstr.to_list(self.ptr.nodes)

    @property
    def start_time(self):
        return _raw_time(self.ptr.start_time)

    @property
    def run_time(self):
        return _raw_time(self.ptr.run_time, on_noval=0, on_inf=0)

    @property
    def run_time_remaining(self):
        limit = self.time_limit
        if limit is None:
            return None

        return (limit*60) - self.run_time

    @property
    def elapsed_cpu_time(self):
        return self.run_time * self.cpus

    @property
    def partition(self):
        return cstr.to_unicode(self.ptr.partition)

    @property
    def state(self):
        return cstr.to_unicode(slurm_job_state_string(self.ptr.state))

    @property
    def cpus(self):
        return u32_parse(self.ptr.num_cpus, on_noval=1)

    @property
    def ntasks(self):
        return u32_parse(self.ptr.num_tasks, on_noval=1)

    @property
    def distribution(self):
        return TaskDistribution.from_int(self.ptr.task_dist)

    @property
    def command(self):
        return cstr.to_unicode(self.ptr.submit_line)

    @property
    def slurm_protocol_version(self):
        return u32_parse(self.ptr.start_protocol_ver)

    cdef _get_stdio(self, char *path):
        cdef char *tmp_path = slurm.slurm_expand_step_stdio_fields(path, self.ptr)
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
    def working_directory(self):
        return cstr.to_unicode(self.ptr.cwd)

    @property
    def gres(self):
        tres = self.tres
        return tres.gres if tres else {}

    @property
    def gpus(self):
        return {k: v for k, v in self.gres.items() if isinstance(v, GPU)}

    @property
    def tres(self):
        return TrackableResources.from_cstr(self.ptr.tres_fmt_alloc_str)

    @property
    def tres_per_step(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_step)

    @property
    def tres_per_socket(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_socket)

    @property
    def tres_per_node(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_node)

    @property
    def tres_per_task(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_task)
