#########################################################################
# job.pyx - interface to retrieve slurm job information
#########################################################################
# Copyright (C) 2023 Toni Harzendorf <toni.harzendorf@gmail.com>
#
# Note: Some functions in this File are annotated with additional Copyright
# notices. These functions are:
#
# - get_batch_script
# - get_resource_layout_per_node
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

from os import WIFSIGNALED, WIFEXITED, WTERMSIG, WEXITSTATUS
import re
from typing import Union
from pyslurm.utils import cstr, ctime
from pyslurm.utils.uint import *
from pyslurm.core.job.util import *
from pyslurm import settings
from pyslurm import xcollections
from pyslurm.core.error import (
    RPCError,
    verify_rpc,
    slurm_errno,
)
from pyslurm.utils.ctime import _raw_time
from pyslurm.utils.helpers import (
    uid_to_name,
    gid_to_name,
    signal_to_num,
    _getgrall_to_dict,
    _getpwall_to_dict,
    instance_to_dict,
    _get_exit_code,
    cpu_freq_int_to_str,
)


cdef class Jobs(MultiClusterMap):

    def __cinit__(self):
        self.info = NULL

    def __dealloc__(self):
        slurm_free_job_info_msg(self.info)

    def __init__(self, jobs=None, frozen=False):
        self.frozen = frozen
        self.stats = JobStatistics()
        super().__init__(data=jobs,
                         typ="Jobs",
                         val_type=Job,
                         id_attr=Job.id,
                         key_type=int)

    @staticmethod
    def load(preload_passwd_info=False, frozen=False):
        """Retrieve all Jobs from the Slurm controller

        Args:
            preload_passwd_info (bool, optional):
                Decides whether to query passwd and groups information from
                the system.
                Could potentially speed up access to attributes of the Job
                where a UID/GID is translated to a name. If True, the
                information will fetched and stored in each of the Job
                instances.
            frozen (bool, optional):
                Decide whether this collection of Jobs should be frozen.

        Returns:
            (pyslurm.Jobs): A collection of Job objects.

        Raises:
            (pyslurm.RPCError): When getting all the Jobs from the slurmctld
                failed.

        Examples:
            >>> import pyslurm
            >>> jobs = pyslurm.Jobs.load()
            >>> print(jobs)
            pyslurm.Jobs({1: pyslurm.Job(1), 2: pyslurm.Job(2)})
            >>> print(jobs[1])
            pyslurm.Job(1)
        """
        cdef:
            dict passwd = {}
            dict groups = {}
            Jobs jobs = Jobs(frozen=frozen)
            int flags = slurm.SHOW_ALL | slurm.SHOW_DETAIL
            Job job

        verify_rpc(slurm_load_jobs(0, &jobs.info, flags))

        # If requested, preload the passwd and groups database to potentially
        # speedup lookups for an attribute in a Job, e.g. user_name or
        # group_name.
        if preload_passwd_info:
            passwd = _getpwall_to_dict()
            groups = _getgrall_to_dict()

        # zero-out a dummy job_step_info_t
        memset(&jobs.tmp_info, 0, sizeof(slurm_job_info_t))

        # Put each job pointer into its own "Job" instance.
        for cnt in range(jobs.info.record_count):
            job = Job.from_ptr(&jobs.info.job_array[cnt])

            # Prevent double free if xmalloc fails mid-loop and a MemoryError
            # is raised by replacing it with a zeroed-out slurm_job_info_t.
            jobs.info.job_array[cnt] = jobs.tmp_info

            if preload_passwd_info:
                job.passwd = passwd
                job.groups = groups

            cluster = job.cluster
            if cluster not in jobs.data:
                jobs.data[cluster] = {}
            jobs[cluster][job.id] = job

        # We have extracted all pointers
        jobs.info.record_count = 0
        jobs.frozen = frozen
        return jobs

    def reload(self):
        """Reload the information for jobs in a collection.

        Returns:
            (pyslurm.Jobs): Returns self

        Raises:
            (pyslurm.RPCError): When getting the Jobs from the slurmctld
                failed.
        """
        return xcollections.multi_reload(self, frozen=self.frozen)

    def load_steps(self):
        """Load all Job steps for this collection of Jobs.

        This function fills in the `steps` attribute for all Jobs in the
        collection.

        !!! note

            Pending Jobs will be ignored, since they don't have any Steps yet.

        Raises:
            (pyslurm.RPCError): When retrieving the information for all the
                Steps failed.
        """
        cdef dict steps = JobSteps.load_all()
        for job in self.values():
            jid = job.id
            if jid in steps:
                job.steps = steps[jid]

    def load_stats(self):
        """Load realtime stats for this collection of Jobs.

        This function additionally fills in the `stats` attribute for all Jobs
        in the collection, and also populates its own `stats` attribute.
        Implicitly calls `load_steps()`.

        !!! note

            Pending Jobs will be ignored, since they don't have any Stats yet.

        Returns:
            (pyslurm.db.JobStatistics): The statistics of this job collection.

        Raises:
            (pyslurm.RPCError): When retrieving the stats for all the Jobs
                failed.

        Examples:
            >>> import pyslurm
            >>> jobs = pyslurm.Jobs.load()
            >>> stats = jobs.load_stats()
            >>>
            >>> # Print the CPU Time Used
            >>> print(stats.total_cpu_time)
        """
        self.load_steps()
        stats = JobStatistics()
        for job in self.values():
            job.load_stats()
            stats.add(job.stats)

        self.stats = stats
        return self.stats

    @property
    def memory(self):
        return xcollections.sum_property(self, Job.memory)

    @property
    def cpus(self):
        return xcollections.sum_property(self, Job.cpus)

    @property
    def ntasks(self):
        return xcollections.sum_property(self, Job.ntasks)

    @property
    def elapsed_cpu_time(self):
        return xcollections.sum_property(self, Job.elapsed_cpu_time)


cdef class Job:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, job_id):
        self._alloc_impl()
        self.ptr.job_id = job_id
        self.passwd = {}
        self.groups = {}
        cstr.fmalloc(&self.ptr.cluster, settings.LOCAL_CLUSTER)
        self.steps = JobSteps()
        self.stats = JobStatistics()
        self.pids = {}

    def _alloc_impl(self):
        if not self.ptr:
            self.ptr = <slurm_job_info_t*>try_xmalloc(sizeof(slurm_job_info_t))
            if not self.ptr:
                raise MemoryError("xmalloc failed for job_info_t")

    def _dealloc_impl(self):
        slurm_free_job_info(self.ptr)
        self.ptr = NULL

    def __dealloc__(self):
        self._dealloc_impl()

    def __repr__(self):
        return f'pyslurm.{self.__class__.__name__}({self.id})'

    @staticmethod
    def load(job_id):
        """Load information for a specific Job.

        Implements the slurm_load_job RPC.

        !!! note

            If the Job is not pending, the related Job steps will also be
            loaded. Job statistics are however not loaded automatically.

        Args:
            job_id (int):
                An Integer representing a Job-ID.

        Returns:
            (pyslurm.Job): Returns a new Job instance

        Raises:
            (pyslurm.RPCError): If requesting the Job information from the
                slurmctld was not successful.

        Examples:
            >>> import pyslurm
            >>> job = pyslurm.Job.load(9999)
        """
        cdef:
            job_info_msg_t *info = NULL
            Job wrap = None

        try:
            verify_rpc(slurm_load_job(&info, job_id, slurm.SHOW_DETAIL))

            if info and info.record_count:
                wrap = Job.from_ptr(&info.job_array[0])
                info.record_count = 0

                if not slurm.IS_JOB_PENDING(wrap.ptr):
                    # Just ignore if the steps couldn't be loaded here.
                    try:
                        wrap.steps = JobSteps._load_single(wrap)
                    except RPCError:
                        pass
            else:
                raise RPCError(msg=f"RPC was successful but got no job data, "
                               "this should never happen")
        except Exception as e:
            raise e
        finally:
            slurm_free_job_info_msg(info)

        return wrap

    @staticmethod
    cdef Job from_ptr(slurm_job_info_t *in_ptr):
        cdef Job wrap = Job.__new__(Job)
        wrap._alloc_impl()
        wrap.passwd = {}
        wrap.groups = {}
        wrap.steps = JobSteps.__new__(JobSteps)
        wrap.stats = JobStatistics()
        wrap.pids = {}
        memcpy(wrap.ptr, in_ptr, sizeof(slurm_job_info_t))
        return wrap

    cdef _swap_data(Job dst, Job src):
        cdef slurm_job_info_t *tmp = NULL
        if dst.ptr and src.ptr:
            tmp = dst.ptr
            dst.ptr = src.ptr
            src.ptr = tmp

    def as_dict(self):
        return self.to_dict()

    def to_dict(self, recursive = False):
        """Job information formatted as a dictionary.

        Returns:
            (dict): Job information as dict
        """
        return instance_to_dict(self, recursive)

    def send_signal(self, signal, steps="children", hurry=False):
        """Send a signal to a running Job.

        Implements the slurm_signal_job RPC.

        Args:
            signal (Union[str, int]):
                Any valid signal which will be sent to the Job. Can be either
                a str like `SIGUSR1`, or simply an [int][].
            steps (str):
                Selects which steps should be signaled. Valid values for this
                are: `all`, `batch` and `children`. The default value is
                `children`, where all steps except the batch-step will be
                signaled.
                The value `batch` in contrast means, that only the batch-step
                will be signaled. With `all` every step is signaled.
            hurry (bool):
                If True, no burst buffer data will be staged out. The default
                value is False.

        Raises:
            (pyslurm.RPCError): When sending the signal was not successful.

        Examples:
            Specifying the signal as a string:

            >>> from pyslurm import Job
            >>> Job(9999).send_signal("SIGUSR1")

            or passing in a numeric signal:

            >>> Job(9999).send_signal(9)
        """
        cdef uint16_t flags = 0

        if steps.casefold() == "all":
            flags |= slurm.KILL_FULL_JOB
        elif steps.casefold() == "batch":
            flags |= slurm.KILL_JOB_BATCH

        if hurry:
            flags |= slurm.KILL_HURRY

        sig = signal_to_num(signal)
        slurm_kill_job(self.id, sig, flags)

        # Ignore errors when the Job is already done or when SIGKILL was
        # specified and the job id is already purged from slurmctlds memory.
        errno = slurm_errno()
        if (errno == slurm.ESLURM_ALREADY_DONE
                or errno == slurm.ESLURM_INVALID_JOB_ID and sig == 9):
            pass
        else:
            verify_rpc(errno)

    def cancel(self):
        """Cancel a Job.

        Implements the slurm_kill_job RPC.

        Raises:
            (pyslurm.RPCError): When cancelling the Job was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Job(9999).cancel()
        """
        self.send_signal(9)

    def suspend(self):
        """Suspend a running Job.

        Implements the slurm_suspend RPC.

        Raises:
            (pyslurm.RPCError): When suspending the Job was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Job(9999).suspend()
        """
        # TODO: Report as a misbehaviour to schedmd that slurm_suspend is not
        # correctly returning error code when it cannot find the job in
        # _slurm_rpc_suspend it should return ESLURM_INVALID_JOB_ID, but
        # returns -1
        # https://github.com/SchedMD/slurm/blob/master/src/slurmctld/proc_req.c#L4693
        verify_rpc(slurm_suspend(self.id))

    def unsuspend(self):
        """Unsuspend a currently suspended Job.

        Implements the slurm_resume RPC.

        Raises:
            (pyslurm.RPCError): When unsuspending the Job was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Job(9999).unsuspend()
        """
        # Same problem as described in suspend()
        verify_rpc(slurm_resume(self.id))

    def modify(self, JobSubmitDescription changes):
        """Modify a Job.

        Implements the slurm_update_job RPC.

        Args:
            changes (pyslurm.JobSubmitDescription):
                A JobSubmitDescription object which contains all the
                modifications that should be done on the Job.

        Raises:
            (pyslurm.RPCError): When updating the Job was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> # Setting the new time-limit to 20 days
            >>> changes = pyslurm.JobSubmitDescription(time_limit="20-00:00:00")
            >>> pyslurm.Job(9999).modify(changes)
        """
        changes._create_job_submit_desc(is_update=True)
        changes.ptr.step_id.job_id = self.id
        verify_rpc(slurm_update_job(changes.ptr))

    def hold(self, mode=None):
        """Hold a currently pending Job, preventing it from being scheduled.

        Args:
            mode (str):
                Determines in which mode the Job should be held. Possible
                values are `user` or `admin`. By default, the Job is held in
                `admin` mode, meaning only an Administrator will be able to
                release the Job again. If you specify the mode as `user`, the
                User will also be able to release the job.

        Raises:
            (pyslurm.RPCError): When holding the Job was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> # Holding a Job (in "admin" mode by default)
            >>> pyslurm.Job(9999).hold()
            >>>
            >>> # Holding a Job in "user" mode
            >>> pyslurm.Job(9999).hold(mode="user")
        """
        cdef JobSubmitDescription job_sub = JobSubmitDescription(priority=0)

        if mode and mode.casefold() == "user":
            job_sub.ptr.alloc_sid = slurm.ALLOC_SID_USER_HOLD

        self.modify(job_sub)

    def release(self):
        """Release a currently held Job, allowing it to be scheduled again.

        Raises:
            (pyslurm.RPCError): When releasing a held Job was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Job(9999).release()
        """
        self.modify(JobSubmitDescription(priority=slurm.INFINITE))

    def requeue(self, hold=False):
        """Requeue a currently running Job.

        Implements the slurm_requeue RPC.

        Args:
            hold (bool, optional):
                Controls whether the Job should be put in a held state or not.
                Default for this is `False`, so it will not be held.

        Raises:
            (pyslurm.RPCError): When requeing the Job was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> # Requeing a Job while allowing it to be
            >>> # scheduled again immediately
            >>> pyslurm.Job(9999).requeue()
            >>>
            >>> # Requeing a Job while putting it in a held state
            >>> pyslurm.Job(9999).requeue(hold=True)
        """
        cdef uint32_t flags = 0

        if hold:
            flags |= slurm.JOB_REQUEUE_HOLD

        verify_rpc(slurm_requeue(self.id, flags))

    def notify(self, msg):
        """Sends a message to the Jobs stdout.

        Implements the slurm_notify_job RPC.

        Args:
            msg (str):
                The message that should be sent.

        Raises:
            (pyslurm.RPCError): When sending the message to the Job was not
                successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Job(9999).notify("Hello Friends!")
        """
        verify_rpc(slurm_notify_job(self.id, msg))

    def load_stats(self):
        """Load realtime statistics for a Job and its steps.

        Calling this function returns the Job statistics, and additionally
        populates the `stats` and `pids` attribute of the instance.

        Returns:
            (pyslurm.db.JobStatistics): The statistics of the job.

        Raises:
            (pyslurm.RPCError): When receiving the Statistics was not
            successful.

        Examples:
            >>> import pyslurm
            >>> job = pyslurm.Job.load(9999)
            >>> stats = job.load_stats()
            >>>
            >>> # Print the CPU Time Used
            >>> print(stats.total_cpu_time)
            >>>
            >>> # Print the Process-IDs for the whole Job, organized by hostname
            >>> print(job.pids)
        """
        if not self.steps:
            job = Job.load(self.id)
            self.steps = job.steps

        all_pids = {}
        for step in self.steps.values():
            step.load_stats()
            self.stats._sum_steps(step.stats)

            for node, pids in step.pids.items():
                if node not in all_pids:
                    all_pids[node] = []

                all_pids[node].extend(pids)

        self.stats.elapsed_cpu_time = self.run_time * self.cpus

        self.pids = all_pids
        return self.stats

    def get_batch_script(self):
        """Return the content of the script for a Batch-Job.

        Returns:
            (str): The content of the batch script.

        Raises:
            (pyslurm.RPCError): When retrieving the Batch-Script for the Job
                was not successful.

        Examples:
            >>> import pyslurm
            >>> script = pyslurm.Job(9999).get_batch_script()
        """
        # The code for this function was taken from here:
        # https://github.com/SchedMD/slurm/blob/7162f15af8deaf02c3bbf940d59e818cdeb5c69d/src/api/job_info.c#L1319
        # and therefore reimplements the slurm_job_batch_script API call, with
        # slight modifications (e.g. Cython syntax). Otherwise we would have
        # to parse the FILE* ptr we get from it back into a char* which
        # would be a bit silly.
        #
        # The copyright notices for the file this function was taken from is
        # included below:
        #
        # Portions Copyright (C) 2010-2017 SchedMD LLC <https://www.schedmd.com>.
        # Copyright (C) 2002-2007 The Regents of the University of California.
        # Copyright (C) 2008-2010 Lawrence Livermore National Security.
        # Produced at Lawrence Livermore National Laboratory (cf, DISCLAIMER).
        # Written by Morris Jette <jette1@llnl.gov> et. al.
        # CODE-OCEC-09-009. All rights reserved.
        #
        # Slurm is licensed under the GNU General Public License. For the full
        # text of Slurm's License, please see here:
        # pyslurm/slurm/SLURM_LICENSE
        #
        # Please, as mentioned above, also have a look at Slurm's DISCLAIMER
        # under pyslurm/slurm/SLURM_DISCLAIMER
        cdef:
            job_id_msg_t msg
            slurm_msg_t req
            slurm_msg_t resp
            int rc = slurm.SLURM_SUCCESS
            str script = None

        slurm_msg_t_init(&req)
        slurm_msg_t_init(&resp)

        memset(&msg, 0, sizeof(msg))
        msg.step_id.job_id = self.id
        req.msg_type = slurm.REQUEST_BATCH_SCRIPT
        req.data = &msg

        rc = slurm_send_recv_controller_msg(&req, &resp, working_cluster_rec)
        verify_rpc(rc)

        if resp.msg_type == slurm.RESPONSE_BATCH_SCRIPT:
            script = cstr.to_unicode(<char*>resp.data)
            xfree(resp.data)
        elif resp.msg_type == slurm.RESPONSE_SLURM_RC:
            rc = (<return_code_msg_t*> resp.data).return_code
            slurm_free_return_code_msg(<return_code_msg_t*>resp.data)
            verify_rpc(rc)
        else:
            verify_rpc(slurm.SLURM_ERROR)

        return script

    @property
    def name(self):
        return cstr.to_unicode(self.ptr.name)

    @property
    def id(self):
        return self.ptr.job_id

    @property
    def association_id(self):
        return u32_parse(self.ptr.assoc_id)

    @property
    def account(self):
        return cstr.to_unicode(self.ptr.account)

    @property
    def user_id(self):
        return u32_parse(self.ptr.user_id, zero_is_noval=False)

    @property
    def user_name(self):
        return uid_to_name(self.ptr.user_id, lookup=self.passwd)

    @property
    def group_id(self):
        return u32_parse(self.ptr.group_id, zero_is_noval=False)

    @property
    def group_name(self):
        return gid_to_name(self.ptr.group_id, lookup=self.groups)

    @property
    def priority(self):
        return u32_parse(self.ptr.priority, zero_is_noval=False)

    @property
    def nice(self):
        if self.ptr.nice == slurm.NO_VAL:
            return None

        return int(self.ptr.nice) - slurm.NICE_OFFSET

    @property
    def qos(self):
        return cstr.to_unicode(self.ptr.qos)

    @property
    def min_cpus_per_node(self):
        return u32_parse(self.ptr.pn_min_cpus)

    # I don't think this is used anymore - there is no way in sbatch to ask
    # for a "maximum cpu" count, so it will always be empty.
    # @property
    # def max_cpus(self):
    #     """Maximum Amount of CPUs the Job requested."""
    #     return u32_parse(self.ptr.max_cpus)

    @property
    def state(self):
        return cstr.to_unicode(slurm_job_state_string(self.ptr.job_state))

    @property
    def state_reason(self):
        if self.ptr.state_desc:
            return cstr.to_unicode(self.ptr.state_desc)

        return cstr.to_unicode(slurm_job_state_reason_string(self.ptr.state_reason))

    @property
    def is_requeueable(self):
        return u16_parse_bool(self.ptr.requeue)

    @property
    def requeue_count(self):
        return u16_parse(self.ptr.restart_cnt, on_noval=0)

    @property
    def is_batch_job(self):
        return u16_parse_bool(self.ptr.batch_flag)

    @property
    def requires_node_reboot(self):
        return u8_parse_bool(self.ptr.reboot)

    @property
    def dependencies(self):
        return dependency_str_to_dict(cstr.to_unicode(self.ptr.dependency))

    @property
    def time_limit(self):
        return _raw_time(self.ptr.time_limit)

    @property
    def time_limit_min(self):
        return _raw_time(self.ptr.time_min)

    @property
    def submit_time(self):
        return _raw_time(self.ptr.submit_time)

    @property
    def eligible_time(self):
        return _raw_time(self.ptr.eligible_time)

    @property
    def accrue_time(self):
        return _raw_time(self.ptr.accrue_time)

    @property
    def start_time(self):
        return _raw_time(self.ptr.start_time)

    @property
    def resize_time(self):
        return _raw_time(self.ptr.resize_time)

    @property
    def deadline(self):
        return _raw_time(self.ptr.deadline)

    @property
    def preempt_eligible_time(self):
        return _raw_time(self.ptr.preemptable_time)

    @property
    def preempt_time(self):
        return _raw_time(self.ptr.preempt_time)

    @property
    def suspend_time(self):
        return _raw_time(self.ptr.suspend_time)

    @property
    def last_sched_evaluation_time(self):
        return _raw_time(self.ptr.last_sched_eval)

    @property
    def pre_suspension_time(self):
        return _raw_time(self.ptr.pre_sus_time)

    @property
    def mcs_label(self):
        return cstr.to_unicode(self.ptr.mcs_label)

    @property
    def partition(self):
        return cstr.to_unicode(self.ptr.partition)

    @property
    def submit_host(self):
        return cstr.to_unicode(self.ptr.alloc_node)

    @property
    def submit_session_id(self):
        return u32_parse(self.ptr.alloc_sid)

    @property
    def batch_host(self):
        return cstr.to_unicode(self.ptr.batch_host)

    @property
    def num_nodes(self):
        return u32_parse(self.ptr.num_nodes)

    @property
    def max_nodes(self):
        return u32_parse(self.ptr.max_nodes)

    @property
    def allocated_nodes(self):
        return cstr.to_unicode(self.ptr.nodes)

    @property
    def required_nodes(self):
        return cstr.to_unicode(self.ptr.req_nodes)

    @property
    def excluded_nodes(self):
        return cstr.to_unicode(self.ptr.exc_nodes)

    @property
    def scheduled_nodes(self):
        return cstr.to_unicode(self.ptr.sched_nodes)

    @property
    def derived_exit_code(self):
        ec, _ = _get_exit_code(self.ptr.derived_ec)
        return ec

    @property
    def derived_exit_code_signal(self):
        _, sig = _get_exit_code(self.ptr.derived_ec)
        return sig

    @property
    def exit_code(self):
        ec, _ = _get_exit_code(self.ptr.exit_code)
        return ec

    @property
    def exit_code_signal(self):
        _, sig = _get_exit_code(self.ptr.exit_code)
        return sig

    @property
    def batch_constraints(self):
        return cstr.to_list(self.ptr.batch_features)

    @property
    def federation_origin(self):
        return cstr.to_unicode(self.ptr.fed_origin_str)

    @property
    def federation_siblings_active(self):
        return cstr.to_list(self.ptr.fed_siblings_active_str)

    @property
    def federation_siblings_viable(self):
        return cstr.to_list(self.ptr.fed_siblings_viable_str)

    @property
    def cpus(self):
        return u32_parse(self.ptr.num_cpus, on_noval=1)

    @property
    def cpus_per_task(self):
        if self.ptr.cpus_per_tres:
            return None

        return u16_parse(self.ptr.cpus_per_task, on_noval=1)

    @property
    def cpus_per_gpu(self):
        if (not self.ptr.cpus_per_tres
                or self.ptr.cpus_per_task != slurm.NO_VAL16):
            return None

        # TODO: Make a function that, given a GRES type, safely extracts its
        # value from the string.
        val = cstr.to_unicode(self.ptr.cpus_per_tres).split(":")[2]
        return u16_parse(val)

    @property
    def boards_per_node(self):
        return u16_parse(self.ptr.boards_per_node)

    @property
    def sockets_per_board(self):
        return u16_parse(self.ptr.sockets_per_board)

    @property
    def sockets_per_node(self):
        return u16_parse(self.ptr.sockets_per_node)

    @property
    def cores_per_socket(self):
        return u16_parse(self.ptr.cores_per_socket)

    @property
    def threads_per_core(self):
        return u16_parse(self.ptr.threads_per_core)

    @property
    def ntasks(self):
        return u32_parse(self.ptr.num_tasks, on_noval=1)

    @property
    def ntasks_per_node(self):
        return u16_parse(self.ptr.ntasks_per_node)

    @property
    def ntasks_per_board(self):
        return u16_parse(self.ptr.ntasks_per_board)

    @property
    def ntasks_per_socket(self):
        return u16_parse(self.ptr.ntasks_per_socket)

    @property
    def ntasks_per_core(self):
        return u16_parse(self.ptr.ntasks_per_core)

    @property
    def ntasks_per_gpu(self):
        return u16_parse(self.ptr.ntasks_per_tres)

    @property
    def delay_boot_time(self):
        return _raw_time(self.ptr.delay_boot)

    @property
    def constraints(self):
        return cstr.to_list(self.ptr.features)

    @property
    def preferred_features(self):
        return cstr.to_list(self.ptr.prefer)

    @property
    def cluster(self):
        return cstr.to_unicode(self.ptr.cluster)

    @property
    def cluster_constraints(self):
        return cstr.to_list(self.ptr.cluster_features)

    @property
    def reservation(self):
        return cstr.to_unicode(self.ptr.resv_name)

    @property
    def resource_sharing(self):
        return cstr.to_unicode(slurm_job_share_string(self.ptr.shared))

    @property
    def requires_contiguous_nodes(self):
        return u16_parse_bool(self.ptr.contiguous)

    @property
    def licenses(self):
        return cstr.to_list(self.ptr.licenses)

    @property
    def allocated_licenses(self):
        return cstr.to_list(self.ptr.licenses_allocated)

    @property
    def network(self):
        return cstr.to_unicode(self.ptr.network)

    @property
    def command(self):
        return cstr.to_unicode(self.ptr.command)

    @property
    def working_directory(self):
        return cstr.to_unicode(self.ptr.work_dir)

    @property
    def admin_comment(self):
        return cstr.to_unicode(self.ptr.admin_comment)

    @property
    def system_comment(self):
        return cstr.to_unicode(self.ptr.system_comment)

    @property
    def container(self):
        return cstr.to_unicode(self.ptr.container)

    @property
    def container_id(self):
        return cstr.to_unicode(self.ptr.container_id)

    @property
    def comment(self):
        return cstr.to_unicode(self.ptr.comment)

    @property
    def standard_input(self):
        cdef char tmp[1024]
        slurm_get_job_stdin(tmp, sizeof(tmp), self.ptr)
        return cstr.to_unicode(tmp)

    @property
    def standard_output(self):
        cdef char tmp[1024]
        slurm_get_job_stdout(tmp, sizeof(tmp), self.ptr)
        return cstr.to_unicode(tmp)

    @property
    def standard_error(self):
        cdef char tmp[1024]
        slurm_get_job_stderr(tmp, sizeof(tmp), self.ptr)
        return cstr.to_unicode(tmp)

    @property
    def required_switches(self):
        return u32_parse(self.ptr.req_switch)

    @property
    def max_wait_time_switches(self):
        return _raw_time(self.ptr.wait4switch)

    @property
    def burst_buffer(self):
        return cstr.to_unicode(self.ptr.burst_buffer)

    @property
    def burst_buffer_state(self):
        return cstr.to_unicode(self.ptr.burst_buffer_state)

    @property
    def cpu_frequency_min(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_min)

    @property
    def cpu_frequency_max(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_max)

    @property
    def cpu_frequency_governor(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_gov)

    #   @property
    #   def tres_bindings(self):
    #       """str: ?"""
    #       # TODO: Find out how it works
    #       return cstr.to_unicode(self.ptr.tres_bind)

    #   @property
    #   def tres_frequency(self):
    #       """?"""
    #       # TODO: Find out how it works
    #       return cstr.to_unicode(self.ptr.tres_freq)

    @property
    def billable_tres(self):
        if self.ptr.billable_tres == <double>slurm.NO_VAL:
            return 0.0

        return self.ptr.billable_tres

    @property
    def wckey(self):
        return cstr.to_unicode(self.ptr.wckey)

    @property
    def mail_user(self):
        return cstr.to_list(self.ptr.mail_user)

    @property
    def mail_types(self):
        return mail_type_int_to_list(self.ptr.mail_type)

    @property
    def heterogeneous_id(self):
        return u32_parse(self.ptr.het_job_id, noval=0)

    @property
    def heterogeneous_offset(self):
        return u32_parse(self.ptr.het_job_offset, noval=0)

    #   @property
    #   def hetjob_component_ids(self):
    #       """str: ?"""
    #       # TODO: Find out how to parse it in a more proper way?
    #       return cstr.to_unicode(self.ptr.het_job_id_set)

    @property
    def temporary_disk_per_node(self):
        return u32_parse(self.ptr.pn_min_tmp_disk)

    @property
    def array_id(self):
        return u32_parse(self.ptr.array_job_id)

    @property
    def array_tasks_parallel(self):
        return u32_parse(self.ptr.array_max_tasks)

    @property
    def array_task_id(self):
        return u32_parse(self.ptr.array_task_id)

    @property
    def array_tasks_waiting(self):
        task_str = cstr.to_unicode(self.ptr.array_task_str)
        if not task_str:
            return None

        if "%" in task_str:
            # We don't want this % character and everything after it
            # in here, so remove it.
            task_str = task_str[:task_str.rindex("%")]

        return task_str

    @property
    def end_time(self):
        return _raw_time(self.ptr.end_time)

    # https://github.com/SchedMD/slurm/blob/d525b6872a106d32916b33a8738f12510ec7cf04/src/api/job_info.c#L480
    cdef _calc_run_time(self):
        cdef time_t rtime
        cdef time_t etime

        if slurm.IS_JOB_PENDING(self.ptr) or not self.ptr.start_time:
            return 0
        elif slurm.IS_JOB_SUSPENDED(self.ptr):
            return self.pre_suspension_time
        else:
            if slurm.IS_JOB_RUNNING(self.ptr) or self.ptr.end_time == 0:
                etime = ctime.time(NULL)
            else:
                etime = self.ptr.end_time

            if self.ptr.suspend_time:
                rtime = <time_t>ctime.difftime(etime, self.ptr.suspend_time)
                rtime += self.ptr.pre_sus_time
            else:
                rtime = <time_t>ctime.difftime(etime, self.ptr.start_time)

        return u64_parse(rtime, on_noval=0)

    @property
    def run_time(self):
        return self._calc_run_time()

    @property
    def cores_reserved_for_system(self):
        if self.ptr.core_spec != slurm.NO_VAL16:
            if not self.ptr.core_spec & slurm.CORE_SPEC_THREAD:
                return self.ptr.core_spec

    @property
    def threads_reserved_for_system(self):
        if self.ptr.core_spec != slurm.NO_VAL16:
            if self.ptr.core_spec & slurm.CORE_SPEC_THREAD:
                return self.ptr.core_spec & (~slurm.CORE_SPEC_THREAD)

    @property
    def memory(self):
        mem_cpu = self.memory_per_cpu
        if mem_cpu is not None:
            total_cpus = self.cpus
            if total_cpus is not None:
                mem_cpu *= total_cpus
            return mem_cpu

        mem_node = self.memory_per_node
        if mem_node is not None:
            num_nodes = self.num_nodes
            if num_nodes is not None:
                mem_node *= num_nodes
            return mem_node


        # TODO
        #   mem_gpu = self.memory_per_gpu
        #   if mem_gpu is not None:
        #       num_nodes = self.min_nodes
        #       if num_nodes is not None:
        #           mem_node *= num_nodes
        #       return mem_cpu

        return None

    @property
    def memory_per_cpu(self):
        if self.ptr.pn_min_memory != slurm.NO_VAL64:
            if self.ptr.pn_min_memory & slurm.MEM_PER_CPU:
                mem = self.ptr.pn_min_memory & (~slurm.MEM_PER_CPU)
                return u64_parse(mem)
        else:
            return None

    @property
    def memory_per_node(self):
        if self.ptr.pn_min_memory != slurm.NO_VAL64:
            if not self.ptr.pn_min_memory & slurm.MEM_PER_CPU:
                return u64_parse(self.ptr.pn_min_memory)
        else:
            return None

    @property
    def memory_per_gpu(self):
        if self.ptr.mem_per_tres and self.ptr.pn_min_memory == slurm.NO_VAL64:
            # TODO: Make a function that, given a GRES type, safely extracts
            # its value from the string.
            mem = int(cstr.to_unicode(self.ptr.mem_per_tres).split(":")[2])
            return u64_parse(mem)
        else:
            return None

    @property
    def gres_per_node(self):
        # DEPRECATED - removed in the next major release
        return cstr.to_gres_dict(self.ptr.tres_per_node)

    @property
    def gpus(self):
        return {k: v for k, v in self.gres.items() if isinstance(v, GPU)}

    @property
    def gres(self):
        job_running = slurm.IS_JOB_RUNNING(self.ptr)
        tres = self.allocated_tres if job_running else self.requested_tres
        return tres.gres if tres else {}

    @property
    def tres(self):
        job_running = slurm.IS_JOB_RUNNING(self.ptr)
        return self.allocated_tres if job_running else self.requested_tres

    @property
    def tres_per_node(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_node)

    @property
    def tres_per_task(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_task)

    @property
    def tres_per_job(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_job)

    @property
    def tres_per_socket(self):
        return TrackableResources.from_cstr(self.ptr.tres_per_socket)

    @property
    def allocated_tres(self):
        return TrackableResources.from_cstr(self.ptr.tres_alloc_str)

    @property
    def requested_tres(self):
        return TrackableResources.from_cstr(self.ptr.tres_req_str)

    @property
    def profile_types(self):
        return acctg_profile_int_to_list(self.ptr.profile)

    @property
    def gres_binding(self):
        if self.ptr.bitflags & slurm.GRES_ENFORCE_BIND:
            return "enforce-binding"
        elif self.ptr.bitflags & slurm.GRES_DISABLE_BIND:
            return "disable-binding"
        else:
            return None

    @property
    def gres_tasks_per_sharing(self):
        if self.ptr.bitflags & slurm.GRES_MULT_TASKS_PER_SHARING:
            return "multiple"
        elif self.ptr.bitflags & slurm.GRES_ONE_TASK_PER_SHARING:
            return "one"
        else:
            return None

    @property
    def kill_on_invalid_dependency(self):
        return u64_parse_bool_flag(self.ptr.bitflags, slurm.KILL_INV_DEP)

    @property
    def kill_step_when_oom(self):
        return u16_parse_bool(self.ptr.oom_kill_step)

    @property
    def spreads_over_nodes(self):
        return u64_parse_bool_flag(self.ptr.bitflags, slurm.SPREAD_JOB)

    @property
    def is_cronjob(self):
        return u64_parse_bool_flag(self.ptr.bitflags, slurm.CRON_JOB)

    @property
    def cronjob_time(self):
        return cstr.to_unicode(self.ptr.cronspec)

    @property
    def elapsed_cpu_time(self):
        return self.cpus * self.run_time

    @property
    def submit_line(self):
        return cstr.to_unicode(self.ptr.submit_line)

    @property
    def run_time_remaining(self):
        limit = self.time_limit
        if limit is None:
            return None

        return (self.time_limit*60) - self.run_time

    @property
    def extra(self):
        return cstr.to_unicode(self.ptr.extra)

    @property
    def failed_node(self):
        return cstr.to_unicode(self.ptr.failed_node)

    @property
    def reserved_ports(self):
        return cstr.to_unicode(self.ptr.resv_ports)

    @property
    def selinux_context(self):
        return cstr.to_unicode(self.ptr.selinux_context)

    @property
    def segment_size(self):
        return u16_parse(self.ptr.segment_size)

    @property
    def spread_segments(self):
        return u64_parse_bool_flag(self.ptr.bitflags, slurm.SPREAD_SEGMENTS)

    @property
    def consolidate_segments(self):
        return u64_parse_bool_flag(self.ptr.bitflags, slurm.CONSOLIDATE_SEGMENTS)

    @property
    def slurm_protocol_version(self):
        return u16_parse(self.ptr.start_protocol_ver)

    def get_resource_layout_per_node(self):
        """Retrieve the resource layout of this Job on each node.

        !!! warning

            Return type may still be subject to change in the future

        Returns:
            (dict): Resource layout, where the key is the name of the node and
                the value another dict with the keys `cpu_ids`, `memory` and
                `gres`.
        """
        # The code for this function is a modified reimplementation from here:
        # https://github.com/SchedMD/slurm/blob/d525b6872a106d32916b33a8738f12510ec7cf04/src/api/job_info.c#L739
        #
        # The copyright notices for the file that contains the original code
        # is below:
        #
        # Portions Copyright (C) 2010-2017 SchedMD LLC <https://www.schedmd.com>.
        # Copyright (C) 2002-2007 The Regents of the University of California.
        # Copyright (C) 2008-2010 Lawrence Livermore National Security.
        # Produced at Lawrence Livermore National Laboratory (cf, DISCLAIMER).
        # Written by Morris Jette <jette1@llnl.gov> et. al.
        # CODE-OCEC-09-009. All rights reserved.
        #
        # Slurm is licensed under the GNU General Public License. For the full
        # text of Slurm's License, please see here:
        # pyslurm/slurm/SLURM_LICENSE
        #
        # Please, as mentioned above, also have a look at Slurm's DISCLAIMER
        # under pyslurm/slurm/SLURM_DISCLAIMER
        #
        # TODO: Explain the structure of the return value a bit more.
        cdef:
            slurm.job_resources *resources = <slurm.job_resources*>self.ptr.job_resrcs
            slurm.hostlist_t *hl
            uint32_t rel_node_inx
            int bit_inx = 0
            int bit_reps = 0
            int sock_inx = 0
            uint32_t sock_reps = 0
            int i = 0, j
            uint32_t k = 0
            char *host
            char *gres = NULL
            slurm.bitstr_t *cpu_bitmap
            char cpu_bitmap_str[128]
            uint32_t threads
            dict output = {}

        if not resources or not resources.core_bitmap:
            return output

        hl = slurm.slurm_hostlist_create(resources.nodes)
        if not hl:
            raise ValueError("Unable to create hostlist.")

        for rel_node_inx in range(resources.nhosts):
            # Check how many consecutive nodes have the same cpu allocation
            # layout.
            if sock_reps >= resources.sock_core_rep_count[sock_inx]:
                sock_inx += 1
                sock_reps = 0
            sock_reps += 1

            # Get the next node from the list of nodenames
            host = slurm.slurm_hostlist_shift(hl)

            # How many rounds we have to do in order to calculate the complete
            # cpu bitmap.
            bit_reps = (resources.sockets_per_node[sock_inx]
                        * resources.cores_per_socket[sock_inx])

            # Calculate the amount of threads per core this job has on the
            # specific host.
            threads = _threads_per_core(host)

            # Allocate a new, big enough cpu bitmap
            cpu_bitmap = slurm.slurm_bit_alloc(bit_reps * threads)

            # Calculate the cpu bitmap for this host.
            for j in range(bit_reps):
                if slurm.slurm_bit_test(resources.core_bitmap, bit_inx):
                    for k in range(threads):
                        slurm.slurm_bit_set(cpu_bitmap, (j*threads)+k)
                bit_inx += 1

            # Extract the cpu bitmap into a char *cpu_bitmap_str
            slurm.slurm_bit_fmt(cpu_bitmap_str,
                                sizeof(cpu_bitmap_str), cpu_bitmap)
            slurm.slurm_bit_free(&cpu_bitmap)

            nodename = cstr.to_unicode(host)
            cpu_ids  = cstr.to_unicode(cpu_bitmap_str)
            mem      = None

            if rel_node_inx < self.ptr.gres_detail_cnt:
                gres = self.ptr.gres_detail_str[rel_node_inx]

            if resources.memory_allocated:
                mem = u64_parse(resources.memory_allocated[rel_node_inx])

            if nodename:
                output[nodename] = {
                    "cpu_ids": cpu_ids,
                    "gres":    cstr.to_gres_dict(gres),
                    "memory":  mem,
                }

            free(host)

        slurm.slurm_hostlist_destroy(hl)
        return output


# https://github.com/SchedMD/slurm/blob/d525b6872a106d32916b33a8738f12510ec7cf04/src/api/job_info.c#L99
cdef _threads_per_core(char *host):
    # TODO
    return 1
