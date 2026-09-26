#########################################################################
# job.pyx - pyslurm slurmdbd job api
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

from typing import Union, Any
from pyslurm.core.error import RPCError, PyslurmError
from pyslurm.utils.uint import *
from pyslurm import settings
from pyslurm import xcollections
from pyslurm.utils.ctime import (
    date_to_timestamp,
    timestr_to_mins,
    _raw_time,
)
from pyslurm.utils.helpers import (
    gid_to_name,
    group_to_gid,
    user_to_uid,
    uid_to_name,
    nodelist_to_range_str,
    instance_to_dict,
    _get_exit_code,
    gres_from_tres_dict,
)
from pyslurm.db.connection import _open_conn_or_error
from pyslurm.enums import SchedulerType


cdef class JobFilter:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __dealloc__(self):
        self._dealloc()

    def _dealloc(self):
        slurmdb_destroy_job_cond(self.ptr)
        self.ptr = NULL

    def _alloc(self):
        self._dealloc()
        self.ptr = <slurmdb_job_cond_t*>try_xmalloc(sizeof(slurmdb_job_cond_t))
        if not self.ptr:
            raise MemoryError("xmalloc failed for slurmdb_job_cond_t")

        self.ptr.db_flags = slurm.SLURMDB_JOB_FLAG_NOTSET
        self.ptr.flags |= slurm.JOBCOND_FLAG_NO_TRUNC

    def _parse_qos(self):
        if not self.qos:
            return None

        qos_id_list = []
        qos_data = QualitiesOfService.load()
        for user_input in self.qos:
            found = False
            for qos in qos_data.values():
                if (qos.id == user_input
                        or qos.name == user_input
                        or qos == user_input):
                    qos_id_list.append(str(qos.id))
                    found = True
                    break

            if not found:
                raise ValueError(f"QoS '{user_input}' does not exist")

        return qos_id_list

    def _parse_groups(self):
        if not self.groups:
            return None
        return list({group_to_gid(group) for group in self.groups})

    def _parse_users(self):
        if not self.users:
            return None
        return list({user_to_uid(user) for user in self.users})

    def _parse_clusters(self):
        if not self.clusters:
            # This is a requirement for some other parameters to function
            # correctly, like self.nodelist
            return [settings.LOCAL_CLUSTER]
        elif self.clusters == "all":
            return None
        else:
            return self.clusters

    def _parse_state(self):
        # TODO: implement
        return None

    def _create(self):
        self._alloc()
        cdef:
            slurmdb_job_cond_t *ptr = self.ptr
            slurm_selected_step_t *selected_step

        ptr.usage_start = date_to_timestamp(self.start_time)
        ptr.usage_end = date_to_timestamp(self.end_time)
        ptr.cpus_min = u32(self.cpus, on_noval=0)
        ptr.cpus_max = u32(self.max_cpus, on_noval=0)
        ptr.nodes_min = u32(self.nodes, on_noval=0)
        ptr.nodes_max = u32(self.max_nodes, on_noval=0)
        ptr.timelimit_min = u32(timestr_to_mins(self.timelimit), on_noval=0)
        ptr.timelimit_max = u32(timestr_to_mins(self.max_timelimit),
                                on_noval=0)
        make_char_list(&ptr.acct_list, self.accounts)
        make_char_list(&ptr.associd_list, self.association_ids)
        make_char_list(&ptr.cluster_list, self._parse_clusters())
        make_char_list(&ptr.constraint_list, self.constraints)
        make_char_list(&ptr.jobname_list, self.names)
        make_char_list(&ptr.groupid_list, self._parse_groups())
        make_char_list(&ptr.userid_list, self._parse_users())
        make_char_list(&ptr.wckey_list, self.wckeys)
        make_char_list(&ptr.partition_list, self.partitions)
        make_char_list(&ptr.qos_list, self._parse_qos())
        make_char_list(&ptr.state_list, self._parse_state())

        if self.nodelist:
            cstr.fmalloc(&ptr.used_nodes,
                         nodelist_to_range_str(self.nodelist))

        if self.truncate_time:
            ptr.flags &= ~slurm.JOBCOND_FLAG_NO_TRUNC

        if self.ids:
            # These are only allowed by the slurmdbd when specific jobs are
            # requested.
            if self.with_script and self.with_env:
                raise ValueError("with_script and with_env are mutually "
                                 "exclusive")

            if self.with_script:
                ptr.flags |= slurm.JOBCOND_FLAG_SCRIPT
            elif self.with_env:
                ptr.flags |= slurm.JOBCOND_FLAG_ENV

            ptr.step_list = slurm_list_create(slurm_destroy_selected_step)
            already_added = []
            for i in self.ids:
                job_id = u32(i)
                if job_id in already_added:
                    continue

                selected_step = NULL
                selected_step = <slurm_selected_step_t*>try_xmalloc(
                        sizeof(slurm_selected_step_t))
                if not selected_step:
                    raise MemoryError("xmalloc failed for slurm_selected_step_t")

                selected_step.array_task_id = slurm.NO_VAL
                selected_step.het_job_offset = slurm.NO_VAL
                selected_step.step_id.step_id = slurm.NO_VAL
                selected_step.step_id.job_id = job_id
                slurm_list_append(ptr.step_list, selected_step)
                already_added.append(job_id)

        # This must be at the end because it makes decisions based on some
        # conditions that might be set.
        slurmdb_job_cond_def_start_end(ptr)


# Alias
JobSearchFilter = JobFilter


cdef class Jobs(MultiClusterMap):

    def __init__(self, jobs=None):
        super().__init__(data=jobs,
                         typ="db.Jobs",
                         val_type=Job,
                         id_attr=Job.id,
                         key_type=int)
        self._reset_stats()

    @staticmethod
    def load(JobFilter db_filter=None, Connection db_connection=None):
        """Load Jobs from the Slurm Database

        Implements the slurmdb_jobs_get RPC.

        Args:
            db_filter (pyslurm.db.JobFilter):
                A search filter that the slurmdbd will apply when retrieving
                Jobs from the database.
            db_connection (pyslurm.db.Connection):
                An open database connection. By default if none is specified,
                one will be opened automatically.

        Returns:
            (pyslurm.db.Jobs): A Collection of database Jobs.

        Raises:
            (pyslurm.RPCError): When getting the Jobs from the Database was not
                successful

        Examples:
            Without a Filter the default behaviour applies, which is
            simply retrieving all Jobs from the same day:

            >>> import pyslurm
            >>> db_jobs = pyslurm.db.Jobs.load()
            >>> print(db_jobs)
            pyslurm.db.Jobs({1: pyslurm.db.Job(1), 2: pyslurm.db.Job(2)})
            >>> print(db_jobs[1])
            pyslurm.db.Job(1)

            Now with a Job Filter, so only Jobs that have specific Accounts
            are returned:

            >>> import pyslurm
            >>> accounts = ["acc1", "acc2"]
            >>> db_filter = pyslurm.db.JobFilter(accounts=accounts)
            >>> db_jobs = pyslurm.db.Jobs.load(db_filter)
        """
        cdef:
            Jobs out = Jobs()
            Job job
            JobFilter cond = db_filter
            SlurmList job_data
            SlurmListItem job_ptr
            Connection conn
            QualitiesOfService qos_data
            TrackableResources tres_data

        # Prepare SQL Filter
        if not db_filter:
            cond = JobFilter()
        cond._create()

        # Setup DB Conn
        conn = _open_conn_or_error(db_connection)

        # Fetch Job data
        job_data = SlurmList.wrap(slurmdb_jobs_get(conn.ptr, cond.ptr))
        if job_data.is_null:
            raise RPCError(msg="Failed to get Jobs from slurmdbd")

        # Fetch other necessary dependencies needed for translating some
        # attributes (i.e QoS IDs to its name)
        qos_data = QualitiesOfService.load(db_connection=conn,
                                           name_is_key=False)
        tres_data = TrackableResources.load(db_connection=conn)

        # TODO: How to handle the possibility of duplicate job ids that could
        # appear if IDs on a cluster are reset?
        for job_ptr in SlurmList.iter_and_pop(job_data):
            job = Job.from_ptr(<slurmdb_job_rec_t*>job_ptr.data)
            job.qos_data = qos_data
            job.tres_data = tres_data
            job._create_steps()
            job.stats = JobStatistics.from_steps(job.steps)

            elapsed = job.elapsed_time if job.elapsed_time else 0
            cpus = job.cpus if job.cpus else 1
            job.stats.elapsed_cpu_time = elapsed * cpus

            cluster = job.cluster
            if cluster not in out.data:
                out.data[cluster] = {}
            out[cluster][job.id] = job

            out._add_stats(job)

        return out

    def _reset_stats(self):
        self.stats = JobStatistics()
        self.cpus = 0
        self.nodes = 0
        self.memory = 0

    def _add_stats(self, job):
        self.stats.add(job.stats)
        self.cpus += job.cpus
        self.nodes += job.num_nodes
        self.memory += job.memory

    def calc_stats(self):
        """(Re)Calculate Statistics for the Job Collection."""
        self._reset_stats()
        for job in self.values():
            self._add_stats(job)

    @staticmethod
    def modify(db_filter, Job changes, db_connection=None):
        """Modify Slurm database Jobs.

        Implements the slurm_job_modify RPC.

        Args:
            db_filter (Union[pyslurm.db.JobFilter, pyslurm.db.Jobs]):
                A filter to decide which Jobs should be modified.
            changes (pyslurm.db.Job):
                Another [pyslurm.db.Job][] object that contains all the
                changes to apply. Check the `Other Parameters` of the
                [pyslurm.db.Job][] class to see which properties can be
                modified.
            db_connection (pyslurm.db.Connection):
                A Connection to the slurmdbd. By default, if no connection is
                supplied, one will automatically be created internally. This
                means that when the changes were considered successful by the
                slurmdbd, those modifications will be **automatically
                committed**.

                If you however decide to provide your own Connection instance
                (which must be already opened before), and the changes were
                successful, they will basically be in a kind of "staging
                area". By the time this function returns, the changes are not
                actually made.
                You are then responsible to decide whether the changes should
                be committed or rolled back by using the respective methods on
                the connection object. This way, you have a chance to see
                which Jobs were modified before you commit the changes.

        Returns:
            (list[int]): A list of Jobs that were modified

        Raises:
            (pyslurm.RPCError): When a failure modifying the Jobs occurred.

        Examples:
            In its simplest form, you can do something like this:

            >>> import pyslurm
            >>>
            >>> db_filter = pyslurm.db.JobFilter(ids=[9999])
            >>> changes = pyslurm.db.Job(comment="A comment for the job")
            >>> modified_jobs = pyslurm.db.Jobs.modify(db_filter, changes)
            >>> print(modified_jobs)
            [9999]

            In the above example, the changes will be automatically committed
            if successful.
            You can however also control this manually by providing your own
            connection object:

            >>> import pyslurm
            >>>
            >>> db_conn = pyslurm.db.Connection.open()
            >>> db_filter = pyslurm.db.JobFilter(ids=[9999])
            >>> changes = pyslurm.db.Job(comment="A comment for the job")
            >>> modified_jobs = pyslurm.db.Jobs.modify(
            ...             db_filter, changes, db_conn)

            Now you can first examine which Jobs have been modified:

            >>> print(modified_jobs)
            [9999]

            And then you can actually commit the changes:

            >>> db_conn.commit()

            You can also explicitly rollback these changes instead of
            committing, so they will not become active:

            >>> db_conn.rollback()
        """
        cdef:
            JobFilter cond
            Connection conn
            SlurmList response
            SlurmListItem response_ptr
            list out = []

        # Prepare SQL Filter
        if isinstance(db_filter, Jobs):
            job_ids = [job.id for job in self]
            cond = JobFilter(ids=job_ids)
        else:
            cond = <JobFilter>db_filter
        cond._create()

        # Setup DB Conn
        conn = _open_conn_or_error(db_connection)

        # Modify Jobs, get the result
        # This returns a List of char* with the Jobs ids that were
        # modified
        response = SlurmList.wrap(
                slurmdb_job_modify(conn.ptr, cond.ptr, changes.ptr))

        if not response.is_null and response.cnt:
            for response_ptr in response:
                response_str = cstr.to_unicode(<char*>response_ptr.data)
                if not response_str:
                    continue

                # The strings in the list returned above have a structure
                # like this:
                #
                # "<job_id> submitted at <timestamp>"
                #
                # We are just interested in the Job-ID, so extract it
                job_id = response_str.split(" ")[0]
                if job_id and job_id.isdigit():
                    out.append(int(job_id))

        elif not response.is_null:
            # There was no real error, but simply nothing has been modified
            raise RPCError(msg="Nothing was modified")
        else:
            # Autodetects the last slurm error
            raise RPCError()

        if not db_connection:
            # Autocommit if no connection was explicitly specified.
            conn.commit()

        return out


cdef class Job:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, job_id=0, cluster=None, **kwargs):
        self._alloc_impl()
        self.ptr.jobid = int(job_id)
        cstr.fmalloc(&self.ptr.cluster,
                     settings.LOCAL_CLUSTER if not cluster else cluster)
        self.qos_data = QualitiesOfService()
        self.steps = JobSteps()
        self.stats = JobStatistics()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __dealloc__(self):
        self._dealloc_impl()

    def _dealloc_impl(self):
        slurmdb_destroy_job_rec(self.ptr)
        self.ptr = NULL

    def _alloc_impl(self):
        if not self.ptr:
            self.ptr = slurmdb_create_job_rec()

    @staticmethod
    cdef Job from_ptr(slurmdb_job_rec_t *in_ptr):
        cdef Job wrap = Job.__new__(Job)
        wrap.ptr = in_ptr
        wrap.steps = JobSteps.__new__(JobSteps)
        return wrap

    @staticmethod
    def load(job_id, cluster=None, with_script=False, with_env=False):
        """Load the information for a specific Job from the Database.

        Args:
            job_id (int):
                ID of the Job to be loaded.
            cluster (str):
                Name of the Cluster to search in. Default is the local
                Cluster.
            with_script (bool):
                Whether the Job-Script should also be loaded. Mutually
                exclusive with `with_env`.
            with_env (bool):
                Whether the Job Environment should also be loaded. Mutually
                exclusive with `with_script`.

        Returns:
            (pyslurm.db.Job): Returns a new Database Job instance

        Raises:
            (pyslurm.RPCError): If requesting the information for the database
                Job was not successful.

        Examples:
            >>> import pyslurm
            >>> db_job = pyslurm.db.Job.load(10000)

            In the above example, attributes like `script` and `environment`
            are not populated. You must explicitly request one of them to be
            loaded:

            >>> import pyslurm
            >>> db_job = pyslurm.db.Job.load(10000, with_script=True)
            >>> print(db_job.script)
        """
        cluster = settings.LOCAL_CLUSTER if not cluster else cluster
        jfilter = JobFilter(ids=[int(job_id)], clusters=[cluster],
                            with_script=with_script, with_env=with_env)
        job = Jobs.load(jfilter).get((cluster, int(job_id)))
        if not job:
            raise RPCError(msg=f"Job {job_id} does not exist on "
                           f"Cluster {cluster}")

        # TODO: There might be multiple entries when job ids were reset.
        return job

    def _create_steps(self):
        cdef:
            JobStep step
            SlurmList step_list
            SlurmListItem step_ptr

        step_list = SlurmList.wrap(self.ptr.steps, owned=False)
        for step_ptr in SlurmList.iter_and_pop(step_list):
            step = JobStep.from_ptr(<slurmdb_step_rec_t*>step_ptr.data)
            step.tres_data = self.tres_data
            self.steps[step.id] = step

    def as_dict(self):
        return self.to_dict()

    def to_dict(self, recursive = False):
        """Convert Database Job information to a dictionary.

        Returns:
            (dict): Database Job information as dict

        Examples:
            >>> import pyslurm
            >>> myjob = pyslurm.db.Job.load(10000)
            >>> myjob_dict = myjob.to_dict()
        """
        return instance_to_dict(self, recursive)

    def __repr__(self):
        return f'pyslurm.db.{self.__class__.__name__}({self.id})'

    def modify(self, changes, db_connection=None):
        """Modify a Slurm database Job.

        Args:
            changes (pyslurm.db.Job):
                Another [pyslurm.db.Job][] object that contains all the
                changes to apply. Check the `Other Parameters` of the
                [pyslurm.db.Job][] class to see which properties can be
                modified.
            db_connection (pyslurm.db.Connection):
                A slurmdbd connection. See
                [pyslurm.db.Jobs.modify][pyslurm.db.job.Jobs.modify] for more
                info on this parameter.

        Raises:
            (pyslurm.RPCError): When modifying the Job failed.
        """
        cdef JobFilter jfilter = JobFilter(ids=[self.id])
        Jobs.modify(jfilter, changes, db_connection)

    @property
    def account(self):
        return cstr.to_unicode(self.ptr.account)

    @property
    def admin_comment(self):
        return cstr.to_unicode(self.ptr.admin_comment)

    @admin_comment.setter
    def admin_comment(self, val):
        cstr.fmalloc(&self.ptr.admin_comment, val)

    @property
    def num_nodes(self):
        val = TrackableResources.find_count_in_str(self.ptr.tres_alloc_str,
                                                   slurm.TRES_NODE)
        if val is not None:
            # Job is already running and has nodes allocated
            return val
        else:
            # Job is still pending, so we return the number of requested nodes
            # instead.
            val = TrackableResources.find_count_in_str(self.ptr.tres_req_str,
                                                       slurm.TRES_NODE)
            return val

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
    def association_id(self):
        return u32_parse(self.ptr.associd)

    @property
    def block_id(self):
        return cstr.to_unicode(self.ptr.blockid)

    @property
    def cluster(self):
        return cstr.to_unicode(self.ptr.cluster)

    @property
    def constraints(self):
        return cstr.to_list(self.ptr.constraints)

    @property
    def container(self):
        return cstr.to_list(self.ptr.container)

    @property
    def db_index(self):
        return u64_parse(self.ptr.db_index)

    @property
    def derived_exit_code(self):
        ec, _ = _get_exit_code(self.ptr.derived_ec)
        return ec

    @derived_exit_code.setter
    def derived_exit_code(self, val):
        self.ptr.derived_ec = int(val)

    @property
    def derived_exit_code_signal(self):
        _, sig = _get_exit_code(self.ptr.derived_ec)
        return sig

    @property
    def comment(self):
        return cstr.to_unicode(self.ptr.derived_es)

    @comment.setter
    def comment(self, val):
        cstr.fmalloc(&self.ptr.derived_es, val)

    @property
    def elapsed_time(self):
        return _raw_time(self.ptr.elapsed)

    @property
    def eligible_time(self):
        return _raw_time(self.ptr.eligible)

    @property
    def end_time(self):
        return _raw_time(self.ptr.end)

    @property
    def extra(self):
        return cstr.to_unicode(self.ptr.extra)

    @extra.setter
    def extra(self, val):
        cstr.fmalloc(&self.ptr.extra, val)

    @property
    def exit_code(self):
        ec, _ = _get_exit_code(self.ptr.exitcode)
        return ec

    @property
    def exit_code_signal(self):
        _, sig = _get_exit_code(self.ptr.exitcode)
        return sig

    # uint32_t flags

    @property
    def failed_node(self):
        return cstr.to_unicode(self.ptr.failed_node)

    def group_id(self):
        return u32_parse(self.ptr.gid, zero_is_noval=False)

    def group_name(self):
        return gid_to_name(self.ptr.gid)

    @property
    def heterogeneous_id(self):
        return u32_parse(self.ptr.het_job_id, noval=0)

    @property
    def heterogeneous_offset(self):
        return u32_parse(self.ptr.het_job_offset, noval=0)

    @property
    def id(self):
        return self.ptr.jobid

    @property
    def name(self):
        return cstr.to_unicode(self.ptr.jobname)

    # uint32_t lft

    @property
    def mcs_label(self):
        return cstr.to_unicode(self.ptr.mcs_label)

    @property
    def nodelist(self):
        return cstr.to_unicode(self.ptr.nodes)

    @property
    def partition(self):
        return cstr.to_unicode(self.ptr.partition)

    @property
    def priority(self):
        return u32_parse(self.ptr.priority, zero_is_noval=False)

    @property
    def qos(self):
        _qos = self.qos_data.get(self.ptr.qosid, None)
        return _qos.name if _qos else None

    # requested QOS - qos_req

    @property
    def requeue_count(self):
        return u16_parse(self.ptr.restart_cnt, on_noval=0)

    @property
    def cpus(self):
        val = TrackableResources.find_count_in_str(self.ptr.tres_alloc_str,
                                                   slurm.TRES_CPU)
        if val is not None:
            # Job is already running and has cpus allocated
            return val
        else:
            # Job is still pending, so we return the number of requested cpus
            # instead.
            return u32_parse(self.ptr.req_cpus, on_noval=1)

    @property
    def memory(self):
        val = TrackableResources.find_count_in_str(self.ptr.tres_req_str,
                                                   slurm.TRES_MEM)
        return val

    @property
    def requested_reservations(self):
        return cstr.to_list(self.ptr.resv_req)

    @property
    def reservation(self):
        return cstr.to_unicode(self.ptr.resv_name)

    @property
    def reservation_id(self):
        return u32_parse(self.ptr.resvid)

    @property
    def script(self):
        return cstr.to_unicode(self.ptr.script)

    @property
    def environment(self):
        return cstr.to_dict(self.ptr.env, delim1="\n", delim2="=")

    @property
    def start_time(self):
        return _raw_time(self.ptr.start)

    @property
    def state(self):
        return cstr.to_unicode(slurm_job_state_string(self.ptr.state))

    @property
    def state_reason(self):
        return cstr.to_unicode(slurm_job_state_reason_string
                               (self.ptr.state_reason_prev))

    @property
    def cancelled_by(self):
        return uid_to_name(self.ptr.requid)

    @property
    def submit_time(self):
        return _raw_time(self.ptr.submit)

    @property
    def submit_command(self):
        return cstr.to_unicode(self.ptr.submit_line)

    @property
    def suspended_time(self):
        return _raw_time(self.ptr.suspended)

    @property
    def system_comment(self):
        return cstr.to_unicode(self.ptr.system_comment)

    @system_comment.setter
    def system_comment(self, val):
        cstr.fmalloc(&self.ptr.system_comment, val)

    @property
    def time_limit(self):
        # TODO: Perhaps we should just find out what the actual PartitionLimit
        # is?
        return _raw_time(self.ptr.timelimit, "PartitionLimit")

    @property
    def user_id(self):
        return u32_parse(self.ptr.uid, zero_is_noval=False)

    @property
    def user_name(self):
        # There's also a ptr->user
        # https://github.com/SchedMD/slurm/blob/6365a8b7c9480c48678eeedef99864d8d3b6a6b5/src/sacct/print.c#L1946
        return uid_to_name(self.ptr.uid)

    # TODO: used gres

    @property
    def wckey(self):
        return cstr.to_unicode(self.ptr.wckey)

    @wckey.setter
    def wckey(self, val):
        cstr.fmalloc(&self.ptr.wckey, val)

    @property
    def wckey_id(self):
        return u32_parse(self.ptr.wckeyid)

#    @property
#    def wckey_id(self):
#        return u32_parse(self.ptr.wckeyid)

    @property
    def working_directory(self):
        return cstr.to_unicode(self.ptr.work_dir)

    @property
    def lineage(self):
        return cstr.to_unicode(self.ptr.lineage)

    @property
    def licenses(self):
        return cstr.to_list(self.ptr.licenses)

    cdef _get_stdio(self, char *path):
        cdef char *tmp_path = slurm.slurmdb_expand_job_stdio_fields(path, self.ptr)
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
    def segment_size(self):
        return u16_parse(self.ptr.segment_size)

    @property
    def scheduler(self):
        return SchedulerType.from_flag(self.ptr.flags, default=SchedulerType.UNKNOWN)

    @property
    def start_rpc_received(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.SLURMDB_JOB_FLAG_START_R)

    @property
    def tres(self):
        return self.allocated_tres or self.requested_tres

    @property
    def gres(self):
        return self.tres.gres if self.tres else {}

    @property
    def gpus(self):
        return {k: v for k, v in self.gres.items() if isinstance(v, GPU)}

    @property
    def allocated_tres(self):
        return TrackableResources.from_cstr(self.ptr.tres_alloc_str, self.tres_data)

    @property
    def requested_tres(self):
        return TrackableResources.from_cstr(self.ptr.tres_req_str, self.tres_data)
