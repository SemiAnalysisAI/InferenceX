#########################################################################
# partition.pyx - interface to work with partitions in slurm
#########################################################################
# Copyright (C) 2023 Toni Harzendorf <toni.harzendorf@gmail.com>
# Copyright (C) 2025 PySlurm Developers
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
from pyslurm.utils import cstr
from pyslurm.utils import ctime
from pyslurm.utils.uint import *
from pyslurm.core.error import RPCError, verify_rpc
from pyslurm.utils.ctime import timestamp_to_date, _raw_time
from pyslurm.constants import UNLIMITED
from pyslurm import settings
from pyslurm.core import slurmctld
from pyslurm.core.slurmctld.config import _get_memory
from pyslurm import xcollections
from pyslurm.utils.helpers import (
    uid_to_name,
    gid_to_name,
    _getgrall_to_dict,
    _getpwall_to_dict,
    cpubind_to_num,
    instance_to_dict,
    dehumanize,
)
from pyslurm.utils.ctime import (
    timestr_to_mins,
    timestr_to_secs,
)


cdef class Partitions(MultiClusterMap):

    def __dealloc__(self):
        slurm_free_partition_info_msg(self.info)

    def __cinit__(self):
        self.info = NULL

    def __init__(self, partitions=None):
        super().__init__(data=partitions,
                         typ="Partitions",
                         val_type=Partition,
                         id_attr=Partition.name,
                         key_type=str)

    @staticmethod
    def load():
        """Load all Partitions in the system.

        Returns:
            (pyslurm.Partitions): Collection of Partition objects.

        Raises:
            (pyslurm.RPCError): When getting all the Partitions from the
                slurmctld failed.
        """
        cdef:
            Partitions partitions = Partitions()
            int flags = slurm.SHOW_ALL
            Partition partition
            int power_save_enabled = 0

        verify_rpc(slurm_load_partitions(0, &partitions.info, flags))
        slurm_conf = slurmctld.Config.load()

        # zero-out a dummy partition_info_t
        memset(&partitions.tmp_info, 0, sizeof(partition_info_t))

        if slurm_conf.suspend_program and slurm_conf.resume_program:
            power_save_enabled = 1

        # Put each pointer into its own instance.
        for cnt in range(partitions.info.record_count):
            partition = Partition.from_ptr(&partitions.info.partition_array[cnt])

            # Prevent double free if xmalloc fails mid-loop and a MemoryError
            # is raised by replacing it with a zeroed-out partition_info_t.
            partitions.info.partition_array[cnt] = partitions.tmp_info

            cluster = partition.cluster
            if cluster not in partitions.data:
                partitions.data[cluster] = {}

            partition.power_save_enabled = power_save_enabled
            partition.slurm_conf = slurm_conf
            partitions.data[cluster][partition.name] = partition

        # We have extracted all pointers
        partitions.info.record_count = 0
        return partitions

    def reload(self):
        """Reload the information for Partitions in a collection.

        !!! note

            Only information for Partitions which are already in the
            collection at the time of calling this method will be reloaded.

        Returns:
            (pyslurm.Partitions): Returns self

        Raises:
            (pyslurm.RPCError): When getting the Partitions from the slurmctld
                failed.
        """
        return xcollections.multi_reload(self)

    def modify(self, changes):
        """Modify all Partitions in a Collection.

        Args:
            changes (pyslurm.Partition):
                Another Partition object that contains all the changes to
                apply. Check the `Other Parameters` of the Partition class to
                see which properties can be modified.

        Raises:
            (pyslurm.RPCError): When updating at least one Partition failed.

        Examples:
            >>> import pyslurm
            >>>
            >>> parts = pyslurm.Partitions.load()
            >>> # Prepare the changes
            >>> changes = pyslurm.Partition(state="DRAIN")
            >>> # Apply the changes to all the partitions
            >>> parts.modify(changes)
        """
        for part in self.values():
            part.modify(changes)

    @property
    def total_cpus(self):
        return xcollections.sum_property(self, Partition.total_cpus)

    @property
    def total_nodes(self):
        return xcollections.sum_property(self, Partition.total_nodes)


cdef class Partition:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, name=None, **kwargs):
        self._alloc_impl()
        self.name = name
        self.cluster = settings.LOCAL_CLUSTER
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _alloc_impl(self):
        if not self.ptr:
            self.ptr = <partition_info_t*>try_xmalloc(sizeof(partition_info_t))
            if not self.ptr:
                raise MemoryError("xmalloc failed for partition_info_t")

            slurm_init_part_desc_msg(self.ptr)

    def _dealloc_impl(self):
        slurm_free_partition_info_members(self.ptr)
        xfree(self.ptr)

    def __dealloc__(self):
        self._dealloc_impl()

    def __repr__(self):
        return f'pyslurm.{self.__class__.__name__}({self.name})'

    @staticmethod
    cdef Partition from_ptr(partition_info_t *in_ptr):
        cdef Partition wrap = Partition.__new__(Partition)
        wrap._alloc_impl()
        wrap.cluster = settings.LOCAL_CLUSTER
        memcpy(wrap.ptr, in_ptr, sizeof(partition_info_t))
        return wrap

    def _error_or_name(self):
        if not self.name:
            raise ValueError("You need to set a Partition name for this "
                             "instance.")
        return self.name

    def as_dict(self):
        return self.to_dict()

    def to_dict(self, recursive = False):
        """Partition information formatted as a dictionary.

        Returns:
            (dict): Partition information as dict

        Examples:
            >>> import pyslurm
            >>> mypart = pyslurm.Partition.load("mypart")
            >>> mypart_dict = mypart.to_dict()
        """
        return instance_to_dict(self, recursive)

    @staticmethod
    def load(name):
        """Load information for a specific Partition.

        Args:
            name (str):
                The name of the Partition to load.

        Returns:
            (pyslurm.Partition): Returns a new Partition instance.

        Raises:
            (pyslurm.RPCError): If requesting the Partition information from
                the slurmctld was not successful.

        Examples:
            >>> import pyslurm
            >>> part = pyslurm.Partition.load("normal")
        """
        part = Partitions.load().get(name)
        if not part:
            raise RPCError(msg=f"Partition '{name}' doesn't exist")

        return part

    def create(self):
        """Create a Partition.

        Implements the slurm_create_partition RPC.

        Returns:
            (pyslurm.Partition): This function returns the current Partition
                instance object itself.

        Raises:
            (pyslurm.RPCError): If creating the Partition was not successful.

        Examples:
            >>> import pyslurm
            >>> part = pyslurm.Partition("debug").create()
        """
        self._error_or_name()
        verify_rpc(slurm_create_partition(self.ptr))
        return self

    def modify(self, Partition changes):
        """Modify a Partition.

        Implements the slurm_update_partition RPC.

        Args:
            changes (pyslurm.Partition):
                Another Partition object that contains all the changes to
                apply. Check the `Other Parameters` of the Partition class to
                see which properties can be modified.

        Raises:
            (pyslurm.RPCError): When updating the Partition was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> part = pyslurm.Partition.load("normal")
            >>> # Prepare the changes
            >>> changes = pyslurm.Partition(state="DRAIN")
            >>> # Apply the changes to the "normal" Partition
            >>> part.modify(changes)
        """
        cdef Partition part = <Partition>changes
        part.name = self._error_or_name()
        verify_rpc(slurm_update_partition(part.ptr))

    def delete(self):
        """Delete a Partition.

        Implements the slurm_delete_partition RPC.

        Raises:
            (pyslurm.RPCError): When deleting the Partition was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Partition("normal").delete()
        """
        cdef delete_part_msg_t del_part_msg
        memset(&del_part_msg, 0, sizeof(del_part_msg))
        del_part_msg.name = cstr.from_unicode(self._error_or_name())
        verify_rpc(slurm_delete_partition(&del_part_msg))

    # If using property getter/setter style internally becomes too messy at
    # some point, we can easily switch to normal "cdef public" attributes and
    # just extract the getter/setter logic into two functions, where one
    # creates a pointer from the instance attributes, and the other parses
    # pointer values into instance attributes.
    #
    # From a user perspective nothing would change.

    @property
    def name(self):
        return cstr.to_unicode(self.ptr.name)

    @property
    def _id(self):
        return self.name

    @name.setter
    def name(self, val):
        cstr.fmalloc(&self.ptr.name, val)

    @property
    def allowed_submit_nodes(self):
        return cstr.to_list(self.ptr.allow_alloc_nodes, ["ALL"])

    @allowed_submit_nodes.setter
    def allowed_submit_nodes(self, val):
        cstr.from_list(&self.ptr.allow_alloc_nodes, val)

    @property
    def allowed_accounts(self):
        return cstr.to_list(self.ptr.allow_accounts, ["ALL"])

    @allowed_accounts.setter
    def allowed_accounts(self, val):
        cstr.from_list(&self.ptr.allow_accounts, val)

    @property
    def allowed_groups(self):
        return cstr.to_list(self.ptr.allow_groups, ["ALL"])

    @allowed_groups.setter
    def allowed_groups(self, val):
        cstr.from_list(&self.ptr.allow_groups, val)

    @property
    def allowed_qos(self):
        return cstr.to_list(self.ptr.allow_qos, ["ALL"])

    @allowed_qos.setter
    def allowed_qos(self, val):
        cstr.from_list(&self.ptr.allow_qos, val)

    @property
    def alternate(self):
        return cstr.to_unicode(self.ptr.alternate)

    @alternate.setter
    def alternate(self, val):
        cstr.fmalloc(&self.ptr.alternate, val)

    @property
    def select_type_parameters(self):
        return _select_type_int_to_list(self.ptr.cr_type)

    @property
    def cpu_binding(self):
        cdef char cpu_bind[128]
        slurm_sprint_cpu_bind_type(cpu_bind,
                                   <cpu_bind_type_t>self.ptr.cpu_bind)
        if cpu_bind == "(null type)":
            return None

        return cstr.to_unicode(cpu_bind)

    @cpu_binding.setter
    def cpu_binding(self, val):
        self.ptr.cpu_bind = cpubind_to_num(val)

    @property
    def default_memory_per_cpu(self):
        return _get_memory(self.ptr.def_mem_per_cpu, per_cpu=True)

    @default_memory_per_cpu.setter
    def default_memory_per_cpu(self, val):
        self.ptr.def_mem_per_cpu = u64(dehumanize(val))
        self.ptr.def_mem_per_cpu |= slurm.MEM_PER_CPU

    @property
    def default_memory_per_node(self):
        return _get_memory(self.ptr.def_mem_per_cpu, per_cpu=False)

    @default_memory_per_node.setter
    def default_memory_per_node(self, val):
        self.ptr.def_mem_per_cpu = u64(dehumanize(val))

    @property
    def max_memory_per_cpu(self):
        return _get_memory(self.ptr.max_mem_per_cpu, per_cpu=True)

    @max_memory_per_cpu.setter
    def max_memory_per_cpu(self, val):
        self.ptr.max_mem_per_cpu = u64(dehumanize(val))
        self.ptr.max_mem_per_cpu |= slurm.MEM_PER_CPU

    @property
    def max_memory_per_node(self):
        return _get_memory(self.ptr.max_mem_per_cpu, per_cpu=False)

    @max_memory_per_node.setter
    def max_memory_per_node(self, val):
        self.ptr.max_mem_per_cpu = u64(dehumanize(val))

    @property
    def default_time(self):
        return _raw_time(self.ptr.default_time, on_inf=UNLIMITED)

    @default_time.setter
    def default_time(self, val):
        self.ptr.default_time = timestr_to_mins(val)

    @property
    def denied_qos(self):
        return cstr.to_list(self.ptr.deny_qos, ["ALL"])

    @denied_qos.setter
    def denied_qos(self, val):
        cstr.from_list(&self.ptr.deny_qos, val)

    @property
    def denied_accounts(self):
        return cstr.to_list(self.ptr.deny_accounts, ["ALL"])

    @denied_accounts.setter
    def denied_accounts(self, val):
        cstr.from_list(&self.ptr.deny_accounts, val)

    @property
    def preemption_grace_time(self):
        return _raw_time(self.ptr.grace_time)

    @preemption_grace_time.setter
    def preemption_grace_time(self, val):
        self.ptr.grace_time = timestr_to_secs(val)

    @property
    def default_cpus_per_gpu(self):
        def_dict = cstr.to_dict(self.ptr.job_defaults_str)
        if def_dict and "DefCpuPerGpu" in def_dict:
            return int(def_dict["DefCpuPerGpu"])

        return _extract_job_default_item(slurm.JOB_DEF_CPU_PER_GPU,
                                         self.ptr.job_defaults_list)

    @default_cpus_per_gpu.setter
    def default_cpus_per_gpu(self, val):
        _concat_job_default_str("DefCpuPerGpu", val,
                                &self.ptr.job_defaults_str)

    @property
    def default_memory_per_gpu(self):
        def_dict = cstr.to_dict(self.ptr.job_defaults_str)
        if def_dict and "DefMemPerGpu" in def_dict:
            return int(def_dict["DefMemPerGpu"])

        return _extract_job_default_item(slurm.JOB_DEF_MEM_PER_GPU,
                                         self.ptr.job_defaults_list)

    @default_memory_per_gpu.setter
    def default_memory_per_gpu(self, val):
        _concat_job_default_str("DefMemPerGpu", val,
                                &self.ptr.job_defaults_str)

    @property
    def max_cpus_per_node(self):
        return u32_parse(self.ptr.max_cpus_per_node)

    @max_cpus_per_node.setter
    def max_cpus_per_node(self, val):
        self.ptr.max_cpus_per_node = u32(val)

    @property
    def max_cpus_per_socket(self):
        return u32_parse(self.ptr.max_cpus_per_socket)

    @max_cpus_per_socket.setter
    def max_cpus_per_socket(self, val):
        self.ptr.max_cpus_per_socket = u32(val)

    @property
    def max_nodes(self):
        return u32_parse(self.ptr.max_nodes)

    @max_nodes.setter
    def max_nodes(self, val):
        self.ptr.max_nodes = u32(val)

    @property
    def min_nodes(self):
        return u32_parse(self.ptr.min_nodes, zero_is_noval=False)

    @min_nodes.setter
    def min_nodes(self, val):
        self.ptr.min_nodes = u32(val, zero_is_noval=False)

    @property
    def max_time(self):
        return _raw_time(self.ptr.max_time, on_inf=UNLIMITED)

    @max_time.setter
    def max_time(self, val):
        self.ptr.max_time = timestr_to_mins(val)

    @property
    def oversubscribe(self):
        return _oversubscribe_int_to_str(self.ptr.max_share)

    @oversubscribe.setter
    def oversubscribe(self, val):
        self.ptr.max_share = _oversubscribe_str_to_int(val)

    @property
    def nodes(self):
        return cstr.to_unicode(self.ptr.nodes)

    @nodes.setter
    def nodes(self, val):
        cstr.from_list(&self.ptr.nodes, val)

    @property
    def nodesets(self):
        return cstr.to_list(self.ptr.nodesets)

    @nodesets.setter
    def nodesets(self, val):
        cstr.from_list(&self.ptr.nodesets, val)

    @property
    def over_time_limit(self):
        return u16_parse(self.ptr.over_time_limit)

    @over_time_limit.setter
    def over_time_limit(self, val):
        self.ptr.over_time_limit = u16(self.ptr.over_time_limit)

    @property
    def preempt_mode(self):
        return _preempt_mode_int_to_str(self.ptr.preempt_mode, self.slurm_conf)

    @preempt_mode.setter
    def preempt_mode(self, val):
        self.ptr.preempt_mode = _preempt_mode_str_to_int(val)

    @property
    def priority_job_factor(self):
        return u16_parse(self.ptr.priority_job_factor)

    @priority_job_factor.setter
    def priority_job_factor(self, val):
        self.ptr.priority_job_factor = u16(val)

    @property
    def priority_tier(self):
        return u16_parse(self.ptr.priority_tier)

    @priority_tier.setter
    def priority_tier(self, val):
        self.ptr.priority_tier = u16(val)

    @property
    def qos(self):
        return cstr.to_unicode(self.ptr.qos_char)

    @qos.setter
    def qos(self, val):
        cstr.fmalloc(&self.ptr.qos_char, val)

    @property
    def total_cpus(self):
        return u32_parse(self.ptr.total_cpus, on_noval=0)

    @property
    def total_nodes(self):
        return u32_parse(self.ptr.total_nodes, on_noval=0)

    @property
    def state(self):
        return _partition_state_int_to_str(self.ptr.state_up)

    @state.setter
    def state(self, val):
        self.ptr.state_up = _partition_state_str_to_int(val)

    @property
    def is_default(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_DEFAULT)

    @is_default.setter
    def is_default(self, val):
        u32_set_bool_flag(&self.ptr.flags, val,
                          slurm.PART_FLAG_DEFAULT, slurm.PART_FLAG_DEFAULT_CLR)

    @property
    def allow_root_jobs(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_NO_ROOT)

    @allow_root_jobs.setter
    def allow_root_jobs(self, val):
        u32_set_bool_flag(&self.ptr.flags, val, slurm.PART_FLAG_NO_ROOT,
                          slurm.PART_FLAG_NO_ROOT_CLR)

    @property
    def is_user_exclusive(self):
        return u32_parse_bool_flag(self.ptr.flags,
                                   slurm.PART_FLAG_EXCLUSIVE_USER)

    @is_user_exclusive.setter
    def is_user_exclusive(self, val):
        u32_set_bool_flag(&self.ptr.flags, val, slurm.PART_FLAG_EXCLUSIVE_USER,
                          slurm.PART_FLAG_EXC_USER_CLR)

    @property
    def is_hidden(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_HIDDEN)

    @is_hidden.setter
    def is_hidden(self, val):
        u32_set_bool_flag(&self.ptr.flags, val,
                          slurm.PART_FLAG_HIDDEN, slurm.PART_FLAG_HIDDEN_CLR)

    @property
    def least_loaded_nodes_scheduling(self):
        return u16_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_LLN)

    @least_loaded_nodes_scheduling.setter
    def least_loaded_nodes_scheduling(self, val):
        u32_set_bool_flag(&self.ptr.flags, val, slurm.PART_FLAG_LLN,
                          slurm.PART_FLAG_LLN_CLR)

    @property
    def is_root_only(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_ROOT_ONLY)

    @is_root_only.setter
    def is_root_only(self, val):
        u32_set_bool_flag(&self.ptr.flags, val, slurm.PART_FLAG_ROOT_ONLY,
                          slurm.PART_FLAG_ROOT_ONLY_CLR)

    @property
    def requires_reservation(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_REQ_RESV)

    @requires_reservation.setter
    def requires_reservation(self, val):
        u32_set_bool_flag(&self.ptr.flags, val, slurm.PART_FLAG_REQ_RESV,
                          slurm.PART_FLAG_REQ_RESV_CLR)

    @property
    def power_down_on_idle(self):
        return u32_parse_bool_flag(self.ptr.flags, slurm.PART_FLAG_PDOI)

    @power_down_on_idle.setter
    def power_down_on_idle(self, val):
        u32_set_bool_flag(&self.ptr.flags, val, slurm.PART_FLAG_PDOI,
                          slurm.PART_FLAG_PDOI_CLR)

    # TODO: tres_fmt_str


def _partition_state_int_to_str(state):
    if state == slurm.PARTITION_UP:
        return "UP"
    elif state == slurm.PARTITION_DOWN:
        return "DOWN"
    elif state == slurm.PARTITION_INACTIVE:
        return "INACTIVE"
    elif state == slurm.PARTITION_DRAIN:
        return "DRAIN"
    else:
        return "UNKNOWN"


def _partition_state_str_to_int(state):
    state = state.upper()

    if state == "UP":
        return slurm.PARTITION_UP
    elif state == "DOWN":
        return slurm.PARTITION_DOWN
    elif state == "INACTIVE":
        return slurm.PARTITION_INACTIVE
    elif state == "DRAIN":
        return slurm.PARTITION_DRAIN
    else:
        choices = "UP, DOWN, INACTIVE, DRAIN"
        raise ValueError(f"Invalid partition state: {state}, valid choices "
                         f"are {choices}")


def _oversubscribe_int_to_str(shared):
    if shared == slurm.NO_VAL16:
        return None

    is_forced = shared & slurm.SHARED_FORCE
    max_jobs = shared & (~slurm.SHARED_FORCE)

    if not max_jobs:
        return "EXCLUSIVE"
    elif is_forced:
        return f"FORCE:{max_jobs}"
    elif max_jobs == 1:
        return "NO"
    else:
        return f"YES:{max_jobs}"


def _oversubscribe_str_to_int(typ):
    typ = typ.upper()

    if typ == "NO":
        return 1
    elif typ == "EXCLUSIVE":
        return 0
    elif "YES" in typ:
        return _split_oversubscribe_str(typ)
    elif "FORCE" in typ:
        return _split_oversubscribe_str(typ) | slurm.SHARED_FORCE
    else:
        return slurm.NO_VAL16


def _split_oversubscribe_str(val):
    max_jobs = val.split(":", 1)
    if len(max_jobs) == 2:
        return int(max_jobs[1])
    else:
        return 4


def _select_type_int_to_list(stype):
    # The rest of the SELECT_* stuff are just some extra parameters to the select
    # plugin
    out = _select_type_int_to_cons_res(stype)

    if stype & slurm.SELECT_ONE_TASK_PER_CORE:
        out.append("ONE_TASK_PER_CORE")

    if stype & slurm.SELECT_PACK_NODES:
        out.append("PACK_NODES")

    if stype & slurm.SELECT_CORE_DEFAULT_DIST_BLOCK:
        out.append("CORE_DEFAULT_DIST_BLOCK")

    if stype & slurm.SELECT_LLN:
        out.append("LLN")

    return out


def _select_type_int_to_cons_res(stype):
    # https://github.com/SchedMD/slurm/blob/257ca5e4756a493dc4c793ded3ac3c1a769b3c83/slurm/slurm.h#L996
    # The 3 main select types are mutually exclusive, and may be combined with
    # CR_MEMORY
    # CR_BOARD exists but doesn't show up in the documentation, so ignore it.
    if stype & slurm.SELECT_CPU and stype & slurm.SELECT_MEMORY:
        return "CPU_MEMORY"
    elif stype & slurm.SELECT_CORE and stype & slurm.SELECT_MEMORY:
        return "CORE_MEMORY"
    elif stype & slurm.SELECT_SOCKET and stype & slurm.SELECT_MEMORY:
        return "SOCKET_MEMORY"
    elif stype & slurm.SELECT_CPU:
        return "CPU"
    elif stype & slurm.SELECT_CORE:
        return "CORE"
    elif stype & slurm.SELECT_SOCKET:
        return "SOCKET"
    elif stype & slurm.SELECT_MEMORY:
        return "MEMORY"
    else:
        return []


def _preempt_mode_str_to_int(mode):
    if not mode:
        return slurm.NO_VAL16

    pmode = slurm_preempt_mode_num(str(mode))
    if pmode == slurm.NO_VAL16:
        raise ValueError(f"Invalid Preempt mode: {mode}")

    return pmode


def _preempt_mode_int_to_str(mode, slurm_conf):
    if mode == slurm.NO_VAL16:
        return slurm_conf.preempt_mode if slurm_conf else None
    else:
        return cstr.to_unicode(slurm_preempt_mode_string(mode))


cdef _extract_job_default_item(typ, list_t *job_defaults_list):
    cdef:
        job_defaults_t *default_item
        SlurmList job_def_list
        SlurmListItem job_def_item

    job_def_list = SlurmList.wrap(job_defaults_list, owned=False)
    for job_def_item in job_def_list:
        default_item = <job_defaults_t*>job_def_item.data
        if default_item.type == typ:
            return default_item.value

    return None


cdef _concat_job_default_str(typ, val, char **job_defaults_str):
    cdef uint64_t _val = u64(dehumanize(val))

    current = cstr.to_dict(job_defaults_str[0])
    if _val == slurm.NO_VAL64:
        current.pop(typ, None)
    else:
        current.update({typ : _val})

    cstr.from_dict(job_defaults_str, current)
