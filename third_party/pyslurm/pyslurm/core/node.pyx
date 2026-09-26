#########################################################################
# node.pyx - interface to work with nodes in slurm
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
from pyslurm.utils import cstr
from pyslurm.utils import ctime
from pyslurm.utils.uint import *
from pyslurm.core.error import RPCError, verify_rpc
from pyslurm.utils.ctime import timestamp_to_date, _raw_time
from pyslurm import settings
from pyslurm import xcollections
from pyslurm.utils.helpers import (
    uid_to_name,
    gid_to_name,
    humanize,
    _getgrall_to_dict,
    _getpwall_to_dict,
    cpubind_to_num,
    instance_to_dict,
    nodelist_from_range_str,
    nodelist_to_range_str,
    gres_from_tres_dict,
)


cdef class Nodes(MultiClusterMap):

    def __dealloc__(self):
        slurm_free_node_info_msg(self.info)
        slurm_free_partition_info_msg(self.part_info)

    def __cinit__(self):
        self.info = NULL
        self.part_info = NULL

    def __init__(self, nodes=None):
        super().__init__(data=nodes,
                         typ="Nodes",
                         val_type=Node,
                         id_attr=Node.name,
                         key_type=str)

    @staticmethod
    def load(preload_passwd_info=False):
        """Load all nodes in the system.

        Args:
            preload_passwd_info (bool):
                Decides whether to query passwd and groups information from
                the system.
                Could potentially speed up access to attributes of the Node
                where a UID/GID is translated to a name.
                If True, the information will fetched and stored in each of
                the Node instances. The default is False.

        Returns:
            (pyslurm.Nodes): Collection of node objects.

        Raises:
            (pyslurm.RPCError): When getting all the Nodes from the slurmctld
                failed.
        """
        cdef:
            dict passwd = {}
            dict groups = {}
            Nodes nodes = Nodes()
            int flags = slurm.SHOW_ALL | slurm.SHOW_DETAIL
            Node node

        verify_rpc(slurm_load_node(0, &nodes.info, flags))
        verify_rpc(slurm_load_partitions(0, &nodes.part_info, flags))
        slurm_populate_node_partitions(nodes.info, nodes.part_info)

        # If requested, preload the passwd and groups database to potentially
        # speedup lookups for an attribute in a node, e.g "owner".
        if preload_passwd_info:
            passwd = _getpwall_to_dict()
            groups = _getgrall_to_dict()

        # zero-out a dummy node_info_t
        memset(&nodes.tmp_info, 0, sizeof(node_info_t))

        # Put each node pointer into its own "Node" instance.
        for cnt in range(nodes.info.record_count):
            node = Node.from_ptr(&nodes.info.node_array[cnt])

            # Prevent double free if xmalloc fails mid-loop and a MemoryError
            # is raised by replacing it with a zeroed-out node_info_t.
            nodes.info.node_array[cnt] = nodes.tmp_info

            name = node.name
            if not name:
                # Could be possible if there are nodes configured in
                # slurm.conf that cannot be reached anymore.
                continue

            if preload_passwd_info:
                node.passwd = passwd
                node.groups = groups

            cluster = node.cluster
            if cluster not in nodes.data:
                nodes.data[cluster] = {}
            nodes.data[cluster][name] = node

        # We have extracted all pointers
        nodes.info.record_count = 0
        return nodes

    def reload(self):
        """Reload the information for Nodes in a collection.

        !!! note

            Only information for nodes which are already in the collection at
            the time of calling this method will be reloaded.

        Returns:
            (pyslurm.Nodes): Returns self

        Raises:
            (pyslurm.RPCError): When getting the Nodes from the slurmctld
                failed.
        """
        return xcollections.multi_reload(self)

    def modify(self, Node changes):
        """Modify all Nodes in a collection.

        Args:
            changes (pyslurm.Node):
                Another Node object that contains all the changes to apply.
                Check the `Other Parameters` of the Node class to see which
                properties can be modified.

        Raises:
            (pyslurm.RPCError): When updating the Node was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> nodes = pyslurm.Nodes.load()
            >>> # Prepare the changes
            >>> changes = pyslurm.Node(state="DRAIN", reason="DRAIN Reason")
            >>> # Apply the changes to all the nodes
            >>> nodes.modify(changes)
        """
        cdef Node n = <Node>changes
        node_str = nodelist_to_range_str(list(self.keys()))
        n._alloc_umsg()
        cstr.fmalloc(&n.umsg.node_names, node_str)
        verify_rpc(slurm_update_node(n.umsg))

    @property
    def free_memory(self):
        return xcollections.sum_property(self, Node.free_memory)

    @property
    def real_memory(self):
        return xcollections.sum_property(self, Node.real_memory)

    @property
    def idle_memory(self):
        return xcollections.sum_property(self, Node.idle_memory)

    @property
    def allocated_memory(self):
        return xcollections.sum_property(self, Node.allocated_memory)

    @property
    def total_cpus(self):
        return xcollections.sum_property(self, Node.total_cpus)

    @property
    def idle_cpus(self):
        return xcollections.sum_property(self, Node.idle_cpus)

    @property
    def allocated_cpus(self):
        return xcollections.sum_property(self, Node.allocated_cpus)

    @property
    def effective_cpus(self):
        return xcollections.sum_property(self, Node.effective_cpus)

    @property
    def current_watts(self):
        return xcollections.sum_property(self, Node.current_watts)

    @property
    def avg_watts(self):
        return xcollections.sum_property(self, Node.avg_watts)


cdef class Node:

    def __cinit__(self):
        self.info = NULL
        self.umsg = NULL

    def __init__(self, name=None, **kwargs):
        self._alloc_impl()
        self.name = name
        self.cluster = settings.LOCAL_CLUSTER
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _alloc_impl(self):
        self._alloc_info()
        self._alloc_umsg()

    def _alloc_info(self):
        if not self.info:
            self.info = <node_info_t*>try_xmalloc(sizeof(node_info_t))
            if not self.info:
                raise MemoryError("xmalloc failed for node_info_t")

    def _alloc_umsg(self):
        if not self.umsg:
            self.umsg = <update_node_msg_t*>try_xmalloc(sizeof(update_node_msg_t))
            if not self.umsg:
                raise MemoryError("xmalloc failed for update_node_msg_t")
            slurm_init_update_node_msg(self.umsg)

    def _dealloc_impl(self):
        slurm_free_update_node_msg(self.umsg)
        self.umsg = NULL
        slurm_free_node_info_members(self.info)
        xfree(self.info)

    def __dealloc__(self):
        self._dealloc_impl()

    def __setattr__(self, name, val):
        # When a user wants to set attributes on a Node instance that was
        # created by calling Nodes(), the "umsg" pointer is not yet allocated.
        # We only allocate memory for it by the time the user actually wants
        # to modify something.
        self._alloc_umsg()
        # Call descriptors __set__ directly
        Node.__dict__[name].__set__(self, val)

    def __repr__(self):
        return f'pyslurm.{self.__class__.__name__}({self.name})'

    @staticmethod
    cdef Node from_ptr(node_info_t *in_ptr):
        cdef Node wrap = Node.__new__(Node)
        wrap._alloc_info()
        wrap.passwd = {}
        wrap.groups = {}
        wrap.cluster = settings.LOCAL_CLUSTER
        memcpy(wrap.info, in_ptr, sizeof(node_info_t))
        return wrap

    cdef _swap_data(Node dst, Node src):
        cdef node_info_t *tmp = NULL
        if dst.info and src.info:
            tmp = dst.info
            dst.info = src.info
            src.info = tmp

    @staticmethod
    def load(name):
        """Load information for a specific node.

        Implements the slurm_load_node_single RPC.

        Args:
            name (str):
                The name of the Node to load.

        Returns:
            (pyslurm.Node): Returns a new Node instance.

        Raises:
            (pyslurm.RPCError): If requesting the Node information from the
                slurmctld was not successful.

        Examples:
            >>> import pyslurm
            >>> node = pyslurm.Node.load("localhost")
        """
        cdef:
            node_info_msg_t      *node_info = NULL
            partition_info_msg_t *part_info = NULL
            Node wrap = None

        try:
            verify_rpc(slurm_load_node_single(&node_info,
                                              name, slurm.SHOW_ALL))
            verify_rpc(slurm_load_partitions(0, &part_info, slurm.SHOW_ALL))
            slurm_populate_node_partitions(node_info, part_info)

            if node_info and node_info.record_count:
                wrap = Node.from_ptr(&node_info.node_array[0])
                node_info.record_count = 0
            else:
                raise RPCError(msg=f"Node '{name}' does not exist")
        except Exception as e:
            raise e
        finally:
            slurm_free_node_info_msg(node_info)
            slurm_free_partition_info_msg(part_info)

        return wrap

    def create(self, state="future"):
        """Create a node.

        Implements the slurm_create_node RPC.

        Args:
            state (str, optional):
                An optional state the created Node should have. Allowed values
                are `future` and `cloud`. `future` is the default.

        Returns:
            (pyslurm.Node): This function returns the current Node-instance
                object itself.

        Raises:
            (pyslurm.RPCError): If creating the Node was not successful.

        Examples:
            >>> import pyslurm
            >>> node = pyslurm.Node("testnode").create()
        """
        if not self.name:
            raise ValueError("You need to set a node name first.")

        self._alloc_umsg()
        cstr.fmalloc(&self.umsg.extra,
                     f"NodeName={self.name} State={state}")
        verify_rpc(slurm_create_node(self.umsg))

        return self

    def modify(self, Node changes):
        """Modify a node.

        Implements the slurm_update_node RPC.

        Args:
            changes (pyslurm.Node):
                Another Node object that contains all the changes to apply.
                Check the `Other Parameters` of the Node class to see which
                properties can be modified.

        Raises:
            (pyslurm.RPCError): When updating the Node was not successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> mynode = pyslurm.Node.load("localhost")
            >>> # Prepare the changes
            >>> changes = pyslurm.Node(state="DRAIN", reason="DRAIN Reason")
            >>> # Modify it
            >>> mynode.modify(changes)
        """
        cdef Node n = <Node>changes
        n._alloc_umsg()
        cstr.fmalloc(&n.umsg.node_names, self.name)
        verify_rpc(slurm_update_node(n.umsg))

    def delete(self):
        """Delete a node.

        Implements the slurm_delete_node RPC.

        Raises:
            (pyslurm.RPCError): If deleting the Node was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Node("localhost").delete()
        """
        self._alloc_umsg()
        verify_rpc(slurm_delete_node(self.umsg))

    def as_dict(self):
        return self.to_dict()

    def to_dict(self, recursive = False):
        """Node information formatted as a dictionary.

        Returns:
            (dict): Node information as dict

        Examples:
            >>> import pyslurm
            >>> mynode = pyslurm.Node.load("mynode")
            >>> mynode_dict = mynode.to_dict()
        """
        return instance_to_dict(self, recursive)

    @property
    def name(self):
        return cstr.to_unicode(self.info.name)

    @name.setter
    def name(self, val):
        cstr.fmalloc2(&self.info.name, &self.umsg.node_names, val)

    @property
    def architecture(self):
        return cstr.to_unicode(self.info.arch)

    @property
    def configured_gres(self):
        return cstr.to_gres_dict(self.info.gres)

    @configured_gres.setter
    def configured_gres(self, val):
        cstr.fmalloc2(&self.info.gres, &self.umsg.gres,
                      cstr.from_gres_dict(val))

    @property
    def owner(self):
        return uid_to_name(self.info.owner, lookup=self.passwd)

    @property
    def address(self):
        return cstr.to_unicode(self.info.node_addr)

    @address.setter
    def address(self, val):
        cstr.fmalloc2(&self.info.node_addr, &self.umsg.node_addr, val)

    @property
    def hostname(self):
        return cstr.to_unicode(self.info.node_hostname)

    @hostname.setter
    def hostname(self, val):
        cstr.fmalloc2(&self.info.node_hostname, &self.umsg.node_hostname, val)

    @property
    def extra(self):
        return cstr.to_unicode(self.info.extra)

    @extra.setter
    def extra(self, val):
        cstr.fmalloc2(&self.info.extra, &self.umsg.extra, val)

    @property
    def reason(self):
        return cstr.to_unicode(self.info.reason)

    @reason.setter
    def reason(self, val):
        cstr.fmalloc2(&self.info.reason, &self.umsg.reason, val)

    @property
    def reason_user(self):
        return uid_to_name(self.info.reason_uid, lookup=self.passwd)

    @property
    def comment(self):
        return cstr.to_unicode(self.info.comment)

    @comment.setter
    def comment(self, val):
        cstr.fmalloc2(&self.info.comment, &self.umsg.comment, val)

    @property
    def bcast_address(self):
        return cstr.to_unicode(self.info.bcast_address)

    @property
    def slurm_version(self):
        return cstr.to_unicode(self.info.version)

    @property
    def operating_system(self):
        return cstr.to_unicode(self.info.os)

    @property
    def allocated_gres(self):
        return gres_from_tres_dict(self.allocated_tres)

    @property
    def mcs_label(self):
        return cstr.to_unicode(self.info.mcs_label)

    @property
    def allocated_memory(self):
        return u64_parse(self.info.alloc_memory, on_noval=0)

    @property
    def real_memory(self):
        return u64_parse(self.info.real_memory)

    @property
    def free_memory(self):
        return u64_parse(self.info.free_mem)

    @property
    def idle_memory(self):
        real = self.real_memory
        return 0 if not real else real - self.allocated_memory

    @property
    def memory_reserved_for_system(self):
        return u64_parse(self.info.mem_spec_limit)

    @property
    def temporary_disk(self):
        return u32_parse(self.info.tmp_disk)

    @property
    def weight(self):
        return u32_parse(self.info.weight)

    @weight.setter
    def weight(self, val):
        self.info.weight=self.umsg.weight = u32(val)

    @property
    def effective_cpus(self):
        return u16_parse(self.info.cpus_efctv, on_noval=0)

    @property
    def total_cpus(self):
        return u16_parse(self.info.cpus, on_noval=0)

    @property
    def sockets(self):
        return u16_parse(self.info.sockets, on_noval=0)

    @property
    def cores_reserved_for_system(self):
        return u16_parse(self.info.core_spec_cnt)

    @property
    def boards(self):
        return u16_parse(self.info.boards)

    @property
    def cores_per_socket(self):
        return u16_parse(self.info.cores)

    @property
    def threads_per_core(self):
        return u16_parse(self.info.threads)

    @property
    def available_features(self):
        return cstr.to_list(self.info.features)

    @available_features.setter
    def available_features(self, val):
        cstr.from_list2(&self.info.features, &self.umsg.features, val)

    @property
    def active_features(self):
        return cstr.to_list(self.info.features_act)

    @active_features.setter
    def active_features(self, val):
        cstr.from_list2(&self.info.features_act, &self.umsg.features_act, val)

    @property
    def partitions(self):
        return cstr.to_list(self.info.partitions)

    @property
    def boot_time(self):
        return _raw_time(self.info.boot_time)

    @property
    def slurmd_start_time(self):
        return _raw_time(self.info.slurmd_start_time)

    @property
    def last_busy_time(self):
        return _raw_time(self.info.last_busy)

    @property
    def reason_time(self):
        return _raw_time(self.info.reason_time)

#   @property
#   def tres_configured(self):
#       """dict: TRES that are configured on the node."""
#       return cstr.to_dict(self.info.tres_fmt_str)

    @property
    def allocated_tres(self):
        return cstr.to_dict(self.info.alloc_tres_fmt_str)

    @property
    def allocated_cpus(self):
        return u16_parse(self.info.alloc_cpus, on_noval=0)

    @property
    def idle_cpus(self):
        return self.effective_cpus - self.allocated_cpus

    @property
    def cpu_binding(self):
        cdef char cpu_bind[128]
        slurm_sprint_cpu_bind_type(cpu_bind,
                                   <cpu_bind_type_t>self.info.cpu_bind)
        if cpu_bind == "(null type)":
            return None

        return cstr.to_unicode(cpu_bind)

    @cpu_binding.setter
    def cpu_binding(self, val):
        self.info.cpu_bind=self.umsg.cpu_bind = cpubind_to_num(val)

    @property
    def current_watts(self):
        if not self.info.energy:
            return 0
        return u32_parse(self.info.energy.current_watts, on_noval=0)

    @property
    def avg_watts(self):
        if not self.info.energy:
            return 0
        return u32_parse(self.info.energy.ave_watts, on_noval=0)

    @property
    def _node_state(self):
        idle_cpus = self.idle_cpus
        state = self.info.node_state

        if idle_cpus and idle_cpus != self.effective_cpus:
            # If we aren't idle but also not allocated, then set state to
            # MIXED.
            state &= slurm.NODE_STATE_FLAGS
            state |= slurm.NODE_STATE_MIXED

        return state

    @property
    def state(self):
        cdef char* state = slurm_node_state_string_complete(self._node_state)
        state_str = cstr.to_unicode(state)
        xfree(state)
        return state_str

    @state.setter
    def state(self, val):
        self.umsg.node_state=self.info.node_state = _node_state_from_str(val)

    @property
    def next_state(self):
        state = self._node_state
        if ((self.info.next_state != slurm.NO_VAL)
                and (state & slurm.NODE_STATE_REBOOT_REQUESTED
                or   state & slurm.NODE_STATE_REBOOT_ISSUED)):
            return cstr.to_unicode(
                    slurm_node_state_string(self.info.next_state))
        else:
            return None

    @property
    def cpu_load(self):
        load = u32_parse(self.info.cpu_load)
        return load / 100.0 if load is not None else 0.0

    @property
    def slurmd_port(self):
        return u16_parse(self.info.port)


def _node_state_from_str(state, err_on_invalid=True):
    if not state:
        return slurm.NO_VAL
    ustate = state.upper()

    # Following states are explicitly possible as per documentation
    # https://slurm.schedmd.com/scontrol.html#OPT_State_1
    if ustate == "CANCEL_REBOOT":
        return slurm.NODE_STATE_CANCEL_REBOOT
    elif ustate == "DOWN":
        return slurm.NODE_STATE_DOWN
    elif ustate == "DRAIN":
        return slurm.NODE_STATE_DRAIN
    elif ustate == "FAIL":
        return slurm.NODE_STATE_FAIL
    elif ustate == "FUTURE":
        return slurm.NODE_STATE_FUTURE
    elif ustate == "NORESP" or ustate == "NO_RESP":
        return slurm.NODE_STATE_NO_RESPOND
    elif ustate == "POWER_DOWN":
        return slurm.NODE_STATE_POWER_DOWN
    elif ustate == "POWER_DOWN_ASAP":
        # Drain and mark for power down
        return slurm.NODE_STATE_POWER_DOWN | slurm.NODE_STATE_POWER_DRAIN
    elif ustate == "POWER_DOWN_FORCE":
        # Kill all Jobs and power down
        return slurm.NODE_STATE_POWER_DOWN | slurm.NODE_STATE_POWERED_DOWN
    elif ustate == "POWER_UP":
        return slurm.NODE_STATE_POWER_UP
    elif ustate == "RESUME":
        return slurm.NODE_RESUME
    elif ustate == "UNDRAIN":
        return slurm.NODE_STATE_UNDRAIN

    if err_on_invalid:
        raise ValueError(f"Invalid Node state: {state}")
    else:
        return slurm.NO_VAL
