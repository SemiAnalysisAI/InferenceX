#########################################################################
# node.pxd - interface to work with nodes in slurm
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

from libc.string cimport memcpy, memset
from pyslurm cimport slurm
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from pyslurm.slurm cimport (
    node_info_t,
    node_info_msg_t,
    update_node_msg_t,
    partition_info_msg_t,
    slurm_load_node,
    slurm_load_node_single,
    slurm_update_node,
    slurm_delete_node,
    slurm_create_node,
    slurm_load_partitions,
    slurm_free_update_node_msg,
    slurm_init_update_node_msg,
    slurm_populate_node_partitions,
    slurm_free_node_info_msg,
    slurm_free_node_info_members,
    slurm_free_update_node_msg,
    slurm_free_partition_info_msg,
    slurm_sprint_cpu_bind_type,
    slurm_node_state_string_complete,
    slurm_node_state_string,
    cpu_bind_type_t,
    xfree,
    try_xmalloc,
)
from pyslurm.utils cimport cstr
from pyslurm.utils cimport ctime
from pyslurm.utils.ctime cimport time_t
from pyslurm.utils.uint cimport *
from pyslurm.xcollections cimport MultiClusterMap


cdef class Nodes(MultiClusterMap):
    """A [`Multi Cluster`][pyslurm.xcollections.MultiClusterMap] collection of [pyslurm.Node][] objects.

    Args:
        nodes (Union[list[str], dict[str, pyslurm.Node], str], optional=None):
            Nodes to initialize this collection with.

    Attributes:
        free_memory (int):
            Amount of free memory in this node collection. (in Mebibytes)
            Note that this means actual free memory as returned by the `free`
            command
        real_memory (int):
            Amount of real memory in this node collection. (in Mebibytes)
        idle_memory (int):
            Amount of idle memory in this node collection. (in Mebibytes)
        allocated_memory (int):
            Amount of alloc Memory in this node collection. (in Mebibytes)
        total_cpus (int):
            Total amount of CPUs in this node collection.
        idle_cpus (int):
            Total amount of idle CPUs in this node collection.
        allocated_cpus (int):
            Total amount of allocated CPUs in this node collection.
        effective_cpus (int):
            Total amount of effective CPUs in this node collection.
        current_watts (int):
            Total amount of Watts consumed in this node collection.
        avg_watts (int):
            Amount of average watts consumed in this node collection.
    """
    cdef:
        node_info_msg_t *info
        partition_info_msg_t *part_info
        node_info_t tmp_info


cdef class Node:
    """A Slurm node.

    Args:
        name (str, optional=None):
            Name of a node

    Other Parameters:
        configured_gres (dict):
            Configured GRES for the node
        address (str):
            Address of the node
        hostname (str):
            Hostname of the node
        extra (str):
            Arbitrary extra string
        comment (str):
            Comment for the node
        weight (int):
            Weight associated to the node
        available_features (list):
            Available features for the node
        active_features (list):
            Active features for the node
        cpu_binding (str):
            Default CPU-Binding for the node
        state (str):
            State of the node
        reason (str):
            Reason for the Node, typically used along with updating the node
            state

    Attributes:
        name (str):
            Name of the node.
        architecture (str):
            Architecture of the node (e.g. x86_64)
        configured_gres (dict):
            Generic Resources this Node is configured with.
        owner (str):
            User that owns the Node.
        address (str):
            Address of the node.
        hostname (str):
            Hostname of the node.
        extra (str):
            Arbitrary string attached to the Node.
        reason (str):
            Reason why this node is in its current state.
        reason_user (str):
            Name of the User who set the reason.
        comment (str):
            Arbitrary node comment.
        bcast_address (str):
            Address of the node for sbcast.
        slurm_version (str):
            Version of slurm this node is running on.
        operating_system (str):
            Name of the operating system installed.
        allocated_gres (dict):
            Generic Resources currently in use on the node.
        mcs_label (str):
            MCS label for the node.
        allocated_memory (int):
            Memory in Mebibytes allocated on the node.
        real_memory (int):
            Real Memory in Mebibytes configured for this node.
        free_memory (int):
            Free Memory in Mebibytes on the node.
            Note that this means actual free memory as returned by the `free`
            command
        idle_memory (int):
            Idle Memory in Mebibytes on the node.
        memory_reserved_for_system (int):
            Memory in Mebibytes reserved for the System not usable by Jobs.
        temporary_disk (int):
            Amount of temporary disk space this node has, in Mebibytes.
        weight (int):
            Weight of the node in scheduling.
        effective_cpus (int):
            Number of effective CPUs the node has.
        total_cpus (int):
            Total amount of CPUs the node has.
        sockets (int):
            Number of sockets the node has.
        cores_reserved_for_system (int):
            Number of cores reserved for the System not usable by Jobs.
        boards (int):
            Number of boards the node has.
        cores_per_socket (int):
            Number of cores per socket configured for the node.
        threads_per_core (int):
            Number of threads per core configured for the node.
        available_features (list):
            List of features available on the node.
        active_features (list):
            List of features on the node.
        partitions (list):
            List of partitions this Node is part of.
        boot_time (int):
            Time the node has booted, as unix timestamp.
        slurmd_start_time (int):
            Time the slurmd has started on the Node, as unix timestamp.
        last_busy_time (int):
            Time this node was last busy, as unix timestamp.
        reason_time (int):
            Time the reason was set for the node, as unix timestamp.
        allocated_tres (dict):
            Currently allocated Trackable Resources
        allocated_cpus (int):
            Number of allocated CPUs on the node.
        idle_cpus (int):
            Number of idle CPUs.
        cpu_binding (str):
            Default CPU-Binding on the node.
        current_watts (int):
            Current amount of watts consumed on the node.
        avg_watts (int):
            Average amount of watts consumed on the node.
        state (str):
            State the node is currently in.
        next_state (str):
            Next state the node will be in.
        cpu_load (float):
            CPU Load on the Node.
        slurmd_port (int):
            Port the slurmd is listening on the node.
    """
    cdef:
        node_info_t *info
        update_node_msg_t *umsg
        dict passwd
        dict groups

    cdef readonly cluster

    @staticmethod
    cdef _swap_data(Node dst, Node src)

    @staticmethod
    cdef Node from_ptr(node_info_t *in_ptr)

