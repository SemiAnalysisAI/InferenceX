#########################################################################
# partition.pxd - interface to work with partitions in slurm
#########################################################################
# Copyright (C) 2023 Toni Harzendorf <toni.harzendorf@gmail.com>
# Copyright (C) 2023 PySlurm Developers
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
    list_t,
    partition_info_msg_t,
    job_defaults_t,
    delete_part_msg_t,
    partition_info_t,
    update_part_msg_t,
    slurm_free_partition_info_members,
    slurm_free_partition_info_msg,
    slurm_free_update_part_msg,
    slurm_init_part_desc_msg,
    slurm_load_partitions,
    slurm_sprint_cpu_bind_type,
    cpu_bind_type_t,
    slurm_preempt_mode_string,
    slurm_preempt_mode_num,
    slurm_create_partition,
    slurm_update_partition,
    slurm_delete_partition,
    xfree,
    try_xmalloc,
)
from pyslurm.db.util cimport (
    SlurmList,
    SlurmListItem,
)
from pyslurm.utils cimport cstr
from pyslurm.utils cimport ctime
from pyslurm.utils.ctime cimport time_t
from pyslurm.utils.uint cimport *
from pyslurm.xcollections cimport MultiClusterMap


cdef class Partitions(MultiClusterMap):
    """A [`Multi Cluster`][pyslurm.xcollections.MultiClusterMap] collection of [pyslurm.Partition][] objects.

    Args:
        partitions (Union[list[str], dict[str, pyslurm.Partition], str], optional=None):
            Partitions to initialize this collection with.

    Attributes:
        total_cpus (int):
            Total amount of CPUs the Partitions in a Collection have
        total_nodes (int):
            Total amount of Nodes the Partitions in a Collection have
    """
    cdef:
        partition_info_msg_t *info
        partition_info_t tmp_info


cdef class Partition:
    """A Slurm partition.

    ??? info "Setting Memory related attributes"

        Unless otherwise noted, all attributes in this class representing a
        memory value, like `default_memory_per_cpu`, may also be set with a
        string that contains suffixes like "K", "M", "G" or "T".

        For example:

            default_memory_per_cpu = "10G"

        This will internally be converted to 10240 (how the Slurm API expects
        it)

    Args:
        name (str, optional=None):
            Name of a Partition
        **kwargs (Any, optional=None):
            Every attribute of a Partition can be set, except for:

            * total_cpus
            * total_nodes
            * select_type_parameters

    Attributes:
        name (str):
            Name of the Partition.
        allowed_submit_nodes (list[str]):
            List of Nodes from which Jobs can be submitted to the partition.
        allowed_accounts (list[str]):
            List of Accounts which are allowed to execute Jobs
        allowed_groups (list[str]):
            List of Groups which are allowed to execute Jobs
        allowed_qos (list[str]):
            List of QoS which are allowed to execute Jobs
        alternate (str):
            Name of the alternate Partition in case a Partition is down.
        select_type_parameters (list[str]):
            List of Select type parameters for the select plugin.
        cpu_binding (str):
            Default CPU-binding for Jobs that execute in a Partition.
        default_memory_per_cpu (int):
            Default Memory per CPU for Jobs in this Partition, in Mebibytes.
            Mutually exclusive with `default_memory_per_node`.

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        default_memory_per_node (int):
            Default Memory per Node for Jobs in this Partition, in Mebibytes.
            Mutually exclusive with `default_memory_per_cpu`.

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        max_memory_per_cpu (int):
            Max Memory per CPU allowed for Jobs in this Partition, in
            Mebibytes. Mutually exclusive with `max_memory_per_node`.

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        max_memory_per_node (int):
            Max Memory per Node allowed for Jobs in this Partition, in
            Mebibytes. Mutually exclusive with `max_memory_per_cpu`

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        default_time (int):
            Default run time-limit in minutes for Jobs that don't specify one.

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        denied_qos (list[str]):
            List of QoS that cannot be used in a Partition
        denied_accounts (list[str]):
            List of Accounts that cannot use a Partition
        preemption_grace_time (int):
            Grace Time in seconds when a Job is selected for Preemption.
        default_cpus_per_gpu (int):
            Default CPUs per GPU for Jobs in this Partition
        default_memory_per_gpu (int):
            Default Memory per GPU, in Mebibytes, for Jobs in this Partition
        max_cpus_per_node (int):
            Max CPUs per Node allowed for Jobs in this Partition

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        max_cpus_per_socket (int):
            Max CPUs per Socket allowed for Jobs in this Partition

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        max_nodes (int):
            Max number of Nodes allowed for Jobs

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        min_nodes (int):
            Minimum number of Nodes that must be requested by Jobs
        max_time (int):
            Max Time-Limit in minutes that Jobs can request

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        oversubscribe (str):
            The oversubscribe mode for this Partition
        nodes (str):
            Nodes that are in a Partition
        nodesets (list[str]):
            List of Nodesets that a Partition has configured
        over_time_limit (int):
            Limit in minutes that Jobs can exceed their time-limit

            This can also return [UNLIMITED][pyslurm.constants.UNLIMITED]
        preempt_mode (str):
            Preemption Mode in a Partition
        priority_job_factor (int):
            The Priority Job Factor for a partition
        priority_tier (int):
            The priority tier for a Partition
        qos (str):
            A QoS associated with a Partition, used to extend possible limits
        total_cpus (int):
            Total number of CPUs available in a Partition
        total_nodes (int):
            Total number of nodes available in a Partition
        state (str):
            State the Partition is in
        is_default (bool):
            Whether this Partition is the default partition or not
        allow_root_jobs (bool):
            Whether Jobs by the root user are allowed
        is_user_exclusive (bool):
            Whether nodes will be exclusively allocated to users
        is_hidden (bool):
            Whether the partition is hidden or not
        least_loaded_nodes_scheduling (bool):
            Whether Least-Loaded-Nodes scheduling algorithm is used on a
            Partition
        is_root_only (bool):
            Whether only root is able to use a Partition
        requires_reservation (bool):
            Whether a reservation is required to use a Partition
        power_down_on_idle (bool):
            Whether nodes power down on idle after running jobs
    """
    cdef:
        partition_info_t *ptr
        int power_save_enabled
        slurm_conf

    cdef readonly cluster

    @staticmethod
    cdef Partition from_ptr(partition_info_t *in_ptr)
