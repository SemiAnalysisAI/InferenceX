#########################################################################
# tres.pxd - pyslurm slurmdbd tres api
#########################################################################
# Copyright (C) 2023 Toni Harzendorf <toni.harzendorf@gmail.com>
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
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# cython: c_string_type=unicode, c_string_encoding=default
# cython: language_level=3

from pyslurm cimport slurm
from pyslurm.utils cimport cstr
from libc.stdint cimport uint64_t
from pyslurm.slurm cimport (
    slurmdb_tres_rec_t,
    slurmdb_tres_cond_t,
    slurmdb_destroy_tres_cond,
    slurmdb_init_tres_cond,
    slurmdb_destroy_tres_rec,
    slurmdb_find_tres_count_in_string,
    slurmdb_tres_get,
    try_xmalloc,
)
from pyslurm.db.util cimport (
    SlurmList,
    SlurmListItem,
)
from pyslurm.db.connection cimport Connection

cdef find_tres_count(char *tres_str, typ, on_noval=*, on_inf=*)
cdef find_tres_limit(char *tres_str, typ)
cdef merge_tres_str(char **tres_str, typ, val)
cdef _tres_ids_to_names(char *tres_str, dict tres_id_map)
cdef _set_tres_limits(char **dest, src, tres_data)


cdef class FilesystemResources(dict):
    """Collection of Filesystem TRES. This inherits from `dict`."""
    pass


cdef class GenericResources(dict):
    """Collection of Generic TRES. This inherits from `dict`."""
    pass


cdef class InterconnectResources(dict):
    """Collection of Interconnect TRES. This inherits from `dict`."""
    pass


cdef class LicenseResources(dict):
    """Collection of License TRES. This inherits from `dict`."""
    pass


cdef class OtherResources(dict):
    """Collection of Other TRES. This inherits from `dict`."""
    pass



cdef class GenericResourceLayout:

    cdef public:
        name
        type
        count
        indexes


cdef class TrackableResourceFilter:
    cdef slurmdb_tres_cond_t *ptr


cdef class TrackableResources:
    """Trackable Resources in the Slurm Database.

    Args:
        global_tres_data (Union[dict, TrackableResources], optional=None):
            This is only required when specifying TRES with their numeric ID,
            so they can be properly translated.
        **kwargs (Any, optional=None):
            Any valid attribute of the object.

    Attributes:
        cpu (pyslurm.db.TrackableResource):
            CPU TRES
        mem (pyslurm.db.TrackableResource):
            Mem TRES
        energy (pyslurm.db.TrackableResource):
            Energy TRES
        node (pyslurm.db.TrackableResource):
            Node TRES
        billing (pyslurm.db.TrackableResource):
            Billing TRES
        vmem (pyslurm.db.TrackableResource):
            VMem TRES
        pages (pyslurm.db.TrackableResource):
            Pages TRES
        fs (pyslurm.db.tres.FilesystemResources):
            Filesystem Resources
        gres (pyslurm.db.tres.GenericResources):
            Generic Resources
        license (pyslurm.db.tres.LicenseResources):
            License Resources
        interconnect (pyslurm.db.tres.InterconnectResources):
            Interconnect Resources
        other (pyslurm.db.tres.OtherResources):
            Other Resources
            Here are all Resources that are not built-in TRES-Types, when the
            Slurm source code has been modified to add a custom TRES Type
    """
    cdef public:
        dict _id_map

    cdef public:
        raw_str
        cpu
        mem
        energy
        node
        billing
        fs
        vmem
        pages
        gres
        license
        interconnect
        other

    @staticmethod
    cdef TrackableResources from_cstr(char *tres_str, global_tres_data=*, on_empty=*)

    @staticmethod
    cdef find_count_in_str(char *tres_str, typ, on_noval=*, on_inf=*)


cdef class TrackableResource:
    """A Trackable Resource in the Slurm Database.

    Args:
        tres_type (str):
            The Type of the TRES
        count (int):
            The value/count of the TRES
        name (str, optional=None):
            An optional name of the TRES.
            For example with the "fs/disk" TRES, the name would be "disk", and
            the type is "fs"
        tres_id (int, optional=None):
            The numeric ID of the TRES in the Database.

    Attributes:
        type (str):
            The Type of the TRES
        count (int):
            The value/count of the TRES
        name (str):
            An optional name of the TRES.
            For example with the "fs/disk" TRES, the name would be "disk", and
            the type is "fs"
        id (int):
            The numeric ID of the TRES in the Database.
    """
    cdef public:
        id
        name
        type
        count


    @staticmethod
    cdef TrackableResource from_ptr(slurmdb_tres_rec_t *ptr)


cdef class GPU:
    """GPU as a specific subtype for a Trackable Resource.

    Args:
        count (int):
            The value/count of the TRES
        gpu_type (str):
            A subtype for the GPU, for example "nvidia-a100"
        tres_id (int, optional=None):
            The numeric ID of the GPU TRES in the Database.

    Attributes:
        type (str):
            The subtype for the GPU
        count (int):
            The value/count of the TRES
        name (str):
            The name and subtype of the GPU. For example "gpu:nvidia-a100". If
            there is no subtype, it will just be "gpu"
        type_and_name (int):
            The same as `name`, but also include the "gres/" identifier as
            prefix. For example: "gres/gpu:nvidia-a100"
    """
    cdef public:
        id
        type
        count
