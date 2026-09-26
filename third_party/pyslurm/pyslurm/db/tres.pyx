#########################################################################
# tres.pyx - pyslurm slurmdbd tres api
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
from pyslurm.utils.uint import *
from pyslurm.constants import UNLIMITED
from pyslurm.core.error import RPCError
from pyslurm.utils.helpers import instance_to_dict, dehumanize
from pyslurm.utils import cstr
from pyslurm.db.connection import _open_conn_or_error
from pyslurm import xcollections
import json
import re


TRES_TYPE_DELIM = "/"
GRES_TYPE_DELIM = "gres:"
GRES_TYPE_DELIM_2 = "gres/"

TRES_NAME_REQUIRED = ["fs", "license", "interconnect", "gres"]

gres_pattern = re.compile(r'[/:]')


cdef class FilesystemResources(dict):

    def to_dict(self, recursive=False):
        return xcollections.dict_recursive(self, recursive)


cdef class GenericResources(dict):

    def to_dict(self, recursive=False):
        return xcollections.dict_recursive(self, recursive)


cdef class InterconnectResources(dict):

    def to_dict(self, recursive=False):
        return xcollections.dict_recursive(self, recursive)


cdef class LicenseResources(dict):

    def to_dict(self, recursive=False):
        return xcollections.dict_recursive(self, recursive)


cdef class OtherResources(dict):

    def to_dict(self, recursive=False):
        return xcollections.dict_recursive(self, recursive)


cdef class GenericResourceLayout:

    def __init__(self, name, typ=None, count=1, indexes=None):
        self.name = name
        self.type = typ
        self.count = int(count)
        self.indexes = [] if not indexes else indexes

    def to_dict(self, recursive=False):
        return instance_to_dict(self, recursive)

    @staticmethod
    def from_str(str gres_str):
        cdef:
            dict output = {}
            GenericResourceLayout gres = None

        if not gres_str or gres_str == "(null)":
            return {}

        for item in re.split(",(?=[^,]+?:)", gres_str):

            # char *gres might contain just "gres:gpu", without any count.
            # If not given, the count is always 1, so default to it.
            cnt = typ = "1"

            # Remove the additional "gres" specifier if it exists
            if GRES_TYPE_DELIM in item:
                item = item.replace(GRES_TYPE_DELIM, "")
            elif GRES_TYPE_DELIM_2 in item:
                item = item.replace(GRES_TYPE_DELIM_2, "")

            gres_splitted = re.split(
                ":(?=[^:]+?)",
                item.replace("(", ":", 1).replace(")", "")
            )
            gres_splitted_len = len(gres_splitted)

            name = gres_splitted[0]
            if gres_splitted_len > 1:
                typ = gres_splitted[1]

            # Check if we have a gres type.
            if typ.isdigit():
                cnt = typ
                typ = None
            elif gres_splitted_len > 2:
                cnt = gres_splitted[2]
            else:
                # String is somehow malformed, should never happen when the input
                # comes from the slurmctld. Ignore if it happens.
                continue

            # Dict Key-Name depends on if we have a gres type or not
            name_and_typ = f"{name}:{typ}" if typ else name

            gres = GenericResourceLayout(name, typ, cnt)

            if "IDX" in gres_splitted:
                # Cover cases with IDX
                indexes = gres_splitted[3] if not typ else gres_splitted[4]
                gres.indexes = [int(idx) for idx in indexes.split(",")]

            output[name_and_typ] = gres

        return output


cdef class TrackableResourceFilter:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __dealloc__(self):
        self._dealloc()

    def _dealloc(self):
        slurmdb_destroy_tres_cond(self.ptr)
        self.ptr = NULL

    def _alloc(self):
        self._dealloc()
        self.ptr = <slurmdb_tres_cond_t*>try_xmalloc(sizeof(slurmdb_tres_cond_t))
        if not self.ptr:
            raise MemoryError("xmalloc failed for slurmdb_tres_cond_t")
        slurmdb_init_tres_cond(self.ptr, 0)

    def _create(self):
        self._alloc()


cdef class TrackableResources:

    def __init__(self, global_tres_data=None, **kwargs):
        self._setup_defaults()
        self._init_from_dict(kwargs, global_tres_data)

    def _setup_defaults(self):
        self.fs = FilesystemResources()
        self.gres = GenericResources()
        self.license = LicenseResources()
        self.interconnect = InterconnectResources()
        self.other = OtherResources()
        self._id_map = {}

    def _init_from_dict(self, tres_dict, global_tres_data):
        if not tres_dict:
            return None

        for k, v in tres_dict.items():
            typ, name, tid = k, None, None

            if k.isdigit() and global_tres_data:
                tres_id_map = global_tres_data
                if isinstance(global_tres_data, TrackableResources):
                    tres_id_map = global_tres_data._id_map

                global_tres = tres_id_map.get(int(k))
                if not global_tres:
                    continue

                typ, name, tid = global_tres.type, global_tres.name, global_tres.id
            elif TRES_TYPE_DELIM in k:
                typ, name = self._unflatten_tres(k)

            if isinstance(v, dict):
                flattened = self._flatten_tres(k, v)
                self._init_from_dict(flattened, global_tres_data)
                continue

            # Make sure we actually have a valid value and type
            tres = TrackableResource.from_type_and_count(typ, v)
            if not tres:
                continue

            tres.name = name
            tres.id = tid
            self._handle_tres_type(tres)

    @staticmethod
    def from_str(tres_str, global_tres_data=None, on_empty=None):
        if not tres_str:
            return on_empty

        return TrackableResources.from_cstr(tres_str, global_tres_data, on_empty)

    @staticmethod
    cdef TrackableResources from_cstr(char *tres_str, global_tres_data=None, on_empty=None):
        cdef:
            TrackableResources out = TrackableResources.__new__(TrackableResources)
            dict tres_dict = cstr.to_dict(tres_str)

        if not tres_dict:
            return on_empty

        out._setup_defaults()
        out._init_from_dict(tres_dict, global_tres_data)
        return out

    def _validate(self, tres_data):
        id_dict = _tres_names_to_ids(self.to_dict(flatten_limits=True),
                                    tres_data)
        return id_dict

    def _unflatten_tres(self, type_and_name):
        typ, name = type_and_name.split(TRES_TYPE_DELIM, 1)
        return typ, name

    def _flatten_tres(self, typ, vals):
        cdef dict out = {}
        for name, val in vals.items():
            typ = typ or name
            full_name = f"{typ}{TRES_TYPE_DELIM}{name}"
            out[full_name] = val

        return out

    def to_dict(self, recursive=False, flatten_limits=False):
        cdef dict inst_dict = instance_to_dict(self, recursive)

        if flatten_limits:
            vals = inst_dict.pop("fs")
            inst_dict.update(self._flatten_tres("fs", vals))

            vals = inst_dict.pop("license")
            inst_dict.update(self._flatten_tres("license", vals))

            vals = inst_dict.pop("gres")
            inst_dict.update(self._flatten_tres("gres", vals))

            vals = inst_dict.pop("interconnect")
            inst_dict.update(self._flatten_tres("interconnect", vals))

            vals = inst_dict.pop("other")
            inst_dict.update(self._flatten_tres(None, vals))

        return inst_dict

    def _handle_tres_type(self, tres):
        if tres.type == "gres":
            gres = tres
            if tres.name == "gpu" or tres.name.startswith("gpu:"):
                gres = GPU.from_tres(tres)
            self.gres[gres.name] = gres
        elif isinstance(tres, GPU):
            self.gres[tres.name] = tres
        elif tres.type == "fs":
            self.fs[tres.name] = tres
        elif tres.type == "license":
            self.license[tres.name] = tres
        elif tres.type == "interconnect":
            self.interconnect[tres.name] = tres
        elif hasattr(self, tres.type):
            setattr(self, tres.type, tres)
        elif tres.type:
            print(tres.type, tres.name, tres.type_and_name)
            self.other[tres.type_and_name] = tres

    @staticmethod
    def load(Connection db_connection=None):
        """Load Trackable Resources from the Database."""
        cdef:
            TrackableResources out = TrackableResources()
            TrackableResource tres
            Connection conn
            SlurmList tres_data
            SlurmListItem tres_ptr
            TrackableResourceFilter db_filter = TrackableResourceFilter()

        # Prepare SQL Filter
        db_filter._create()

        # Setup DB Conn
        conn = _open_conn_or_error(db_connection)

        # Fetch TRES data
        tres_data = SlurmList.wrap(slurmdb_tres_get(conn.ptr, db_filter.ptr))

        if tres_data.is_null:
            raise RPCError(msg="Failed to get TRES data from slurmdbd")

        # Setup TRES objects
        for tres_ptr in SlurmList.iter_and_pop(tres_data):
            tres = TrackableResource.from_ptr(
                    <slurmdb_tres_rec_t*>tres_ptr.data)
            out._handle_tres_type(tres)
            out._id_map[tres.id] = tres

        return out

    @staticmethod
    cdef find_count_in_str(char *tres_str, typ, on_noval=0, on_inf=0):
        return find_tres_count(tres_str, typ, on_noval, on_inf)


cdef class TrackableResource:

    def __init__(self, tres_type, count, name=None, tres_id=None):
        self.type = tres_type
        self.id = tres_id
        self.name = name
        self.count = int(count)
        # rec_count
        # alloc_secs

    @staticmethod
    cdef TrackableResource from_ptr(slurmdb_tres_rec_t *ptr):
        return TrackableResource(
            tres_type = cstr.to_unicode(ptr.type),
            count = u64_parse(ptr.count, on_noval=1),
            name = cstr.to_unicode(ptr.name),
            tres_id = int(ptr.id),
        )

    @property
    def type_and_name(self):
        type_and_name = self.type
        if self.name:
            type_and_name = f"{type_and_name}{TRES_TYPE_DELIM}{self.name}"

        return type_and_name

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)

    @staticmethod
    def from_type_and_count(tres_type, count):
        if isinstance(tres_type, str) and tres_type.isdigit():
            return None

        if tres_type == "mem":
            count = dehumanize(count)
        else:
            count = int(count)

        if count == slurm.NO_VAL64:
            return None

        return TrackableResource(tres_type=tres_type, count=count)


cdef class GPU:

    def __init__(self, count, gpu_type=None, tres_id=None):
        self.count = int(count)
        self.type = gpu_type
        self.id = tres_id

    def to_dict(self, recursive = False):
        return instance_to_dict(self, recursive)

    @staticmethod
    def from_tres(tres):
        if ":" in tres.name:
            _, typ = tres.name.split(":")
        else:
            typ = None

        return GPU(
            count = tres.count,
            gpu_type = typ,
            tres_id = tres.id,
        )

    @property
    def name(self):
        return f"gpu:{self.type}" if self.type else "gpu"

    @property
    def type_and_name(self):
        return f"gres{TRES_TYPE_DELIM}{self.name}"


cdef find_tres_count(char *tres_str, typ, on_noval=0, on_inf=0):
    if not tres_str:
        return on_noval

    cdef uint64_t tmp
    tmp = slurmdb_find_tres_count_in_string(tres_str, typ)
    if tmp == slurm.INFINITE64:
        return on_inf
    elif tmp == slurm.NO_VAL64:
        return on_noval
    else:
        return tmp


cdef find_tres_limit(char *tres_str, typ):
    return find_tres_count(tres_str, typ, on_noval=None, on_inf=UNLIMITED)


cdef merge_tres_str(char **tres_str, typ, val):
    cdef uint64_t _val = u64(dehumanize(val))

    current = cstr.to_dict(tres_str[0])
    if _val == slurm.NO_VAL64:
        current.pop(typ, None)
    else:
        current.update({typ : _val})

    cstr.from_dict(tres_str, current)


cdef _tres_ids_to_names(char *tres_str, dict tres_id_map):
    if not tres_str:
        return None

    cdef:
        dict tdict = cstr.to_dict(tres_str)
        list out = []

    if not tres_id_map:
        return None

    for tid, cnt in tdict.items():
        if isinstance(tid, str) and tid.isdigit():
            _tid = int(tid)
            if _tid in tres_id_map:
                out.append(
                    (tres_id_map[_tid].type, tres_id_map[_tid].name, int(cnt))
                )

    return out


def _tres_names_to_ids(dict tres_dict, tres_data):
    cdef dict out = {}
    if not tres_dict:
        return out

    tres_id_map = tres_data
    if isinstance(tres_data, TrackableResources):
        tres_id_map = tres_data._id_map

    for tres in tres_dict.values():
        if not tres:
            continue

        real_id = _validate_tres_single(tres, tres_id_map)
        out[real_id] = tres.count

    return out


def _validate_tres_single(local_tres, dict tres_id_map):
    for global_tres in tres_id_map.values():
        if local_tres.id == global_tres.id or local_tres.type_and_name == global_tres.type_and_name:
            return global_tres.id

    raise ValueError(f"Invalid TRES specified: {local_tres.type_and_name}")


cdef _set_tres_limits(char **dest, src, tres_data):
    cstr.from_dict(dest, src._validate(tres_data))
