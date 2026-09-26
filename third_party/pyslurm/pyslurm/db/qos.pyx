#########################################################################
# qos.pyx - pyslurm slurmdbd qos api
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

from pyslurm.core.error import RPCError
from pyslurm.utils.helpers import instance_to_dict
from pyslurm.db.connection import _open_conn_or_error


cdef class QualitiesOfService(dict):

    def __init__(self):
        pass

    @staticmethod
    def load(QualityOfServiceFilter db_filter=None,
             Connection db_connection=None, name_is_key=True):
        """Load QoS data from the Database

        Args:
            name_is_key (bool, optional):
                By default, the keys in this dict are the names of each QoS.
                If this is set to `False`, then the unique ID of the QoS will
                be used as dict keys.
        """
        cdef:
            QualitiesOfService out = QualitiesOfService()
            QualityOfService qos
            QualityOfServiceFilter cond = db_filter
            SlurmList qos_data
            SlurmListItem qos_ptr
            Connection conn

        # Prepare SQL Filter
        if not db_filter:
            cond = QualityOfServiceFilter()
        cond._create()

        # Setup DB Conn
        conn = _open_conn_or_error(db_connection)

        # Fetch QoS Data
        qos_data = SlurmList.wrap(slurmdb_qos_get(conn.ptr, cond.ptr))

        if qos_data.is_null:
            raise RPCError(msg="Failed to get QoS data from slurmdbd")

        # Setup QOS objects
        for qos_ptr in SlurmList.iter_and_pop(qos_data):
            qos = QualityOfService.from_ptr(<slurmdb_qos_rec_t*>qos_ptr.data)
            _id = qos.name if name_is_key else qos.id
            out[_id] = qos

        return out


cdef class QualityOfServiceFilter:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __dealloc__(self):
        self._dealloc()

    def _dealloc(self):
        slurmdb_destroy_qos_cond(self.ptr)
        self.ptr = NULL

    def _alloc(self):
        self._dealloc()
        self.ptr = <slurmdb_qos_cond_t*>try_xmalloc(sizeof(slurmdb_qos_cond_t))
        if not self.ptr:
            raise MemoryError("xmalloc failed for slurmdb_qos_cond_t")

    def _parse_preempt_modes(self):
        if not self.preempt_modes:
            return 0

        if isinstance(self.preempt_modes, int):
            return self.preempt_modes

        out = 0
        for mode in self.preempt_modes:
            _mode = slurm_preempt_mode_num(mode)
            if _mode == slurm.NO_VAL16:
                raise ValueError(f"Unknown preempt mode: {mode}")

            if _mode == slurm.PREEMPT_MODE_OFF:
                _mode = slurm.PREEMPT_MODE_COND_OFF

            out |= _mode

        return out

    def _create(self):
        self._alloc()
        cdef slurmdb_qos_cond_t *ptr = self.ptr

        make_char_list(&ptr.name_list, self.names)
        make_char_list(&ptr.id_list, self.ids)
        make_char_list(&ptr.description_list, self.descriptions)
        ptr.preempt_mode = self._parse_preempt_modes()
        u16_set_bool_flag(&ptr.flags, self.with_deleted,
                          slurm.QOS_COND_FLAG_WITH_DELETED)


cdef class QualityOfService:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, name=None):
        self._alloc_impl()
        self.name = name

    def __dealloc__(self):
        self._dealloc_impl()

    def _dealloc_impl(self):
        slurmdb_destroy_qos_rec(self.ptr)
        self.ptr = NULL

    def _alloc_impl(self):
        if not self.ptr:
            self.ptr = <slurmdb_qos_rec_t*>try_xmalloc(
                    sizeof(slurmdb_qos_rec_t))
            if not self.ptr:
                raise MemoryError("xmalloc failed for slurmdb_qos_rec_t")

    @staticmethod
    cdef QualityOfService from_ptr(slurmdb_qos_rec_t *in_ptr):
        cdef QualityOfService wrap = QualityOfService.__new__(QualityOfService)
        wrap.ptr = in_ptr
        return wrap

    def __repr__(self):
        return f'pyslurm.db.{self.__class__.__name__}({self.name})'

    def to_dict(self, recursive = False):
        """Database QualityOfService information formatted as a dictionary.

        Returns:
            (dict): Database QualityOfService information as dict
        """
        return instance_to_dict(self, recursive)

    @staticmethod
    def load(name):
        """Load the information for a specific Quality of Service.

        Args:
            name (str):
                Name of the Quality of Service to be loaded.

        Returns:
            (QualityOfService): Returns a new QualityOfService
                instance.

        Raises:
            (pyslurm.RPCError): If requesting the information from the database
                was not successful.
        """
        qfilter = QualityOfServiceFilter(names=[name])
        qos = QualitiesOfService.load(qfilter).get(name)
        if not qos:
            raise RPCError(msg=f"QualityOfService {name} does not exist")

        return qos

    @property
    def name(self):
        return cstr.to_unicode(self.ptr.name)

    @name.setter
    def name(self, val):
        cstr.fmalloc(&self.ptr.name, val)

    @property
    def description(self):
        return cstr.to_unicode(self.ptr.description)

    @property
    def id(self):
        return self.ptr.id


def _qos_names_to_ids(qos_list, QualitiesOfService data):
    cdef list out = []
    if not qos_list:
        return None

    return [_validate_qos_single(qid, data) for qid in qos_list]


def _validate_qos_single(qid, QualitiesOfService data):
    for item in data.values():
        if qid == item.id or qid == item.name:
            return item.id

    raise ValueError(f"Invalid QOS specified: {qid}")


cdef _set_qos_list(list_t **in_list, vals, QualitiesOfService data):
    qos_ids = _qos_names_to_ids(vals, data)
    make_char_list(in_list, qos_ids)
