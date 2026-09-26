#########################################################################
# reservation.pyx - interface to work with reservations in slurm
#########################################################################
# Copyright (C) 2025 Toni Harzendorf <toni.harzendorf@gmail.com>
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
from pyslurm.utils.uint import u32_parse
from pyslurm import settings
from pyslurm.core.slurmctld.config import _get_memory
from datetime import datetime
from pyslurm import xcollections
from pyslurm.utils.helpers import instance_to_dict
from pyslurm.utils.enums import SlurmEnum, SlurmFlag
from enum import auto
from pyslurm.utils.ctime import (
    _raw_time,
    timestr_to_mins,
    timestr_to_secs,
    date_to_timestamp,
)
from pyslurm.core.error import (
    RPCError,
    verify_rpc,
    slurm_errno,
)


cdef class Reservations(MultiClusterMap):

    def __dealloc__(self):
        slurm_free_reservation_info_msg(self.info)
        self.info = NULL

    def __cinit__(self):
        self.info = NULL

    def __init__(self, reservations=None):
        super().__init__(data=reservations,
                         typ="Reservations",
                         val_type=Reservation,
                         id_attr=Reservation.name,
                         key_type=str)

    @staticmethod
    def load():
        """Load all Reservations in the system.

        Returns:
            (pyslurm.Reservations): Collection of [pyslurm.Reservation][]
                objects.

        Raises:
            (pyslurm.RPCError): When getting all the Reservations from the
                slurmctld failed.
        """
        cdef:
            Reservations reservations = Reservations()
            Reservation reservation

        verify_rpc(slurm_load_reservations(0, &reservations.info))

        memset(&reservations.tmp_info, 0, sizeof(reserve_info_t))
        for cnt in range(reservations.info.record_count):
            reservation = Reservation.from_ptr(&reservations.info.reservation_array[cnt])

            # If we already parsed at least one Reservation, and if for some
            # reason a MemoryError is raised after parsing subsequent
            # reservations, invalid behaviour will be shown by Valgrind, since
            # the Memory for the already parsed Reservation will be freed
            # twice. So for all successfully parsed Reservations, replace it
            # with a dummy struct that will be skipped in case of error.
            reservations.info.reservation_array[cnt] = reservations.tmp_info

            cluster = reservation.cluster
            if cluster not in reservations.data:
                reservations.data[cluster] = {}

            reservations.data[cluster][reservation.name] = reservation

        reservations.info.record_count = 0
        return reservations


cdef class Reservation:

    def __cinit__(self):
        self.info = NULL
        self.umsg = NULL

    def __init__(self, name=None, **kwargs):
        self._alloc_impl()
        self.name = name
        self._reoccurrence = ReservationReoccurrence.NO
        self.cluster = settings.LOCAL_CLUSTER
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _alloc_impl(self):
        self._alloc_info()
        self._alloc_umsg()

    def _alloc_info(self):
        if not self.info:
            self.info = <reserve_info_t*>try_xmalloc(sizeof(reserve_info_t))
            if not self.info:
                raise MemoryError("xmalloc failed for reserve_info_t")

    def _alloc_umsg(self):
        if not self.umsg:
            self.umsg = <resv_desc_msg_t*>try_xmalloc(sizeof(resv_desc_msg_t))
            if not self.umsg:
                raise MemoryError("xmalloc failed for resv_desc_msg_t")
            slurm_init_resv_desc_msg(self.umsg)
            self.umsg.flags = 0

    def _dealloc_umsg(self):
        slurm_free_resv_desc_msg(self.umsg)
        self.umsg = NULL

    def _dealloc_impl(self):
        self._dealloc_umsg()
        slurm_free_reserve_info_members(self.info)
        xfree(self.info)
        self.info = NULL

    def __dealloc__(self):
        self._dealloc_impl()

    def __setattr__(self, name, val):
        # When a user wants to set attributes on a Reservation instance that
        # was created by calling Reservations(), the "umsg" pointer is not yet
        # allocated. We only allocate memory for it by the time the user
        # actually wants to modify something.
        self._alloc_umsg()
        # Call descriptors __set__ directly
        Reservation.__dict__[name].__set__(self, val)

    def __repr__(self):
        return f'pyslurm.{self.__class__.__name__}({self.name})'

    @staticmethod
    cdef Reservation from_ptr(reserve_info_t *in_ptr):
        cdef Reservation wrap = Reservation.__new__(Reservation)
        wrap._alloc_info()
        wrap.cluster = settings.LOCAL_CLUSTER
        memcpy(wrap.info, in_ptr, sizeof(reserve_info_t))
        wrap._reoccurrence = ReservationReoccurrence.from_flag(wrap.info.flags,
                                                    default=ReservationReoccurrence.NO)
        wrap.info.flags &= ~wrap.reoccurrence._flag
        return wrap

    def _error_or_name(self):
        if not self.name:
            raise RPCError(msg="No Reservation name was specified. "
                           "Did you set the `name` attribute on the "
                           "Reservation instance?")
        return self.name

    def to_dict(self, recursive = False):
        """Reservation information formatted as a dictionary.

        Returns:
            (dict): Reservation information as dict

        Examples:
            >>> import pyslurm
            >>> resv = pyslurm.Reservation.load("maintenance")
            >>> resv_dict = resv.to_dict()
            >>> print(resv_dict)
        """
        return instance_to_dict(self, recursive)

    @staticmethod
    def load(name):
        """Load information for a specific Reservation.

        Args:
            name (str):
                The name of the Reservation to load.

        Returns:
            (pyslurm.Reservation): Returns a new Reservation instance.

        Raises:
            (pyslurm.RPCError): If requesting the Reservation information from
                the slurmctld was not successful.

        Examples:
            >>> import pyslurm
            >>> reservation = pyslurm.Reservation.load("maintenance")
        """
        resv = Reservations.load().get(name)
        if not resv:
            raise RPCError(msg=f"Reservation '{name}' doesn't exist")

        return resv

    def create(self):
        """Create a Reservation.

        If you did not specify at least a `start_time` and `duration` or
        `end_time`, then by default the Reservation will start effective
        immediately, with a duration of one year.

        Returns:
            (pyslurm.Reservation): This function returns the current
                Reservation instance object itself.

        Raises:
            (pyslurm.RPCError): If creating the Reservation was not successful.

        Examples:
            >>> import pyslurm
            >>> from pyslurm import ReservationFlags, ReservationReoccurrence
            >>> resv = pyslurm.Reservation(
            ...     name = "debug",
            ...     users = ["root"],
            ...     nodes = "node001",
            ...     duration = "1-00:00:00",
            ...     flags = ReservationFlags.MAINTENANCE | ReservationFlags.FLEX,
            ...     reoccurrence = ReservationReoccurrence.DAILY,
            ... )
            >>> resv.create()
        """
        cdef char* new_name = NULL

        if not self.start_time or not (self.duration and self.end_time):
            raise RPCError(msg="You must at least specify a start_time, "
                           " combined with an end_time or a duration.")

        self.name = self._error_or_name()
        new_name = slurm_create_reservation(self.umsg)
        free(new_name)
        verify_rpc(slurm_errno())
        return self

    def modify(self, Reservation changes=None):
        """Modify a Reservation.

        Args:
            changes (pyslurm.Reservation, optional=None):
                Another Reservation object that contains all the changes to
                apply. This is optional - you can also directly modify a
                Reservation object and just call `modify()`, and the changes
                will be sent to `slurmctld`.

        Raises:
            (pyslurm.RPCError): When updating the Reservation was not
                successful.

        Examples:
            >>> import pyslurm
            >>>
            >>> resv = pyslurm.Reservation.load("maintenance")
            >>> # Add 60 Minutes to the reservation
            >>> resv.duration += 60
            >>>
            >>> # You can also add a slurm timestring.
            >>> # For example, extend the duration by another day:
            >>> resv.duration += pyslurm.utils.timestr_to_mins("1-00:00:00")
            >>>
            >>> # Now send the changes to the Controller:
            >>> resv.modify()
        """
        cdef Reservation updates = changes if changes is not None else self
        if not updates.umsg:
            return

        self._error_or_name()
        cstr.fmalloc(&updates.umsg.name, self.info.name)
        verify_rpc(slurm_update_reservation(updates.umsg))

        # Make sure we clean the object from any previous changes.
        updates._dealloc_umsg()

    def delete(self):
        """Delete a Reservation.

        Raises:
            (pyslurm.RPCError): If deleting the Reservation was not successful.

        Examples:
            >>> import pyslurm
            >>> pyslurm.Reservation("maintenance").delete()
        """
        cdef reservation_name_msg_t to_delete
        memset(&to_delete, 0, sizeof(to_delete))
        to_delete.name = cstr.from_unicode(self._error_or_name())
        verify_rpc(slurm_delete_reservation(&to_delete))

    @property
    def accounts(self):
        return cstr.to_list(self.info.accounts)

    @accounts.setter
    def accounts(self, val):
        cstr.from_list2(&self.info.accounts, &self.umsg.accounts, val)

    @property
    def burst_buffer(self):
        return cstr.to_unicode(self.info.burst_buffer)

    @burst_buffer.setter
    def burst_buffer(self, val):
        cstr.fmalloc2(&self.info.burst_buffer, &self.umsg.burst_buffer, val)

    @property
    def comment(self):
        return cstr.to_unicode(self.info.comment)

    @comment.setter
    def comment(self, val):
        cstr.fmalloc2(&self.info.comment, &self.umsg.comment, val)

    @property
    def cpus(self):
        return u32_parse(self.info.core_cnt, zero_is_noval=False)

    @cpus.setter
    def cpus(self, val):
        self.info.core_cnt = self.umsg.core_cnt = int(val)

    @property
    def cpu_ids_by_node(self):
        out = {}
        for i in range(self.info.core_spec_cnt):
            node = cstr.to_unicode(self.info.core_spec[i].node_name)
            if node:
                out[node] = cstr.to_unicode(self.info.core_spec[i].core_id)

        return out

    @property
    def end_time(self):
        return _raw_time(self.info.end_time)

    @end_time.setter
    def end_time(self, val):
        val = date_to_timestamp(val)
        if self.start_time and val < self.info.start_time:
            raise ValueError("end_time cannot be earlier then start_time.")

        self.info.end_time = self.umsg.end_time = val

    @property
    def features(self):
        return cstr.to_list(self.info.features)

    @features.setter
    def features(self, val):
        cstr.from_list2(&self.info.features, &self.umsg.features, val)

    @property
    def groups(self):
        return cstr.to_list(self.info.groups)

    @groups.setter
    def groups(self, val):
        cstr.from_list2(&self.info.groups, &self.umsg.groups, val)

    @property
    def licenses(self):
        return cstr.to_list(self.info.licenses)

    @licenses.setter
    def licenses(self, val):
        cstr.from_list2(&self.info.licenses, &self.umsg.licenses, val)

    @property
    def max_start_delay(self):
        return u32_parse(self.info.max_start_delay)

    @max_start_delay.setter
    def max_start_delay(self, val):
        self.info.max_start_delay = self.umsg.max_start_delay = int(val)

    @property
    def name(self):
        return cstr.to_unicode(self.info.name)

    @name.setter
    def name(self, val):
        cstr.fmalloc2(&self.info.name, &self.umsg.name, val)

    @property
    def node_count(self):
        return u32_parse(self.info.node_cnt, zero_is_noval=False)

    @node_count.setter
    def node_count(self, val):
        self.info.node_cnt = self.umsg.node_cnt = int(val)

    @property
    def nodes(self):
        return cstr.to_unicode(self.info.node_list)

    @nodes.setter
    def nodes(self, val):
        cstr.fmalloc2(&self.info.node_list, &self.umsg.node_list, val)

    @property
    def partition(self):
        return cstr.to_unicode(self.info.partition)

    @partition.setter
    def partition(self, val):
        cstr.fmalloc2(&self.info.partition, &self.umsg.partition, val)

    @property
    def purge_time(self):
        return u32_parse(self.info.purge_comp_time)

    @purge_time.setter
    def purge_time(self, val):
        self.info.purge_comp_time = self.umsg.purge_comp_time = timestr_to_secs(val)
        if ReservationFlags.PURGE not in self.flags:
            self.flags |= ReservationFlags.PURGE

    @property
    def start_time(self):
        return _raw_time(self.info.start_time)

    @start_time.setter
    def start_time(self, val):
        self.info.start_time = self.umsg.start_time = date_to_timestamp(val)

    @property
    def duration(self):
        cdef time_t duration = 0

        if self.start_time and self.info.end_time >= self.info.start_time:
            duration = <time_t>ctime.difftime(self.info.end_time,
                                              self.info.start_time)

        return int(duration / 60)

    @duration.setter
    def duration(self, val):
        val = timestr_to_mins(val)
        if not self.start_time:
            self.start_time = datetime.now()
        self.end_time = self.start_time + (val * 60)

    @property
    def is_active(self):
        cdef time_t now = ctime.time(NULL)
        if self.info.start_time <= now and self.info.end_time >= now:
            return True
        return False

    @property
    def tres(self):
        return cstr.to_dict(self.info.tres_str)

    @tres.setter
    def tres(self, val):
        cstr.fmalloc2(&self.info.tres_str, &self.umsg.tres_str,
                      cstr.dict_to_str(val))

    @property
    def users(self):
        return cstr.to_list(self.info.users)

    @users.setter
    def users(self, val):
        cstr.from_list2(&self.info.users, &self.umsg.users, val)

    @property
    def reoccurrence(self):
        return self._reoccurrence

    @reoccurrence.setter
    def reoccurrence(self, val):
        v = ReservationReoccurrence(val)
        current = self._reoccurrence
        self._reoccurrence = v
        if v == ReservationReoccurrence.NO:
            self.umsg.flags |= current._clear_flag
        else:
            self.umsg.flags |= v._flag

    @property
    def flags(self):
        return ReservationFlags(self.info.flags)

    @flags.setter
    def flags(self, val):
        flag = val
        if isinstance(val, list):
            flag = ReservationFlags.from_list(val)

        self.info.flags = flag.value
        self.umsg.flags = flag._get_flags_cleared()


class ReservationFlags(SlurmFlag):
    """Flags for Reservations that can be set.

    See {scontrol#OPT_Flags} for more info.
    """
    MAINTENANCE           = slurm.RESERVE_FLAG_MAINT,      slurm.RESERVE_FLAG_NO_MAINT
    MAGNETIC              = slurm.RESERVE_FLAG_MAGNETIC,   slurm.RESERVE_FLAG_NO_MAGNETIC
    FLEX                  = slurm.RESERVE_FLAG_FLEX,       slurm.RESERVE_FLAG_NO_FLEX
    IGNORE_RUNNING_JOBS   = slurm.RESERVE_FLAG_IGN_JOBS,   slurm.RESERVE_FLAG_NO_IGN_JOB
    ANY_NODES             = slurm.RESERVE_FLAG_ANY_NODES,  slurm.RESERVE_FLAG_NO_ANY_NODES
    STATIC_NODES          = slurm.RESERVE_FLAG_STATIC,     slurm.RESERVE_FLAG_NO_STATIC
    PARTITION_NODES_ONLY  = slurm.RESERVE_FLAG_PART_NODES, slurm.RESERVE_FLAG_NO_PART_NODES
    USER_DELETION         = slurm.RESERVE_FLAG_USER_DEL,   slurm.RESERVE_FLAG_NO_USER_DEL
    PURGE                 = slurm.RESERVE_FLAG_PURGE_COMP, slurm.RESERVE_FLAG_NO_PURGE_COMP
    SPECIFIC_NODES        = slurm.RESERVE_FLAG_SPEC_NODES
    NO_JOB_HOLD_AFTER_END = slurm.RESERVE_FLAG_NO_HOLD_JOBS
    OVERLAP               = slurm.RESERVE_FLAG_OVERLAP
    ALL_NODES             = slurm.RESERVE_FLAG_ALL_NODES
    SKIP                  = slurm.RESERVE_FLAG_SKIP
    SCHED_FAILED          = slurm.RESERVE_FLAG_SCHED_FAILED


class ReservationReoccurrence(SlurmEnum):
    """Different reocurrences for a Reservation.

    See {scontrol#OPT_Flags} for more info.
    """
    NO      = "NO"
    DAILY   = "DAILY",   slurm.RESERVE_FLAG_DAILY,   slurm.RESERVE_FLAG_NO_DAILY
    HOURLY  = "HOURLY",  slurm.RESERVE_FLAG_HOURLY,  slurm.RESERVE_FLAG_NO_HOURLY
    WEEKLY  = "WEEKLY",  slurm.RESERVE_FLAG_WEEKLY,  slurm.RESERVE_FLAG_NO_WEEKLY
    WEEKDAY = "WEEKDAY", slurm.RESERVE_FLAG_WEEKDAY, slurm.RESERVE_FLAG_NO_WEEKDAY
    WEEKEND = "WEEKEND", slurm.RESERVE_FLAG_WEEKEND, slurm.RESERVE_FLAG_NO_WEEKEND
