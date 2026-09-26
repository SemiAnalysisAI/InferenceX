#########################################################################
# util.pyx - pyslurm slurmdbd util functions
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


cdef make_char_list(list_t **in_list, vals):
    if not vals:
        return None

    # Make a new SlurmList wrapper with the values
    cdef SlurmList slist = SlurmList(vals)

    # Make sure the previous list is deallocated
    if in_list[0]:
        slurm_list_destroy(in_list[0])

    # Assign the pointer from slist to in_list, and give up ownership of slist
    in_list[0] = slist.info
    slist.owned = False


cdef slurm_list_to_pylist(list_t *in_list):
    return SlurmList.wrap(in_list, owned=False).to_pylist()


cdef qos_list_to_pylist(list_t *in_list, qos_data):
    if not in_list:
        return []

    cdef list qos_nums = SlurmList.wrap(in_list, owned=False).to_pylist()
    return [qos.name for qos_id, qos in qos_data.items()
            if qos_id in qos_nums]


cdef class SlurmListItem:

    def __cinit__(self):
        self.data = NULL

    @staticmethod
    cdef SlurmListItem from_ptr(void *item):
        cdef SlurmListItem wrap = SlurmListItem.__new__(SlurmListItem)
        wrap.data = item
        return wrap

    @property
    def has_data(self):
        if self.data:
            return True
        else:
            return False

    def to_str(self):
        # Mostly for debugging purposes. Can only be used "safely" if we have
        # a char* list
        cdef char* entry = <char*>self.data
        return cstr.to_unicode(entry)


cdef class SlurmList:
    """Convenience Wrapper around slurms List type"""
    def __cinit__(self):
        self.info = NULL
        self.itr = NULL
        self.itr_cnt = 0
        self.cnt = 0
        self.owned = True

    def __init__(self, vals=None):
        self.info = slurm_list_create(slurm_xfree_ptr)
        self.append(vals)

    def __dealloc__(self):
        self._dealloc_itr()
        self._dealloc_list()

    def _dealloc_list(self):
        if self.info is not NULL and self.owned:
            slurm_list_destroy(self.info)
            self.cnt = 0
            self.info = NULL

    def _dealloc_itr(self):
        if self.itr:
            slurm_list_iterator_destroy(self.itr)
            self.itr_cnt = 0
            self.itr = NULL

    def __iter__(self):
        self._dealloc_itr()
        if not self.is_null:
            self.itr = slurm_list_iterator_create(self.info)

        return self

    def __next__(self):
        if self.is_null or self.is_itr_null:
            raise StopIteration

        if self.itr_cnt < self.cnt:
            self.itr_cnt += 1
            return SlurmListItem.from_ptr(slurm_list_next(self.itr))

        self._dealloc_itr()
        raise StopIteration

    @staticmethod
    def iter_and_pop(SlurmList li):
        while li.cnt > 0:
            yield SlurmListItem.from_ptr(slurm_list_pop(li.info))
            li.cnt -= 1

    @staticmethod
    cdef SlurmList create(slurm.ListDelF delfunc, owned=True):
        cdef SlurmList wrapper = SlurmList.__new__(SlurmList)
        wrapper.info = slurm_list_create(delfunc)
        wrapper.owned = owned
        return wrapper

    @staticmethod
    cdef SlurmList wrap(list_t *li, owned=True):
        cdef SlurmList wrapper = SlurmList.__new__(SlurmList)
        if not li:
            return wrapper

        wrapper.info = li
        wrapper.cnt = slurm_list_count(li)
        wrapper.owned = owned
        return wrapper

    def to_pylist(self):
        cdef:
            SlurmListItem item
            list out = []

        for item in self:
            if not item.has_data:
                continue

            pystr = cstr.to_unicode(<char*>item.data)
            if pystr:
                out.append(int(pystr) if pystr.isdigit() else pystr)

        return out

    def append(self, vals):
        cdef char *entry = NULL

        if not vals:
            return None

        to_add = vals
        if not isinstance(vals, list):
            # If it is not a list, then anything that can't be casted to str
            # will error below anyways
            to_add = [vals]

        for val in to_add:
            if val:
                entry = NULL
                cstr.fmalloc(&entry, str(val))
                slurm_list_append(self.info, entry)
                self.cnt += 1

    @property
    def is_itr_null(self):
        if not self.itr:
            return True
        else:
            return False

    @property
    def is_null(self):
        if not self.info:
            return True
        else:
            return False
