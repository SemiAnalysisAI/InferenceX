#########################################################################
# collections.pyx - pyslurm custom collections
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
"""Custom Collection utilities"""

from pyslurm import settings
import json
from typing import Union, Any


class BaseView:
    """Base View for all other Views"""
    def __init__(self, mcm):
        self._mcm = mcm
        self._data = mcm.data

    def __len__(self):
        return len(self._mcm)

    def __repr__(self):
        data = ", ".join(map(repr, self))
        return f'{self.__class__.__name__}([{data}])'


class ValuesView(BaseView):
    """A simple Value View

    When iterating over an instance of this View, this will yield all values
    from all clusters.
    """
    def __contains__(self, val):
        try:
            item = self._mcm.get(
                key=self._mcm._item_id(val),
                cluster=val.cluster
            )
            return item is val or item == val
        except AttributeError:
            pass

        return False

    def __iter__(self):
        for cluster in self._mcm.data.values():
            for item in cluster.values():
                yield item


class ClustersView(BaseView):
    """A simple Cluster-Keys View

    When iterating over an instance of this View, it will yield all the
    Cluster names of the collection.
    """
    def __contains__(self, item):
        return item in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        yield from self._data


class MCKeysView(BaseView):
    """A Multi-Cluster Keys View

    Unlike KeysView, when iterating over an MCKeysView instance, this will
    yield a 2-tuple in the form `(cluster, key)`.

    Similarly, when checking whether this View contains a Key with the `in`
    operator, a 2-tuple must be used in the form described above.
    """
    def __contains__(self, item):
        cluster, key, = item
        return key in self._data[cluster]

    def __iter__(self):
        for cluster, keys in self._data.items():
            for key in keys:
                yield (cluster, key)


class KeysView(BaseView):
    """A simple Keys View of a collection

    When iterating, this yields all the keys found from each Cluster in the
    collection. Note that unlike the KeysView from a `dict`, the keys here
    aren't unique and may appear multiple times.

    If you indeed have multiple Clusters in a collection and need to tell the
    keys apart, use the `with_cluster()` function.
    """
    def __contains__(self, item):
        return item in self._mcm

    def __iter__(self):
        for cluster, keys in self._data.items():
            yield from keys

    def with_cluster(self):
        """Return a Multi-Cluster Keys View.

        Returns:
            (MCKeysView): Multi-Cluster Keys View.
        """
        return MCKeysView(self._mcm)


class ItemsView(BaseView):
    """A simple Items View of a collection.

    Returns a 2-tuple in the form of `(key, value)` when iterating.

    Similarly, when checking whether this View contains an Item with the `in`
    operator, a 2-tuple must be used.
    """
    def __contains__(self, item):
        key, val = item

        try:
            out = self._mcm.data[item.cluster][key]
        except (KeyError, AttributeError):
            return False
        else:
            return out is val or out == val

    def __iter__(self):
        for cluster, data in self._mcm.data.items():
            for key in data:
                yield (key, data[key])

    def with_cluster(self):
        """Return a Multi-Cluster Items View.

        Returns:
            (MCItemsView): Multi-Cluster Items View.
        """
        return MCItemsView(self._mcm)


class MCItemsView(BaseView):
    """A Multi-Cluster Items View.

    This differs from ItemsView in that it returns a 3-tuple in the form of
    `(cluster, key, value)` when iterating.

    Similarly, when checking whether this View contains an Item with the `in`
    operator, a 3-tuple must be used.
    """
    def __contains__(self, item):
        cluster, key, val = item

        try:
            out = self._mcm.data[cluster][key]
        except KeyError:
            return False
        else:
            return out is val or out == val

    def __iter__(self):
        for cluster, data in self._mcm.data.items():
            for key in data:
                yield (cluster, key, data[key])


cdef class MultiClusterMap:

    def __init__(self, data, typ=None, val_type=None,
                 key_type=None, id_attr=None, init_data=True):
        self.data = {} if init_data else data
        self._typ = typ
        self._key_type = key_type
        self._val_type = val_type
        self._id_attr = id_attr
        self._cluster = settings.LOCAL_CLUSTER
        if init_data:
            self._init_data(data)

    def _init_data(self, data):
        if isinstance(data, list):
            for item in data:
                if isinstance(item, self._key_type):
                    item = self._val_type(item)
                    if self._cluster not in self.data:
                        self.data[self._cluster] = {}

                self.data[self._cluster].update({self._item_id(item): item})
        elif isinstance(data, str):
            itemlist = data.split(",")
            items = {self._key_type(item):self._val_type(item)
                     for item in itemlist}
            self.data[self._cluster] = items
        elif isinstance(data, dict):
            self.update(data)
        elif data is not None:
            raise TypeError(f"Invalid Type: {type(data).__name__}")

    def _check_for_value(self, val_id, cluster):
        cluster_data = self.data.get(cluster)
        if cluster_data and val_id in cluster_data:
            return True
        return False

    def _get_cluster(self):
        cluster = None
        if not self.data or self._cluster in self.data:
            cluster = self._cluster
        else:
            try:
                cluster = next(iter(self.keys()))
            except StopIteration:
                raise KeyError("Collection is Empty") from None

        return cluster

    def _get_key_and_cluster(self, item):
        if isinstance(item, self._val_type):
            cluster, key = item.cluster, self._item_id(item)
        elif isinstance(item, tuple) and len(item) == 2:
            cluster, key = item
        else:
            cluster, key = self._get_cluster(), item

        return cluster, key

    def _check_val_type(self, item):
        if not isinstance(item, self._val_type):
            raise TypeError(f"Invalid Type: {type(item).__name__}. "
                            f"{self._val_type}.__name__ is required.")

    def _item_id(self, item):
        return self._id_attr.__get__(item)

    def _iter_clusters_dict(self, other):
        for key in other:
            try:
                iterator = iter(other[key])
            except TypeError as e:
                try:
                    cluster = self._get_cluster()
                except KeyError:
                    cluster = self._cluster

                if not cluster in self.data:
                    self.data[cluster] = {}
                yield (cluster, other)
                break
            else:
                cluster = key
                if not cluster in self.data:
                    self.data[cluster] = {}
                yield (cluster, other[cluster])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.data == other.data
        return NotImplemented

    def __getitem__(self, item):
        if item in self.data:
            return self.data[item]

        cluster, key = self._get_key_and_cluster(item)
        return self.data[cluster][key]

    def __setitem__(self, where, item):
        if where in self.data:
            self.data[where] = item
        else:
            cluster, key = self._get_key_and_cluster(where)
            self.data[cluster][key] = item

    def __delitem__(self, item):
        if item in self.data:
            del self.data[item]
        else:
            cluster, key = self._get_key_and_cluster(item)
            del self.data[cluster][key]

    def __len__(self):
        return sum(len(data) for data in self.data.values())

    def __repr__(self):
        data = ", ".join(map(repr, self.data.values()))
        return f'pyslurm.{self._typ}({data})'

    def __contains__(self, item):
        if isinstance(item, self._val_type):
            item = (item.cluster, self._item_id(item))
            return self.get(item, default=None) is not None
            # return self._check_for_value(self._item_id(item), item.cluster)
        elif isinstance(item, self._key_type):
            found = False
            for cluster, data in self.data.items():
                if item in data:
                    found = True
            return found
        elif isinstance(item, tuple):
            return self.get(item, default=None) is not None
            # return self._check_for_value(item, cluster)

        return False

    def __iter__(self):
        return iter(self.keys())

    def __bool__(self):
        return bool(self.data)

    def __copy__(self):
        return self.copy()

    def __or__(self, other):
        if isinstance(other, MultiClusterMap):
            if isinstance(self, dict):
                return NotImplemented

            out = self.copy()
            out |= other
            return out
        elif isinstance(other, dict):
            out = self.copy()
            for cluster, data in self._iter_clusters_dict(other):
                out.data[cluster] = {**self.data[cluster], **data}
            return out
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, MultiClusterMap):
            out = other.copy()
            out |= self
            return out
        elif isinstance(other, dict):
            out = self.copy()
            for cluster, data in self._iter_clusters_dict(other):
                out.data[cluster] = {**data, **self.data[cluster]}
            return out
        return NotImplemented

    def __ior__(self, other):
        if isinstance(other, MultiClusterMap):
            for cluster in other.clusters():
                if not cluster in self.data:
                    self.data[cluster] = {}
                self.data[cluster].update(other.data[cluster])
        else:
            for cluster, data in self._iter_clusters_dict(other):
                self.data[cluster].update(data)
        return self

    def copy(self):
        """Return a Copy of this instance."""
        out = self.__class__.__new__(self.__class__)
        super(self.__class__, out).__init__(
            data=self.data.copy(),
            typ=self._typ,
            key_type=self._key_type,
            val_type=self._val_type,
            init_data=False,
        )
        return out

    def get(self, key, default=None):
        """Get the specific value for a Key

        This behaves like `dict`'s `get` method, with the difference that you
        can additionally pass in a 2-tuple in the form of `(cluster, key)` as
        the key, which can be helpful if this collection contains data from
        multiple Clusters.

        If just a key without notion of the Cluster is given, access to the
        local cluster data is implied. If this collection does however not
        contain data from the local cluster, the first cluster detected
        according to `next(iter(self.keys()))` will be used.

        Examples:
            Get a Job from the local Cluster.

            >>> job_id = 1
            >>> job = data.get(job_id)

            Get a Job from another Cluster in the Collection, by providing a
            2-tuple with the cluster identifier:

            >>> job_id = 1
            >>> job = data.get(("REMOTE_CLUSTER", job_id))
        """
        cluster, key = self._get_key_and_cluster(key)
        return self.data.get(cluster, {}).get(key, default)

    def add(self, item):
        """An Item to add to the collection

        Note that a collection can only hold its specific type.
        For example, a collection of [pyslurm.db.Jobs][] can only hold
        [pyslurm.db.Job][] objects. Trying to add anything other than the
        accepted type will raise a TypeError.

        Args:
            item (Any):
                Item to add to the collection.

        Raises:
            TypeError: When an item with an unexpected type not belonging to
                the collection was added.

        Examples:
            Add a `pyslurm.db.Job` instance to the `pyslurm.db.Jobs`
            collection.

            >>> import pyslurm
            >>> jobs = pyslurm.db.Jobs()
            >>> job = pyslurm.db.Job(1)
            >>> jobs.add(job)
            >>> print(jobs)
            pyslurm.db.Jobs({1: pyslurm.db.Job(1)})
        """
        if item.cluster not in self.data:
            self.data[item.cluster] = {}

        self._check_val_type(item)
        self.data[item.cluster][self._item_id(item)] = item

    def to_json(self, multi_cluster=False):
        """Convert the collection to JSON.

        Returns:
            (str): JSON formatted string from `json.dumps()`
        """
        if not self.data:
            return '{}'

        return json.dumps(self.to_dict(multi_cluster=multi_cluster, recursive=True))

    def to_dict(self, multi_cluster=False, recursive=False):
        if not self.data:
            return {}

        data = multi_dict_recursive(self, recursive)
        if multi_cluster:
            return data
        else:
            cluster = self._get_cluster()
            return data[cluster]

    def keys(self):
        """Return a View of all the Keys in this collection

        Returns:
            (KeysView): View of all Keys

        Examples:
            Iterate over all Keys from all Clusters:

            >>> for key in collection.keys()
            ...     print(key)

            Iterate over all Keys from all Clusters with the name of the
            Cluster additionally provided:

            >>> for cluster, key in collection.keys().with_cluster()
            ...     print(cluster, key)
        """
        return KeysView(self)

    def items(self):
        """Return a View of all the Values in this collection

        Returns:
            (ItemsView): View of all Items

        Examples:
            Iterate over all Items from all Clusters:

            >>> for key, value in collection.items()
            ...     print(key, value)

            Iterate over all Items from all Clusters with the name of the
            Cluster additionally provided:

            >>> for cluster, key, value in collection.items().with_cluster()
            ...     print(cluster, key, value)
        """
        return ItemsView(self)

    def values(self):
        """Return a View of all the Values in this collection

        Returns:
            (ValuesView): View of all Values

        Examples:
            Iterate over all Values from all Clusters:

            >>> for value in collection.values()
            ...     print(value)
        """
        return ValuesView(self)

    def clusters(self):
        """Return a View of all the Clusters in this collection

        Returns:
            (ClustersView): View of Cluster keys

        Examples:
            Iterate over all Cluster-Names the Collection contains:

            >>> for cluster in collection.clusters()
            ...     print(cluster)
        """
        return ClustersView(self)

    def popitem(self):
        """Remove and return a `(key, value)` pair as a 2-tuple"""
        try:
            item = next(iter(self.values()))
        except StopIteration:
            raise KeyError from None

        key = self._item_id(item)
        del self.data[item.cluster][key]
        return (key, item)

    def clear(self):
        """Clear the collection"""
        self.data.clear()

    def pop(self, key, default=None):
        """Remove key from the collection and return the value

        This behaves like `dict`'s `pop` method, with the difference that you
        can additionally pass in a 2-tuple in the form of `(cluster, key)` as
        the key, which can be helpful if this collection contains data from
        multiple Clusters.

        If just a key without notion of the Cluster is given, access to the
        local cluster data is implied. If this collection does however not
        contain data from the local cluster, the first cluster detected
        according to `next(iter(self.keys()))` will be used.
        """
        item = self.get(key, default=default)
        if item is default or item == default:
            return default

        cluster = item.cluster
        del self.data[cluster][key]
        if not self.data[cluster]:
            del self.data[cluster]

        return item

    def update(self, data={}, **kwargs):
        """Update the collection.

        This functions like `dict`'s `update` method.
        """
        for cluster, data in self._iter_clusters_dict(data):
            self.data[cluster].update(data)

        for cluster, data in self._iter_clusters_dict(kwargs):
            self.data[cluster].update(data)


def multi_reload(cur, frozen=True):
    if not cur:
        return cur

    new = cur.__class__.load()
    for cluster, item in list(cur.keys().with_cluster()):
        if (cluster, item) in new.keys().with_cluster():
            cur[cluster][item] = new.pop(item, cluster)
        elif not frozen:
            del cur[cluster][item]

    if not frozen:
        for cluster, item in new.keys().with_cluster():
            if (cluster, item) not in cur.keys().with_cluster():
                cur[cluster][item] = new[cluster][item]

    return cur


# TODO: fix this function name to be less bad
def dict_recursive(collection, recursive = False):
    cdef dict out = {}
    for item_id, item in collection.items():
        if hasattr(item, "to_dict"):
            out[item_id] = item.to_dict(recursive=recursive)
    return out


def to_json(collection):
    return json.dumps(dict_recursive(collection), recursive=True)


# TODO: fix this function name to be less bad
def multi_dict_recursive(collection, recursive = False):
    cdef dict out = collection.data.copy()
    for cluster, data in collection.data.items():
        out[cluster] = dict_recursive(data, recursive)
    return out


def sum_property(collection, prop, startval=0):
    out = startval
    for item in collection.values():
        data = prop.__get__(item)
        if data is not None:
            out += data

    return out
