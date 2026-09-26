#########################################################################
# collections.pxd - pyslurm custom collections
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


cdef class MultiClusterMap:
    """Mapping of Multi-Cluster Data for a Collection.

    !!! note "TL;DR"

        If you have no need to write Multi-Cluster capable code and just work
        on a single Cluster, Collections inheriting from this Class behave
        just like a normal `dict`.

    This class enables collections to hold data from multiple Clusters if
    applicable.
    For quite a few Entities in Slurm it is possible to gather data from
    multiple Clusters. For example, with `sacct`, you can easily query Jobs
    running on different Clusters - provided your Cluster is joined in a
    Federation or simply part of a multi Cluster Setup.

    Collections like [pyslurm.db.Jobs][] inherit from this Class to enable
    holding such data from multiple Clusters. Internally, the data is
    structured in a `dict` like this (with [pyslurm.db.Jobs][] as an example):

    ```python
    data = {
        "LOCAL_CLUSTER": {
            1: pyslurm.db.Job(1),
            2: pyslurm.db.Job(2),
            ...
        },
        "OTHER_REMOTE_CLUSTER": {
            100: pyslurm.db.Job(100),
            101, pyslurm.db.Job(101)
            ...
        },
        ...
    }
    ```

    When a collection inherits from this class, its functionality will
    basically simulate a standard `dict` - with a few extensions to enable
    multi-cluster code.
    By default, even if your Collections contains Data from multiple Clusters,
    any operation will be targeted on the local Cluster data, if available.

    For example, with the data from above:

    ```python
    job = data[1]
    ```

    `job` would then hold the instance for `pyslurm.db.Job(1)` from the
    `LOCAL_CLUSTER` data.

    Alternatively, data can also be accessed like this:

    ```python
    job = data["OTHER_REMOTE_CLUSTER"][100]
    ```

    Here, you are directly specifying which Cluster data you want to access,
    and you will get the instance for `pyslurm.db.Job(100)` from the
    `OTHER_REMOTE_CLUSTER` data.

    Similarly, every method (where applicable) from a standard dict is
    extended with multi-cluster functionality (check out the examples on the
    methods)
    """
    cdef public dict data

    cdef:
        _typ
        _key_type
        _val_type
        _id_attr
        _cluster
