#########################################################################
# task_dist.pyx - job task distribution
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


cdef class TaskDistribution:

    def __init__(self, nodes="block", sockets="cyclic",
                 cores=None, pack=None, plane_size=None):
        self.nodes = nodes
        self.sockets = sockets
        self.cores = cores if cores else self.sockets
        self.pack = pack
        self.plane = plane_size
        self.state = self._get_task_dist_state()

    def __eq__(self, other):
        if not isinstance(other, TaskDistribution):
            return NotImplemented
        return self.as_int() == other.as_int()

    @staticmethod
    def from_int(dist):
        cdef TaskDistribution tdist = None

        if int(dist) <= 0 or dist == slurm.SLURM_DIST_UNKNOWN:
            return None

        if (dist & slurm.SLURM_DIST_STATE_BASE) != slurm.SLURM_DIST_UNKNOWN:
            tdist = _parse_task_dist_from_int(dist)

        dist_flag = dist & slurm.SLURM_DIST_STATE_FLAGS
        tdist = _parse_task_dist_flags_from_int(tdist, dist_flag)

        if tdist:
            tdist.state = dist

        return tdist

    def _to_str_no_flags(self):
        if self.plane:
            return "plane"

        dist_str = ""
        nodes = self.nodes
        if nodes is not None and nodes != "*":
            dist_str = f"{nodes}"
        else:
            dist_str = "block"

        sockets = self.sockets
        if sockets is not None and sockets != "*":
            dist_str = f"{dist_str}:{sockets}"
        else:
            dist_str = f"{dist_str}:cyclic"

        cores = self.cores
        if cores is not None and cores != "*":
            dist_str = f"{dist_str}:{cores}"
        else:
            dist_str = f"{dist_str}:{sockets}"

        return dist_str

    def to_str(self):
        dist_str = self._to_str_no_flags()

        if self.pack is not None:
            dist_str = f"{dist_str},{'Pack' if self.pack else 'NoPack'}"

        return dist_str

    def to_dict(self, recursive = False):
        return {
            "nodes": self.nodes,
            "sockets": self.sockets,
            "cores": self.cores,
            "plane": self.plane,
            "pack": self.pack,
        }

    def as_int(self):
        return self.state

    def _get_task_dist_state(self):
        cdef task_dist_states_t dist_state

        dist_str = self._to_str_no_flags()
        if dist_str == "plane":
            return slurm.SLURM_DIST_PLANE

        dist_state = _parse_str_to_task_dist_int(dist_str)
        if dist_state == slurm.SLURM_DIST_UNKNOWN:
            raise ValueError(f"Invalid distribution specification: {dist_str}")

        # Check for Pack/NoPack
        # Don't do anything if it is None
        if self.pack:
            dist_state = <task_dist_states_t>(dist_state | slurm.SLURM_DIST_PACK_NODES)
        elif self.pack is not None and not self.pack:
            dist_state = <task_dist_states_t>(dist_state | slurm.SLURM_DIST_NO_PACK_NODES)

        return dist_state

    @staticmethod
    def from_str(dist_str):
        cdef TaskDistribution tdist = TaskDistribution.__new__(TaskDistribution)

        # Plane method - return early because nothing else can be
        # specified when this is set.
        if "plane" in dist_str:
            if "plane=" in dist_str:
                plane_size = u16(dist_str.split("=", 1)[1])
                return TaskDistribution(plane_size=plane_size)
            else:
                return TaskDistribution(plane_size=True)

        # [0] = distribution method for nodes:sockets:cores
        # [1] = pack/nopack specification (true or false)
        dist_items = dist_str.split(",", 1)

        # Parse the different methods
        dist_methods = dist_items[0].split(":")
        if len(dist_methods) and dist_methods[0] != "*":
            tdist.nodes = dist_methods[0]

        if len(dist_methods) > 2 and dist_methods[1] != "*":
            tdist.sockets = dist_methods[1]

        if len(dist_methods) >= 3:
            if dist_methods[2] == "*":
                tdist.cores = tdist.sockets
            else:
                tdist.cores = dist_methods[2]

        if len(dist_items) > 1:
            if dist_items[1].casefold() == "pack":
                tdist.pack = True
            elif dist_items[1].casefold() == "nopack":
                tdist.pack = False

        tdist.state = tdist._get_task_dist_state()
        return tdist


# https://github.com/SchedMD/slurm/blob/510ba4f17dfa559b579aa054cb8a415dcc224abc/src/common/proc_args.c#L319
def _parse_task_dist_from_int(dist):
    cdef TaskDistribution out = TaskDistribution.__new__(TaskDistribution)

    state = dist & slurm.SLURM_DIST_STATE_BASE
    if state == slurm.SLURM_DIST_BLOCK:
        out.nodes = "block"
    elif state == slurm.SLURM_DIST_CYCLIC:
        out.nodes = "cyclic"
    elif state == slurm.SLURM_DIST_PLANE:
        out.plane = state
    elif state == slurm.SLURM_DIST_ARBITRARY:
        out.nodes = "arbitrary"
    elif state == slurm.SLURM_DIST_CYCLIC_CYCLIC:
        out.nodes = "cyclic"
        out.sockets = "cyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_BLOCK:
        out.nodes = "cyclic"
        out.sockets = "block"
    elif state == slurm.SLURM_DIST_CYCLIC_CFULL:
        out.nodes = "cyclic"
        out.sockets = "fcyclic"
    elif state == slurm.SLURM_DIST_BLOCK_CYCLIC:
        out.nodes = "block"
        out.sockets = "cyclic"
    elif state == slurm.SLURM_DIST_BLOCK_BLOCK:
        out.nodes = "block"
        out.sockets = "block"
    elif state == slurm.SLURM_DIST_BLOCK_CFULL:
        out.nodes = "block"
        out.sockets = "fcyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_CYCLIC_CYCLIC:
        out.nodes = "cyclic"
        out.sockets = "cyclic"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_CYCLIC_BLOCK:
        out.nodes = "cyclic"
        out.sockets = "cyclic"
        out.cores = "block"
    elif state == slurm.SLURM_DIST_CYCLIC_CYCLIC_CFULL:
        out.nodes = "cyclic"
        out.sockets = "cyclic"
        out.cores = "fcyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_BLOCK_CYCLIC:
        out.nodes = "cyclic"
        out.sockets = "block"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_BLOCK_CYCLIC:
        out.nodes = "cyclic"
        out.sockets = "block"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_BLOCK_BLOCK:
        out.nodes = "cyclic"
        out.sockets = "block"
        out.cores = "block"
    elif state == slurm.SLURM_DIST_CYCLIC_BLOCK_CFULL:
        out.nodes = "cyclic"
        out.sockets = "block"
        out.cores = "fcyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_CFULL_CYCLIC:
        out.nodes = "cyclic"
        out.sockets = "fcyclic"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_CYCLIC_CFULL_BLOCK:
        out.nodes = "cyclic"
        out.sockets = "fcyclic"
        out.cores = "block"
    elif state == slurm.SLURM_DIST_CYCLIC_CFULL_CFULL:
        out.nodes = "cyclic"
        out.sockets = "fcyclic"
        out.cores = "fcyclic"
    elif state == slurm.SLURM_DIST_BLOCK_CYCLIC_CYCLIC:
        out.nodes = "block"
        out.sockets = "cyclic"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_BLOCK_CYCLIC_BLOCK:
        out.nodes = "block"
        out.sockets = "cyclic"
        out.cores = "block"
    elif state == slurm.SLURM_DIST_BLOCK_CYCLIC_CFULL:
        out.nodes = "block"
        out.sockets = "cyclic"
        out.cores = "fcyclic"
    elif state == slurm.SLURM_DIST_BLOCK_BLOCK_CYCLIC:
        out.nodes = "block"
        out.sockets = "block"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_BLOCK_BLOCK_BLOCK:
        out.nodes = "block"
        out.sockets = "block"
        out.cores = "block"
    elif state == slurm.SLURM_DIST_BLOCK_BLOCK_CFULL:
        out.nodes = "block"
        out.sockets = "block"
        out.cores = "fcyclic"
    elif state == slurm.SLURM_DIST_BLOCK_CFULL_CYCLIC:
        out.nodes = "block"
        out.sockets = "fcyclic"
        out.cores = "cyclic"
    elif state == slurm.SLURM_DIST_BLOCK_CFULL_BLOCK:
        out.nodes = "block"
        out.sockets = "fcyclic"
        out.cores = "block"
    elif state == slurm.SLURM_DIST_BLOCK_CFULL_CFULL:
        out.nodes = "block"
        out.sockets = "fcyclic"
        out.cores = "fcyclic"
    else:
        return None

    return out


def _parse_task_dist_flags_from_int(TaskDistribution dst, dist_flag):
    if not dist_flag:
        return dst

    cdef TaskDistribution _dst = dst
    if not _dst:
        _dst = TaskDistribution.__new__(TaskDistribution)

    if dist_flag == slurm.SLURM_DIST_PACK_NODES:
        _dst.pack = True
    elif dist_flag == slurm.SLURM_DIST_NO_PACK_NODES:
        _dst.pack = False

    return _dst


def _parse_str_to_task_dist_int(dist_str):
    # Select the correct distribution method according to dist_str.
    if dist_str == "cyclic":
        return slurm.SLURM_DIST_CYCLIC
    elif dist_str == "block":
        return slurm.SLURM_DIST_BLOCK
    elif dist_str == "arbitrary" or dist_str == "hostfile":
        return slurm.SLURM_DIST_ARBITRARY
    elif dist_str == "cyclic:cyclic":
        return slurm.SLURM_DIST_CYCLIC_CYCLIC
    elif dist_str == "cyclic:block":
        return slurm.SLURM_DIST_CYCLIC_BLOCK
    elif dist_str == "block:block":
        return slurm.SLURM_DIST_BLOCK_BLOCK
    elif dist_str == "block:cyclic":
        return slurm.SLURM_DIST_BLOCK_CYCLIC
    elif dist_str == "block:fcyclic":
        return slurm.SLURM_DIST_BLOCK_CFULL
    elif dist_str == "cyclic:fcyclic":
        return slurm.SLURM_DIST_CYCLIC_CFULL
    elif dist_str == "cyclic:cyclic:cyclic":
        return slurm.SLURM_DIST_CYCLIC_CYCLIC_CYCLIC
    elif dist_str == "cyclic:cyclic:block":
        return slurm.SLURM_DIST_CYCLIC_CYCLIC_BLOCK
    elif dist_str == "cyclic:cyclic:fcyclic":
        return slurm.SLURM_DIST_CYCLIC_CYCLIC_CFULL
    elif dist_str == "cyclic:block:cyclic":
        return slurm.SLURM_DIST_CYCLIC_BLOCK_CYCLIC
    elif dist_str == "cyclic:block:block":
        return slurm.SLURM_DIST_CYCLIC_BLOCK_BLOCK
    elif dist_str == "cyclic:block:fcyclic":
        return slurm.SLURM_DIST_CYCLIC_BLOCK_CFULL
    elif dist_str == "cyclic:fcyclic:cyclic":
        return slurm.SLURM_DIST_CYCLIC_CFULL_CYCLIC
    elif dist_str == "cyclic:fcyclic:block":
        return slurm.SLURM_DIST_CYCLIC_CFULL_BLOCK
    elif dist_str == "cyclic:fcyclic:fcyclic":
        return slurm.SLURM_DIST_CYCLIC_CFULL_CFULL
    elif dist_str == "block:cyclic:cyclic":
        return slurm.SLURM_DIST_BLOCK_CYCLIC_CYCLIC
    elif dist_str == "block:cyclic:block":
        return slurm.SLURM_DIST_BLOCK_CYCLIC_BLOCK
    elif dist_str == "block:cyclic:fcyclic":
        return slurm.SLURM_DIST_BLOCK_CYCLIC_CFULL
    elif dist_str == "block:block:cyclic":
        return slurm.SLURM_DIST_BLOCK_BLOCK_CYCLIC
    elif dist_str == "block:block:block":
        return slurm.SLURM_DIST_BLOCK_BLOCK_BLOCK
    elif dist_str == "block:block:fcyclic":
        return slurm.SLURM_DIST_BLOCK_BLOCK_CFULL
    elif dist_str == "block:fcyclic:cyclic":
        return slurm.SLURM_DIST_BLOCK_CFULL_CYCLIC
    elif dist_str == "block:fcyclic:block":
        return slurm.SLURM_DIST_BLOCK_CFULL_BLOCK
    elif dist_str == "block:fcyclic:fcyclic":
        return slurm.SLURM_DIST_BLOCK_CFULL_CFULL
    else:
        return slurm.SLURM_DIST_UNKNOWN
