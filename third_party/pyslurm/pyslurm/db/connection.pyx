#########################################################################
# connection.pyx - pyslurm slurmdbd database connection
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


def _open_conn_or_error(conn):
    if not conn:
        conn = Connection.open()

    if not conn.is_open:
        raise ValueError("Database connection is not open")

    return conn


cdef class Connection:

    def __cinit__(self):
        self.ptr = NULL
        self.flags = 0

    def __init__(self):
        raise RuntimeError("A new connection should be created through "
                           "calling Connection.open()")

    def __dealloc__(self):
        self.close()

    def __repr__(self):
        state = "open" if self.is_open else "closed"
        return f'pyslurm.db.{self.__class__.__name__} is {state}'

    @staticmethod
    def open():
        """Open a new connection to the slurmdbd

        Raises:
            (pyslurm.RPCError): When opening the connection fails

        Returns:
            (pyslurm.db.Connection): Connection to slurmdbd

        Examples:
            >>> import pyslurm
            >>> connection = pyslurm.db.Connection.open()
            >>> print(connection.is_open)
            True
        """
        cdef Connection conn = Connection.__new__(Connection)
        conn.ptr = <void*>slurmdb_connection_get(&conn.flags)
        if not conn.ptr:
            raise RPCError(msg="Failed to open onnection to slurmdbd")

        return conn

    def close(self):
        """Close the current connection.

        Examples:
            >>> import pyslurm
            >>> connection = pyslurm.db.Connection.open()
            >>> ...
            >>> connection.close()
            >>> print(connection.is_open)
            False
        """
        if self.is_open:
            slurmdb_connection_close(&self.ptr)
            self.ptr = NULL

    def commit(self):
        """Commit recent changes."""
        if slurmdb_connection_commit(self.ptr, 1) == slurm.SLURM_ERROR:
            raise RPCError("Failed to commit database changes.")

    def rollback(self):
        """Rollback recent changes."""
        if slurmdb_connection_commit(self.ptr, 0) == slurm.SLURM_ERROR:
            raise RPCError("Failed to rollback database changes.")

    @property
    def is_open(self):
        if self.ptr:
            return True
        else:
            return False
