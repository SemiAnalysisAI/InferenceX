#########################################################################
# api.pyx - pyslurm core API
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

from pyslurm import settings


def slurm_init(config_path=None):
    """Initialize the Slurm API.

    This function must be called first before certain RPC functions can be
    executed. slurm_init is automatically called when the pyslurm module is
    loaded.

    Args:
        config_path (str, optional):
            An absolute path to the slurm config file to use. The default is
            None, so libslurm will automatically detect its config.
    """
    slurm.slurm_init(cstr.from_unicode(config_path))
    settings.init()


def slurm_fini():
    """Clean up data structures previously allocated through slurm_init."""
    slurm.slurm_fini()
