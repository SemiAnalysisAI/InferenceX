#########################################################################
# slurmctld/base.pyx - pyslurm slurmctld api functions
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

from pyslurm.core.error import verify_rpc, RPCError
from pyslurm.utils.helpers import instance_to_dict
from pyslurm.utils import cstr
from pyslurm.utils.uint import u16_parse
from typing import Union
import time
from enum import IntEnum
from .config import Config
from .enums import ShutdownMode


cdef class PingResponse:

    def to_dict(self, recursive = False):
        """Slurmctld ping response formatted as dictionary.

        Returns:
            (dict): Ping response as a dict

        Examples:
            >>> from pyslurm import slurmctld
            >>> ctld_primary = slurmctld.Config.ping(0)
            >>> primary_dict = ctld_primary.to_dict()
        """
        return instance_to_dict(self, recursive)


def ping(index):
    """Ping a Slurm controller

    Returns:
        (pyslurm.slurmctld.PingResponse): a ping response

    Examples:
        >>> from pyslurm import slurmctld
        >>> resp = slurmctld.ping(0)
        >>> print(resp.hostname, resp.latency)
        slurmctl 1.246
    """
    t0 = time.perf_counter()
    rc = slurm_ping(index)
    t1 = time.perf_counter()

    verify_rpc(rc)
    ctl_cnt = slurm.slurm_conf.control_cnt

    if index >= ctl_cnt:
        raise RPCError(msg="Invalid Index specified.")

    info = PingResponse()
    info.is_primary = index == 0
    info.is_responding = not rc
    info.index = index
    info.hostname = cstr.to_unicode(slurm.slurm_conf.control_machine[index])
    info.latency = round((t1 - t0) * 1000, 3)

    return info


def ping_primary():
    """Ping the primary Slurm Controller.

    See `ping()` for more information and examples.

    Returns:
        (pyslurm.slurmctld.PingResponse): a ping response

    Examples:
        >>> from pyslurm import slurmctld
        >>> resp = slurmctld.ping_primary()
        >>> print(resp.hostname, resp.latency, resp.is_primary)
        slurmctl 1.222 True
    """
    return ping(0)


def ping_backup():
    """Ping the first backup Slurm Controller.

    See `ping()` for more information and examples.

    Returns:
        (pyslurm.slurmctld.PingResponse): a ping response

    Examples:
        >>> from pyslurm import slurmctld
        >>> resp = slurmctld.ping_backup()
        >>> print(resp.hostname, resp.latency, resp.is_primary)
        slurmctlbackup 1.373 False
    """
    return ping(1)


def ping_all():
    """Ping all Slurm Controllers.

    Returns:
        (list[pyslurm.slurmctld.PingResponse]): a list of ping responses

    Raises:
        (pyslurm.RPCError): When the ping was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> resps = slurmctld.ping_all()
        >>> for resp in resps:
        ...     print(resp.hostname, resp.latency)
        ...
        slurmctl 1.246
        slurmctlbackup 1.373
    """
    cdef list out = []

    ctl_cnt = slurm.slurm_conf.control_cnt
    for i in range(ctl_cnt):
        out.append(ping(i))

    return out


def shutdown(mode: Union[ShutdownMode, int]):
    """Shutdown Slurm Controller or all Daemons

    Args:
        mode:
            Whether only the Slurm controller shut be downed, or also all other
            slurmd daemons.

    Raises:
        (pyslurm.RPCError): When shutdowning the daemons was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.shutdown(slurmctld.ShutdownMode.ALL)
    """
    verify_rpc(slurm_shutdown(int(mode)))


def reconfigure():
    """Trigger Slurm Controller to reload the Config

    Raises:
        (pyslurm.RPCError): When reconfiguring was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.reconfigure()
    """
    verify_rpc(slurm_reconfigure())


def takeover(index = 1):
    """Let a Backup Slurm Controller take over as the Primary.

    Args:
        index (int, optional=1):
            Index of the Backup Controller that should take over. By default,
            the `index` is `1`, meaning the next Controller configured after
            the Primary in `slurm.conf` (second `SlurmctldHost` entry) will be
            asked to take over operation.

            If you have more than one backup controller configured, you can for
            example also pass `2` as the index.

    Raises:
        (pyslurm.RPCError): When reconfiguring was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.takeover(1)
    """
    verify_rpc(slurm_takeover(index))


def add_debug_flags(flags):
    """Add DebugFlags to `slurmctld`

    Args:
        flags (list[str]):
            For an available list of possible values, please check the
            `slurm.conf` documentation under `DebugFlags`.

    Raises:
        (pyslurm.RPCError): When setting the debug flags was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.add_debug_flags(["CpuFrequency", "Backfill"])
    """
    if not flags:
        return

    data = _debug_flags_str_to_int(flags)
    if not data:
        raise RPCError(msg="Invalid Debug Flags specified.")

    verify_rpc(slurm_set_debugflags(data, 0))


def remove_debug_flags(flags):
    """Remove DebugFlags from `slurmctld`.

    Args:
        flags (list[str]):
            For an available list of possible values, please check the
            `slurm.conf` documentation under `DebugFlags`.

    Raises:
        (pyslurm.RPCError): When removing the debug flags was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.remove_debug_flags(["CpuFrequency"])
    """
    if not flags:
        return

    data = _debug_flags_str_to_int(flags)
    if not data:
        raise RPCError(msg="Invalid Debug Flags specified.")

    verify_rpc(slurm_set_debugflags(0, data))


def clear_debug_flags():
    """Remove all currently set debug flags from `slurmctld`.

    Raises:
        (pyslurm.RPCError): When removing the debug flags was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.clear_debug_flags()
        >>> print(slurmctld.get_debug_flags())
        []
    """
    current_flags = get_debug_flags()
    if not current_flags:
        return

    data = _debug_flags_str_to_int(current_flags)
    verify_rpc(slurm_set_debugflags(0, data))


def get_debug_flags():
    """Get the current list of debug flags for the `slurmctld`.

    Raises:
        (pyslurm.RPCError): When getting the debug flags was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> flags = slurmctld.get_debug_flags()
        >>> print(flags)
        ['CpuFrequency', 'Backfill']
    """
    return Config.load().debug_flags


def set_log_level(level):
    """Set the logging level for `slurmctld`.

    Args:
        level (str):
            For an available list of possible values, please check the
            `slurm.conf` documentation under `SlurmctldDebug`.

    Raises:
        (pyslurm.RPCError): When setting the log level was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.set_log_level("quiet")
        >>> log_level = slurmctld.get_log_level()
        >>> print(log_level)
        quiet
    """
    data = _log_level_str_to_int(level)
    verify_rpc(slurm_set_debug_level(data))


def get_log_level():
    """Get the current log level for the `slurmctld`.

    Raises:
        (pyslurm.RPCError): When getting the log level was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> log_level = slurmctld.get_log_level()
        >>> print(log_level)
        quiet
    """
    return Config.load().slurmctld_log_level


def enable_scheduler_logging():
    """Enable scheduler logging for `slurmctld`.

    Raises:
        (pyslurm.RPCError): When enabling scheduler logging was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.enable_scheduler_logging()
        >>> print(slurmctld.is_scheduler_logging_enabled())
        True
    """
    verify_rpc(slurm_set_schedlog_level(1))


def is_scheduler_logging_enabled():
    """Check whether scheduler logging is enabled for `slurmctld`.

    Returns:
       (bool): Whether scheduler logging is enabled or not.

    Raises:
        (pyslurm.RPCError): When getting the scheduler logging was not
            successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> print(slurmctld.is_scheduler_logging_enabled())
        False
    """
    return Config.load().scheduler_logging_enabled


def set_fair_share_dampening_factor(factor):
    """Set the FairShare Dampening factor.

    Args:
        factor (int):
            The factor to set. A minimum value of `1`, and a maximum value of
            `65535` are allowed.

    Raises:
        (pyslurm.RPCError): When setting the factor was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> slurmctld.set_fair_share_dampening_factor(100)
        >>> print(slurmctld.get_fair_share_dampening_factor)
        100
    """
    max_value = (2 ** 16) - 1
    if not factor or factor >= max_value:
        raise RPCError(msg=f"Invalid Dampening factor: {factor}. "
                           f"Factor must be between 0 and {max_value}.")

    verify_rpc(slurm_set_fs_dampeningfactor(factor))


def get_fair_share_dampening_factor():
    """Get the currently set FairShare Dampening factor.

    Raises:
        (pyslurm.RPCError): When getting the factor was not successful.

    Examples:
        >>> from pyslurm import slurmctld
        >>> factor = slurmctld.get_fair_share_dampening_factor()
        >>> print(factor)
        100
    """
    return Config.load().fair_share_dampening_factor


def _debug_flags_str_to_int(flags):
    cdef:
        uint64_t flags_num = 0
        char *flags_str = NULL

    flags_str = cstr.from_unicode(cstr.list_to_str(flags))
    slurm.debug_str2flags(flags_str, &flags_num)
    return flags_num


def _log_level_str_to_int(level):
    cdef uint16_t data = slurm.log_string2num(str(level))
    if u16_parse(data, zero_is_noval=False) is None:
        raise RPCError(msg=f"Invalid Log level: {level}.")

    return data

