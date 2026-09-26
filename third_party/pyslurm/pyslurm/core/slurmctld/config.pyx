#########################################################################
# slurmctld/config.pyx - pyslurm slurmctld config api
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
from pyslurm.utils.uint import (
    u16_parse,
    u32_parse,
    u64_parse,
)
from pyslurm.constants import UNLIMITED
from pyslurm.utils.ctime import _raw_time
from pyslurm.utils.helpers import (
    cpu_freq_int_to_str,
    instance_to_dict,
)
from pyslurm.utils import cstr
from typing import Union
import re


cdef class MPIConfig:

    def __init__(self):
        raise RuntimeError("Cannot instantiate class directly. "
                           "Use slurmctld.Config.load() and access the "
                           "mpi_config attribute there")

    def to_dict(self, recursive = False):
        """MPI config formatted as a dictionary.

        Returns:
            (dict): Config as a dict

        Examples:
            >>> from pyslurm import slurmctld
            >>> config = slurmctld.Config.load()
            >>> mpi_config = config.mpi_config.to_dict()
        """
        return instance_to_dict(self, recursive)

    @staticmethod
    cdef MPIConfig from_ptr(void *ptr):
        cdef:
            dict conf = _parse_config_key_pairs(ptr)
            MPIConfig out = MPIConfig.__new__(MPIConfig)

        out.pmix_cli_tmp_dir_base = conf.get("PMIxCliTmpDirBase")
        out.pmix_coll_fence = conf.get("PMIxCollFence")
        out.pmix_debug = bool(int(conf.get("PMIxDebug", 0)))
        out.pmix_direct_conn = _true_false_to_bool(conf.get("PMIxDirectConn", "true"))
        out.pmix_direct_conn_early = _true_false_to_bool(conf.get("PMIxDirectConnEarly", "false"))
        out.pmix_direct_conn_ucx = _true_false_to_bool(conf.get("PMIxDirectConnUCX", "false"))
        out.pmix_direct_same_arch = _true_false_to_bool(conf.get("PMIxDirectSameArch", "false"))
        out.pmix_environment = cstr.to_dict(
                conf.get("PMIxEnv", ""), delim1=";", delim2="=")
        out.pmix_fence_barrier = _true_false_to_bool(conf.get("PMIxFenceBarrier", "false"))
        out.pmix_net_devices_ucx = conf.get("PMIxNetDevicesUCX")
        out.pmix_timeout = int(_get_digits(conf.get("PMIxTimeout", "300")))
        out.pmix_tls_ucx = cstr.to_list(conf.get("PMIxTlsUCX", ""))

        return out


cdef class CgroupConfig:

    def __init__(self):
        raise RuntimeError("Cannot instantiate class directly. "
                           "Use slurmctld.Config.load() and access the "
                           "cgroup_config attribute there")

    def to_dict(self, recursive = False):
        """Cgroup config formatted as a dictionary.

        Returns:
            (dict): Config as a dict

        Examples:
            >>> from pyslurm import slurmctld
            >>> config = slurmctld.Config.load()
            >>> cgroup_config = config.cgroup_config.to_dict()
        """
        return instance_to_dict(self, recursive)

    @staticmethod
    cdef CgroupConfig from_ptr(void *ptr):
        cdef:
            dict conf = _parse_config_key_pairs(ptr)
            CgroupConfig out = CgroupConfig.__new__(CgroupConfig)

        out.mountpoint = conf.get("CgroupMountpoint", "/sys/fs/cgroup")
        out.plugin = conf.get("CgroupPlugin", "autodetect")
        out.systemd_timeout = int(_get_digits(conf.get("SystemdTimeout", "1000")))
        out.ignore_systemd = _yesno_to_bool(conf.get("IgnoreSystemd"))
        out.ignore_systemd_on_failure = _yesno_to_bool(conf.get("IgnoreSystemdOnFailure"))
        out.enable_controllers = _yesno_to_bool(conf.get("EnableControllers"))

        out.allowed_ram_space = float(_get_digits(conf.get("AllowedRAMSpace", "100.0%")))
        out.allowed_swap_space = float(_get_digits(conf.get("AllowedSwapSpace", "0.0%")))
        out.constrain_cores = _yesno_to_bool(conf.get("ConstrainCores", "no"))
        out.constrain_devices = _yesno_to_bool(conf.get("ConstrainDevices", "no"))
        out.constrain_ram_space = _yesno_to_bool(conf.get("ConstrainRAMSpace", "no"))
        out.constrain_swap_space = _yesno_to_bool(conf.get("ConstrainSwapSpace", "no"))
        out.max_ram_percent = float(_get_digits(conf.get("MaxRAMPercent", "100.0%")))
        out.max_swap_percent = float(_get_digits(conf.get("MaxSwapPercent", "100.0%")))
        out.memory_swappiness = float(conf.get("MemorySwappiness", -1.0))
        out.min_ram_space = int(_get_digits(conf.get("MinRAMSpace", "30MB")))

        out.signal_children_processes = _yesno_to_bool(conf.get("SignalChildrenProcesses", "no"))

        return out


cdef class AccountingGatherConfig:

    def __init__(self):
        raise RuntimeError("Cannot instantiate class directly. "
                           "Use slurmctld.Config.load() and access the "
                           "accounting_gather_config attribute there")

    def to_dict(self, recursive = False):
        """AccountingGather config formatted as a dictionary.

        Returns:
            (dict): Config as a dict

        Examples:
            >>> from pyslurm import slurmctld
            >>> config = slurmctld.Config.load()
            >>> acctg_config_dict = config.accounting_gather_config.to_dict()
        """
        return instance_to_dict(self, recursive)

    @staticmethod
    cdef AccountingGatherConfig from_ptr(void *ptr):
        cdef:
            dict conf = _parse_config_key_pairs(ptr)
            AccountingGatherConfig out = AccountingGatherConfig.__new__(AccountingGatherConfig)

        out.energy_ipmi_frequency = int(conf.get("EnergyIPMIFrequency", 30))
        out.energy_ipmi_calc_adjustment = _yesno_to_bool(
                conf.get("EnergyIPMICalcAdjustment"))
        out.energy_ipmi_power_sensors = cstr.to_dict(
                conf.get("EnergyIPMIPowerSensors", ""), delim1=";", delim2="=")
        out.energy_ipmi_user_name = conf.get("EnergyIPMIUsername")
        out.energy_ipmi_password = conf.get("EnergyIPMIPassword")
        out.energy_ipmi_timeout = int(conf.get("EnergyIPMITimeout", 10))

        out.profile_hdf5_dir = conf.get("ProfileHDF5Dir")
        out.profile_hdf5_default = conf.get("ProfileHDF5Default", "").split(",")

        out.profile_influxdb_database = conf.get("ProfileInfluxDBDatabase")
        out.profile_influxdb_default = conf.get("ProfileInfluxDBDefault", "").split(",")
        out.profile_influxdb_host = conf.get("ProfileInfluxDBHost")
        out.profile_influxdb_password = conf.get("ProfileInfluxDBPass")
        out.profile_influxdb_rtpolicy = conf.get("ProfileInfluxDBRTPolicy")
        out.profile_influxdb_user = conf.get("ProfileInfluxDBUser")
        out.profile_influxdb_timeout = int(conf.get("ProfileInfluxDBTimeout", 10))

        out.infiniband_ofed_port = int(conf.get("InfinibandOFEDPort", 1))
        out.sysfs_interfaces = conf.get("SysfsInterfaces", [])

        return out


cdef class Config:

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self):
        raise RuntimeError("Cannot instantiate class directly. "
                           "Use slurmctld.Config.load() to get an instance.")

    def __dealloc__(self):
        slurm_free_ctl_conf(self.ptr)
        self.ptr = NULL

    @staticmethod
    def load():
        """Load the current Slurm configuration (slurm.conf)

        This also loads the following other configurations:
            * `cgroup.conf` (`cgroup_config`)
            * `acct_gather.conf` (`accounting_gather_config`)
            * `mpi.conf` (`mpi_config`)
        """
        cdef Config conf = Config.__new__(Config)
        verify_rpc(slurm_load_ctl_conf(0, &conf.ptr))
        # TODO: node_features_conf
        return conf

    def to_dict(self, recursive = False):
        """Slurmctld config formatted as a dictionary.

        Returns:
            (dict): slurmctld config as a dict

        Examples:
            >>> import pyslurm
            >>> config = pyslurm.slurmctld.Config.load()
            >>> config_dict = config.as_dict()
        """
        out = instance_to_dict(self, recursive)
        return out

    @property
    def cgroup_config(self):
        # TODO: should these be none if there is actually no config?
        if not self._cgroup_config:
            self._cgroup_config =  CgroupConfig.from_ptr(self.ptr.cgroup_conf)
        return self._cgroup_config

    @property
    def accounting_gather_config(self):
        if not self._accounting_gather_config:
            self._accounting_gather_config = AccountingGatherConfig.from_ptr(
                self.ptr.acct_gather_conf)
        return self._accounting_gather_config

    @property
    def mpi_config(self):
        if not self._mpi_config:
            self._mpi_config = MPIConfig.from_ptr(self.ptr.mpi_conf)
        return self._mpi_config

    @property
    def accounting_storage_enforce(self):
        cdef char tmp[128]
        slurm_accounting_enforce_string(self.ptr.accounting_storage_enforce,
                                        tmp, sizeof(tmp))
        out = cstr.to_unicode(tmp)
        if not out or out == "none":
            return []

        return out.upper().split(",")

    @property
    def accounting_storage_backup_host(self):
        return cstr.to_unicode(self.ptr.accounting_storage_backup_host)

    @property
    def accounting_storage_external_hosts(self):
        return cstr.to_list(self.ptr.accounting_storage_ext_host)

    @property
    def accounting_storage_host(self):
        return cstr.to_unicode(self.ptr.accounting_storage_host)

    @property
    def accounting_storage_parameters(self):
        return cstr.to_dict(self.ptr.accounting_storage_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def accounting_storage_port(self):
        return u16_parse(self.ptr.accounting_storage_port)

    @property
    def accounting_storage_tres(self):
        return cstr.to_list(self.ptr.accounting_storage_tres)

    @property
    def accounting_storage_type(self):
        return cstr.to_unicode(self.ptr.accounting_storage_type)

    @property
    def accounting_store_flags(self):
        out = []
        flags = self.ptr.conf_flags

        if flags & slurm.CONF_FLAG_SJC:
            out.append("JOB_COMMENT")
        if flags & slurm.CONF_FLAG_SJE:
            out.append("JOB_ENV")
        if flags & slurm.CONF_FLAG_SJX:
            out.append("JOB_EXTRA")
        if flags & slurm.CONF_FLAG_SJS:
            out.append("JOB_SCRIPT")
        if flags & slurm.CONF_FLAG_NO_STDIO:
            out.append("NO_STDIO")

        return out

    @property
    def accounting_gather_node_frequency(self):
        return u16_parse(self.ptr.acct_gather_node_freq)

    @property
    def accounting_gather_energy_type(self):
        return cstr.to_unicode(self.ptr.acct_gather_energy_type)

    @property
    def accounting_gather_interconnect_type(self):
        return cstr.to_unicode(self.ptr.acct_gather_interconnect_type)

    @property
    def accounting_gather_filesystem_type(self):
        return cstr.to_unicode(self.ptr.acct_gather_filesystem_type)

    @property
    def accounting_gather_profile_type(self):
        return cstr.to_unicode(self.ptr.acct_gather_profile_type)

    @property
    def allow_spec_resource_usage(self):
        if self.ptr.conf_flags & slurm.CONF_FLAG_ASRU:
            return True

        return False

    @property
    def auth_alt_types(self):
        return cstr.to_list(self.ptr.authalttypes)

    @property
    def auth_info(self):
        return cstr.to_list(self.ptr.authinfo)

    @property
    def auth_alt_parameters(self):
        return cstr.to_dict(self.ptr.authalt_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def auth_type(self):
        return cstr.to_unicode(self.ptr.authtype)

    @property
    def batch_start_timeout(self):
        return u16_parse(self.ptr.batch_start_timeout)

    @property
    def bcast_exclude_paths(self):
        return cstr.to_list(self.ptr.bcast_exclude)

    @property
    def bcast_parameters(self):
        return cstr.to_dict(self.ptr.authalt_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def burst_buffer_type(self):
        return cstr.to_unicode(self.ptr.bb_type)

    @property
    def slurmctld_boot_time(self):
        return _raw_time(self.ptr.boot_time)

    @property
    def certmgr_parameters(self):
        return cstr.to_list(self.ptr.certmgr_params)

    @property
    def certmgr_type(self):
        return cstr.to_unicode(self.ptr.certmgr_type)

    @property
    def cli_filter_plugins(self):
        return cstr.to_list(self.ptr.cli_filter_plugins)

    @property
    def cluster_name(self):
        return cstr.to_unicode(self.ptr.cluster_name)

    @property
    def communication_parameters(self):
        return cstr.to_dict(self.ptr.comm_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def complete_wait_time(self):
        return u16_parse(self.ptr.complete_wait)

    @property
    def default_cpu_frequency_governor(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_def)

    @property
    def cpu_frequency_governors(self):
        return cpu_freq_int_to_str(self.ptr.cpu_freq_govs)

    @property
    def credential_type(self):
        return cstr.to_unicode(self.ptr.cred_type)

    @property
    def data_parser_parameters(self):
        return cstr.to_unicode(self.ptr.data_parser_parameters)

    @property
    def debug_flags(self):
        cdef char *data = slurm.debug_flags2str(self.ptr.debug_flags)
        return cstr.to_list_free(&data)

    @property
    def default_memory_per_cpu(self):
        return _get_memory(self.ptr.def_mem_per_cpu, per_cpu=True)

    @property
    def default_memory_per_node(self):
        return _get_memory(self.ptr.def_mem_per_cpu, per_cpu=False)

    # TODO: DefCpuPerGPU
    # TODO: DefMemPerGPU

    @property
    def dependency_parameters(self):
        return cstr.to_dict(self.ptr.dependency_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def disable_root_jobs(self):
        if self.ptr.conf_flags & slurm.CONF_FLAG_DRJ:
            return True
        return False

    @property
    def eio_timeout(self):
        return u16_parse(self.ptr.eio_timeout)

    @property
    def enforce_partition_limits(self):
        cdef char* data = slurm.parse_part_enforce_type_2str(
                self.ptr.enforce_part_limits)
        return cstr.to_unicode(data)

    @property
    def epilog(self):
        return cstr.to_list_with_count(self.ptr.epilog,
                                       self.ptr.epilog_cnt)

    @property
    def epilog_timeout(self):
        return u16_parse(self.ptr.epilog_timeout)

    @property
    def epilog_msg_time(self):
        return u32_parse(self.ptr.epilog_msg_time)

    @property
    def epilog_slurmctld(self):
        return cstr.to_list_with_count(self.ptr.epilog_slurmctld,
                                       self.ptr.epilog_slurmctld_cnt)

    @property
    def fair_share_dampening_factor(self):
        return u16_parse(self.ptr.fs_dampening_factor)

    @property
    def federation_parameters(self):
        return cstr.to_list(self.ptr.fed_params)

    @property
    def first_job_id(self):
        return u32_parse(self.ptr.first_job_id)

    @property
    def gres_types(self):
        return cstr.to_list(self.ptr.gres_plugins)

    @property
    def group_update_force(self):
        return u16_parse_bool(self.ptr.group_force)

    @property
    def group_update_time(self):
        return u16_parse(self.ptr.group_time)

    @property
    def default_gpu_frequency(self):
        return cstr.to_unicode(self.ptr.gpu_freq_def)

    @property
    def hash_plugin(self):
        return cstr.to_unicode(self.ptr.hash_plugin)

    @property
    def hash_value(self):
        val = u32_parse(self.ptr.hash_val)
        if not val:
            return None
        return hex(val)

    @property
    def health_check_interval(self):
        return u16_parse(self.ptr.health_check_interval)

    @property
    def health_check_node_state(self):
        cdef char *data = slurm.health_check_node_state_str(
                self.ptr.health_check_node_state)
        return cstr.to_list_free(&data)

    @property
    def health_check_program(self):
        return cstr.to_unicode(self.ptr.health_check_program)

    @property
    def inactive_limit(self):
        return u16_parse(self.ptr.inactive_limit)

    @property
    def interactive_step_options(self):
        return cstr.to_unicode(self.ptr.interactive_step_opts)

    @property
    def job_accounting_gather_type(self):
        return cstr.to_unicode(self.ptr.job_acct_gather_type)

    @property
    def job_accounting_gather_frequency(self):
        return cstr.to_dict(self.ptr.job_acct_gather_freq)

    @property
    def job_accounting_gather_parameters(self):
        return cstr.to_list(self.ptr.job_acct_gather_params)

    @property
    def job_completion_host(self):
        return cstr.to_unicode(self.ptr.job_comp_host)

    @property
    def job_completion_location(self):
        return cstr.to_unicode(self.ptr.job_comp_loc)

    @property
    def job_completion_parameters(self):
        return cstr.to_dict(self.ptr.job_comp_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def job_completion_port(self):
        return u32_parse(self.ptr.job_comp_port)

    @property
    def job_completion_type(self):
        return cstr.to_unicode(self.ptr.job_comp_type)

    @property
    def job_completion_user(self):
        return cstr.to_unicode(self.ptr.job_comp_user)

    @property
    def namespace_plugin(self):
        return cstr.to_unicode(self.ptr.namespace_plugin)

    @property
    def job_defaults(self):
        cdef char *data = slurm.job_defaults_str(self.ptr.job_defaults_list)
        out = cstr.to_dict(data)
        xfree(data)
        return out

    @property
    def job_file_append(self):
        return u16_parse_bool(self.ptr.job_file_append)

    @property
    def job_requeue(self):
        return u16_parse_bool(self.ptr.job_requeue)

    @property
    def job_submit_plugins(self):
        return cstr.to_list(self.ptr.job_submit_plugins)

    @property
    def kill_on_bad_exit(self):
        return u16_parse_bool(self.ptr.kill_on_bad_exit)

    @property
    def kill_wait_time(self):
        return u16_parse(self.ptr.kill_wait)

    @property
    def launch_parameters(self):
        return cstr.to_list(self.ptr.launch_params)

    @property
    def licenses(self):
        return cstr.to_dict(self.ptr.licenses, delim1=",",
                            delim2=":", def_value=1)

    @property
    def log_time_format(self):
        flag = self.ptr.log_fmt
        if flag == slurm.LOG_FMT_ISO8601_MS:
            return "iso8601_ms"
        elif flag == slurm.LOG_FMT_ISO8601:
            return "iso8601"
        elif flag == slurm.LOG_FMT_RFC5424_MS:
            return "rfc5424_ms"
        elif flag == slurm.LOG_FMT_RFC5424:
            return "rfc5424"
        elif flag == slurm.LOG_FMT_CLOCK:
            return "clock"
        elif flag == slurm.LOG_FMT_SHORT:
            return "short"
        elif flag == slurm.LOG_FMT_THREAD_ID:
            return "thread_id"
        elif flag == slurm.LOG_FMT_RFC3339:
            return "rfc3339"
        else:
            return None

    @property
    def mail_domain(self):
        return cstr.to_unicode(self.ptr.mail_domain)

    @property
    def mail_program(self):
        return cstr.to_unicode(self.ptr.mail_prog)

    @property
    def max_array_size(self):
        return u32_parse(self.ptr.max_array_sz)

    @property
    def max_batch_requeue(self):
        return u32_parse(self.ptr.max_batch_requeue)

    @property
    def max_dbd_msgs(self):
        return u32_parse(self.ptr.max_dbd_msgs)

    @property
    def max_job_count(self):
        return u32_parse(self.ptr.max_job_cnt)

    @property
    def max_job_id(self):
        return u32_parse(self.ptr.max_job_id)

    @property
    def max_memory_per_cpu(self):
        return _get_memory(self.ptr.max_mem_per_cpu, per_cpu=True)

    @property
    def max_memory_per_node(self):
        return _get_memory(self.ptr.max_mem_per_cpu, per_cpu=False)

    @property
    def max_node_count(self):
        return u32_parse(self.ptr.max_node_cnt)

    @property
    def max_step_count(self):
        return u32_parse(self.ptr.max_step_cnt)

    @property
    def max_tasks_per_node(self):
        return u32_parse(self.ptr.max_tasks_per_node)

    @property
    def mcs_plugin(self):
        return cstr.to_unicode(self.ptr.mcs_plugin)

    @property
    def mcs_parameters(self):
        return cstr.to_list(self.ptr.mcs_plugin_params)

    @property
    def metrics_type(self):
        return cstr.to_unicode(self.ptr.metrics_type)

    @property
    def min_job_age(self):
        return u32_parse(self.ptr.min_job_age)

    @property
    def mpi_default(self):
        return cstr.to_unicode(self.ptr.mpi_default)

    @property
    def mpi_parameters(self):
        return cstr.to_dict(self.ptr.mpi_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def message_timeout(self):
        return u16_parse(self.ptr.msg_timeout)

    @property
    def next_job_id(self):
        return u32_parse(self.ptr.next_job_id)

    @property
    def node_features_plugins(self):
        return cstr.to_list(self.ptr.node_features_plugins)

    @property
    def over_time_limit(self):
        return u16_parse(self.ptr.over_time_limit)

    @property
    def plugin_dirs(self):
        return cstr.to_list(self.ptr.plugindir, default=None, delim=":")

    @property
    def plugin_stack_config(self):
        return cstr.to_unicode(self.ptr.plugstack)

    @property
    def preempt_exempt_time(self):
        # seconds?
        return _raw_time(self.ptr.preempt_exempt_time)

    @property
    def preempt_mode(self):
        cdef char *tmp = slurm_preempt_mode_string(self.ptr.preempt_mode)
        return cstr.to_unicode(tmp)

    @property
    def preempt_parameters(self):
        return cstr.to_dict(self.ptr.preempt_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def preempt_type(self):
        return cstr.to_unicode(self.ptr.preempt_type)

    @property
    def prep_parameters(self):
        return cstr.to_list(self.ptr.prep_params)

    @property
    def prep_plugins(self):
        return cstr.to_list(self.ptr.prep_plugins)

    @property
    def priority_decay_half_life(self):
        return u32_parse(self.ptr.priority_decay_hl)

    @property
    def priority_calc_period(self):
        return u32_parse(self.ptr.priority_calc_period)

    @property
    def priority_favor_small(self):
        return u16_parse_bool(self.ptr.priority_favor_small)

    @property
    def priority_flags(self):
        cdef char *data = slurm.priority_flags_string(self.ptr.priority_flags)
        return cstr.to_list_free(&data)

    @property
    def priority_max_age(self):
        return u32_parse(self.ptr.priority_max_age)

    @property
    def priority_parameters(self):
        return cstr.to_unicode(self.ptr.priority_params)

    @property
    def priority_usage_reset_period(self):
        flag = self.ptr.priority_reset_period
        if flag == slurm.PRIORITY_RESET_NONE:
            return None
        elif flag == slurm.PRIORITY_RESET_NOW:
            return "NOW"
        elif flag == slurm.PRIORITY_RESET_DAILY:
            return "DAILY"
        elif flag == slurm.PRIORITY_RESET_WEEKLY:
            return "WEEKLY"
        elif flag == slurm.PRIORITY_RESET_MONTHLY:
            return "MONTHLY"
        elif flag == slurm.PRIORITY_RESET_QUARTERLY:
            return "QUARTERLY"
        elif flag == slurm.PRIORITY_RESET_YEARLY:
            return "YEARLY"
        else:
            return None

    @property
    def priority_type(self):
        return cstr.to_unicode(self.ptr.priority_type)

    @property
    def priority_weight_age(self):
        return u32_parse(self.ptr.priority_weight_age, zero_is_noval=False)

    @property
    def priority_weight_assoc(self):
        return u32_parse(self.ptr.priority_weight_assoc, zero_is_noval=False)

    @property
    def priority_weight_fair_share(self):
        return u32_parse(self.ptr.priority_weight_fs, zero_is_noval=False)

    @property
    def priority_weight_job_size(self):
        return u32_parse(self.ptr.priority_weight_js, zero_is_noval=False)

    @property
    def priority_weight_partition(self):
        return u32_parse(self.ptr.priority_weight_part, zero_is_noval=False)

    @property
    def priority_weight_qos(self):
        return u32_parse(self.ptr.priority_weight_qos, zero_is_noval=False)

    @property
    def priority_weight_tres(self):
        return cstr.to_dict(self.ptr.priority_weight_tres)

    @property
    def private_data(self):
        cdef char tmp[128]
        slurm.private_data_string(self.ptr.private_data, tmp, sizeof(tmp))
        out = cstr.to_unicode(tmp)
        if not out or out == "none":
            return []

        return out.split(",")

    @property
    def proctrack_type(self):
        return cstr.to_unicode(self.ptr.proctrack_type)

    @property
    def prolog(self):
        return cstr.to_list_with_count(self.ptr.prolog,
                                       self.ptr.prolog_cnt)

    @property
    def prolog_timeout(self):
        return u16_parse(self.ptr.prolog_timeout)

    @property
    def prolog_slurmctld(self):
        return cstr.to_list_with_count(self.ptr.prolog_slurmctld,
                                       self.ptr.prolog_slurmctld_cnt)

    @property
    def propagate_prio_process(self):
        return u16_parse(self.ptr.propagate_prio_process, zero_is_noval=False)

    @property
    def prolog_flags(self):
        cdef char *data = slurm.prolog_flags2str(self.ptr.prolog_flags)
        return cstr.to_list_free(&data)

    @property
    def propagate_resource_limits(self):
        return cstr.to_list(self.ptr.propagate_rlimits)

    @property
    def propagate_resource_limits_except(self):
        return cstr.to_list(self.ptr.propagate_rlimits_except)

    @property
    def reboot_program(self):
        return cstr.to_unicode(self.ptr.reboot_program)

    @property
    def reconfig_flags(self):
        cdef char *tmp = slurm.reconfig_flags2str(self.ptr.reconfig_flags)
        return cstr.to_list_free(&tmp)

    @property
    def requeue_exit(self):
        return cstr.to_unicode(self.ptr.requeue_exit)

    @property
    def requeue_exit_hold(self):
        return cstr.to_unicode(self.ptr.requeue_exit_hold)

    @property
    def resume_fail_program(self):
        return cstr.to_unicode(self.ptr.resume_fail_program)

    @property
    def resume_program(self):
        return cstr.to_unicode(self.ptr.resume_program)

    @property
    def resume_rate(self):
        return u16_parse(self.ptr.resume_rate)

    @property
    def resume_timeout(self):
        return u16_parse(self.ptr.resume_timeout)

    @property
    def reservation_epilog(self):
        return cstr.to_unicode(self.ptr.resv_epilog)

    @property
    def reservation_over_run(self):
        return u16_parse(self.ptr.resv_over_run, zero_is_noval=False)

    @property
    def reservation_prolog(self):
        return cstr.to_unicode(self.ptr.resv_prolog)

    @property
    def return_to_service(self):
        return u16_parse(self.ptr.ret2service, zero_is_noval=False)

    @property
    def scheduler_log_file(self):
        return cstr.to_unicode(self.ptr.sched_logfile)

    @property
    def scheduler_logging_enabled(self):
        return u16_parse_bool(self.ptr.sched_log_level)

    @property
    def scheduler_parameters(self):
        return cstr.to_dict(self.ptr.sched_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def scheduler_time_slice(self):
        return u16_parse(self.ptr.sched_time_slice)

    @property
    def scheduler_type(self):
        return cstr.to_unicode(self.ptr.schedtype)

    @property
    def scron_parameters(self):
        return cstr.to_list(self.ptr.scron_params)

    @property
    def select_type(self):
        return cstr.to_unicode(self.ptr.select_type)

    @property
    def select_type_parameters(self):
        cdef char *tmp = slurm.select_type_param_string(self.ptr.select_type_param)
        return cstr.to_list(tmp)

    @property
    def priority_site_factor_plugin(self):
        return cstr.to_unicode(self.ptr.site_factor_plugin)

    @property
    def priority_site_factor_parameters(self):
        return cstr.to_unicode(self.ptr.site_factor_params)

    @property
    def slurm_conf_path(self):
        return cstr.to_unicode(self.ptr.slurm_conf)

    @property
    def slurm_user_id(self):
        return self.ptr.slurm_user_id

    @property
    def slurm_user_name(self):
        return cstr.to_unicode(self.ptr.slurm_user_name)

    @property
    def slurmd_user_id(self):
        return self.ptr.slurm_user_id

    @property
    def slurmd_user_name(self):
        return cstr.to_unicode(self.ptr.slurmd_user_name)

    @property
    def slurmctld_log_level(self):
        return _log_level_int_to_str(self.ptr.slurmctld_debug)

    @property
    def slurmctld_log_file(self):
        return cstr.to_unicode(self.ptr.slurmctld_logfile)

    @property
    def slurmctld_pid_file(self):
        return cstr.to_unicode(self.ptr.slurmctld_pidfile)

    @property
    def slurmctld_port(self):
        port = self.ptr.slurmctld_port
        if self.ptr.slurmctld_port_count > 1:
            # Slurmctld port can be a range actually, calculated by using the
            # number of ports in use that slurm conf reports for slurmctld
            last_port = port + self.ptr.slurmctld_port_count - 1
            port = f"{port}-{last_port}"

        return str(port)

    @property
    def slurmctld_primary_off_program(self):
        return cstr.to_unicode(self.ptr.slurmctld_primary_off_prog)

    @property
    def slurmctld_primary_on_program(self):
        return cstr.to_unicode(self.ptr.slurmctld_primary_on_prog)

    @property
    def slurmctld_syslog_level(self):
        return _log_level_int_to_str(self.ptr.slurmctld_syslog_debug)

    @property
    def slurmctld_timeout(self):
        return u16_parse(self.ptr.slurmctld_timeout)

    @property
    def slurmctld_parameters(self):
        return cstr.to_dict(self.ptr.slurmctld_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def slurmd_log_level(self):
        return _log_level_int_to_str(self.ptr.slurmd_debug)

    @property
    def slurmd_log_file(self):
        return cstr.to_unicode(self.ptr.slurmd_logfile)

    @property
    def slurmd_parameters(self):
        return cstr.to_dict(self.ptr.slurmd_params, delim1=",",
                            delim2="=", def_value=True)

    @property
    def slurmd_pid_file(self):
        return cstr.to_unicode(self.ptr.slurmd_pidfile)

    @property
    def slurmd_port(self):
        return self.ptr.slurmd_port

    @property
    def slurmd_spool_directory(self):
        return cstr.to_unicode(self.ptr.slurmd_spooldir)

    @property
    def slurmd_syslog_level(self):
        return _log_level_int_to_str(self.ptr.slurmd_syslog_debug)

    @property
    def slurmd_timeout(self):
        return u16_parse(self.ptr.slurmd_timeout)

    @property
    def srun_epilog(self):
        return cstr.to_unicode(self.ptr.srun_epilog)

    @property
    def srun_port_range(self):
        if not self.ptr.srun_port_range:
            return None

        low = self.ptr.srun_port_range[0]
        high = self.ptr.srun_port_range[1]
        return f"{low}-{high}"

    @property
    def srun_prolog(self):
        return cstr.to_unicode(self.ptr.srun_prolog)

    @property
    def state_save_location(self):
        return cstr.to_unicode(self.ptr.state_save_location)

    @property
    def suspend_exclude_nodes(self):
        return cstr.to_unicode(self.ptr.suspend_exc_nodes)

    @property
    def suspend_exclude_partitions(self):
        return cstr.to_list(self.ptr.suspend_exc_parts)

    @property
    def suspend_exclude_states(self):
        return cstr.to_list(self.ptr.suspend_exc_states)

    @property
    def suspend_program(self):
        return cstr.to_unicode(self.ptr.suspend_program)

    @property
    def suspend_rate(self):
        return u16_parse(self.ptr.suspend_rate)

    @property
    def suspend_time(self):
        return u32_parse(self.ptr.suspend_time)

    @property
    def suspend_timeout(self):
        return u16_parse(self.ptr.suspend_timeout)

    @property
    def switch_type(self):
        return cstr.to_unicode(self.ptr.switch_type)

    @property
    def switch_parameters(self):
        return cstr.to_dict(self.ptr.switch_param, delim1=",",
                            delim2="=", def_value=True)

    @property
    def task_epilog(self):
        return cstr.to_unicode(self.ptr.task_epilog)

    @property
    def task_plugin(self):
        return cstr.to_unicode(self.ptr.task_plugin)

    @property
    def task_plugin_parameters(self):
        cdef char cpu_bind[256]
        slurm_sprint_cpu_bind_type(cpu_bind,
                                   <cpu_bind_type_t>self.ptr.task_plugin_param)
        if cpu_bind == "(null type)":
            return []

        return cstr.to_list(cpu_bind)

    @property
    def task_prolog(self):
        return cstr.to_unicode(self.ptr.task_prolog)

    @property
    def tls_parameters(self):
        return cstr.to_list(self.ptr.tls_params)

    @property
    def tls_type(self):
        return cstr.to_unicode(self.ptr.tls_type)

    @property
    def tcp_timeout(self):
        return u16_parse(self.ptr.tcp_timeout)

    @property
    def temporary_filesystem(self):
        return cstr.to_unicode(self.ptr.tmp_fs)

    @property
    def topology_parameters(self):
        return cstr.to_dict(self.ptr.topology_param, delim1=",",
                            delim2="=", def_value=True)

    @property
    def topology_plugin(self):
        return cstr.to_unicode(self.ptr.topology_plugin)

    @property
    def tree_width(self):
        return u16_parse(self.ptr.tree_width)

    @property
    def unkillable_step_program(self):
        return cstr.to_unicode(self.ptr.unkillable_program)

    @property
    def unkillable_step_timeout(self):
        return u16_parse(self.ptr.unkillable_timeout)

    @property
    def track_wckey(self):
        if self.ptr.conf_flags & slurm.CONF_FLAG_WCKEY:
            return True
        return False

    @property
    def use_pam(self):
        if self.ptr.conf_flags & slurm.CONF_FLAG_PAM:
            return True
        return False

    @property
    def version(self):
        return cstr.to_unicode(self.ptr.version)

    @property
    def virtual_memory_size_factor(self):
        return u16_parse(self.ptr.vsize_factor)

    @property
    def wait_time(self):
        return u16_parse(self.ptr.wait_time)

    @property
    def x11_parameters(self):
        return cstr.to_list(self.ptr.x11_params)


cdef dict _parse_config_key_pairs(void *ptr, owned=False):
    cdef:
        SlurmList conf = SlurmList.wrap(<list_t*>ptr, owned=owned)
        SlurmListItem item
        config_key_pair_t *key_pair
        dict out = {}

    for item in conf:
        key_pair = <config_key_pair_t*>item.data
        name = cstr.to_unicode(key_pair.name)
        val = cstr.to_unicode(key_pair.value)

        if not val or val == "(null)":
            val = None

        out[name] = val

    return out


def _str_to_bool(val, true_str, false_str):
    if not val:
        return False

    v = val.lower()
    if v == true_str:
        return True
    elif v == false_str:
        return False
    else:
        return False


def _get_digits(val):
    return re.split(r'\D+', val)[0]


def _yesno_to_bool(val):
    return _str_to_bool(val, "yes", "no")


def _true_false_to_bool(val):
    return _str_to_bool(val, "true", "false")


def _log_level_int_to_str(flags):
    data = cstr.to_unicode(slurm.log_num2string(flags))
    if data == "(null)":
        return None
    else:
        return data


def _get_memory(value, per_cpu):
    if value != slurm.NO_VAL64:
        if value & slurm.MEM_PER_CPU and per_cpu:
            if value == slurm.MEM_PER_CPU:
                return UNLIMITED
            return u64_parse(value & (~slurm.MEM_PER_CPU))

        # For these values, Slurm interprets 0 as being equal to
        # INFINITE/UNLIMITED
        elif value == 0 and not per_cpu:
            return UNLIMITED

        elif not value & slurm.MEM_PER_CPU and not per_cpu:
            return u64_parse(value)

    return None
