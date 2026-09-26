#########################################################################
# submission.pyx - interface for submitting slurm jobs
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

from os import getcwd
from os import environ as pyenviron
import re
from typing import Union, Any
import shlex
from pathlib import Path
from pyslurm.utils import cstr
from pyslurm.utils.uint import *
from pyslurm.core.job.util import *
from pyslurm.core.error import RPCError, verify_rpc
from pyslurm.core.job.sbatch_opts import _parse_opts_from_batch_script
from pyslurm.utils.ctime import (
    secs_to_timestr,
    timestr_to_secs,
    mins_to_timestr,
    timestr_to_mins,
    timestamp_to_date,
    date_to_timestamp,
)
from pyslurm.utils.helpers import (
    humanize,
    dehumanize,
    signal_to_num,
    user_to_uid,
    group_to_gid,
    uid_to_name,
    gid_to_name,
)


cdef class JobSubmitDescription:
    def __cinit__(self):
        self.ptr = NULL

    def __init__(self, **kwargs):
        # Initialize explicitly provided attributes, if any.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __dealloc__(self):
        slurm_free_job_desc_msg(self.ptr)

    def _alloc_and_init(self):
        slurm_free_job_desc_msg(self.ptr)

        self.ptr = <job_desc_msg_t*>try_xmalloc(sizeof(job_desc_msg_t))
        if not self.ptr:
            raise MemoryError("xmalloc for job_desc_msg_t failed.")

        slurm_init_job_desc_msg(self.ptr)

    def __repr__(self):
        return f'pyslurm.{self.__class__.__name__}'

    def submit(self):
        """Submit a batch job description.

        Returns:
            (int): The ID of the submitted Job.

        Raises:
            (pyslurm.RPCError): When the job submission was not successful.

        Examples:
            >>> import pyslurm
            >>> desc = pyslurm.JobSubmitDescription(
            ...     name="test-job",
            ...     cpus_per_task=1,
            ...     time_limit="10-00:00:00",
            ...     script="/path/to/your/submit_script.sh")
            >>>
            >>> job_id = desc.submit()
            >>> print(job_id)
            99
        """
        cdef submit_response_msg_t *resp = NULL

        self._create_job_submit_desc()
        verify_rpc(slurm_submit_batch_job(self.ptr, &resp))

        job_id = resp.step_id.job_id
        slurm_free_submit_response_response_msg(resp)

        return job_id

    def load_environment(self, overwrite=False):
        """Load values of attributes provided through the environment.

        !!! note

            Instead of `SBATCH_`, pyslurm uses `PYSLURM_JOBDESC_` as a prefix
            to identify environment variables which should be used to set
            attributes.

        Args:
            overwrite (bool):
                If set to `True`, the value from an option found in the
                environment will override the current value of the attribute
                in this instance. Default is `False`

        Examples:
            Lets consider you want to set the name of the Job, its Account
            name and that the Job cannot be requeued.
            Therefore, you will need to have set these environment variables:

            ```bash
            # Format is: PYSLURM_JOBDESC_{ATTRIBUTE_NAME}
            export PYSLURM_JOBDESC_ACCOUNT="myaccount"
            export PYSLURM_JOBDESC_NAME="myjobname"
            export PYSLURM_JOBDESC_IS_REQUEUEABLE="False"
            ```

            As you can see above, boolean values should be the literal strings
            "False" or "True".
            In python, you can do this now:

            >>> import pyslurm
            >>> desc = pyslurm.JobSubmitDescription(...other args...)
            >>> desc.load_environment()
            >>> print(desc.name, desc.account, desc.is_requeueable)
            myjobname, myaccount, False
        """
        self._parse_env(overwrite)

    def load_sbatch_options(self, overwrite=False):
        """Load values from `#SBATCH` options in the batch script.

        Args:
            overwrite (bool):
                If set to `True`, the value from an option found in the in the
                batch script will override the current value of the attribute
                in this instance. Default is `False`
        """
        if not self.script:
            raise ValueError("You need to set the 'script' attribute first.")
        _parse_opts_from_batch_script(self, self.script, overwrite)

    def _parse_env(self, overwrite=False):
        for attr in dir(self):
            if attr.startswith("_") or callable(attr):
                # Ignore everything starting with "_" and all functions.
                # Arguments directly specified upon object creation will
                # always have precedence.
                continue

            spec = attr.upper()
            val = pyenviron.get(f"PYSLURM_JOBDESC_{spec}")
            if (val is not None
                    and (getattr(self, attr) is None or overwrite)):

                # Just convert literal true/false strings to bool.
                tmp = val.casefold()
                if tmp == "true":
                    val = True
                elif tmp == "false":
                    val = False

                setattr(self, attr, val)

    def _create_job_submit_desc(self, is_update=False):
        self.is_update = is_update
        self._alloc_and_init()
        cdef slurm.job_desc_msg_t *ptr = self.ptr

        if not self.is_update:
            self._validate_options()
            self._set_defaults()

        if self.nice:
            ptr.nice = slurm.NICE_OFFSET + int(self.nice)

        if self.site_factor:
            ptr.site_factor = slurm.NICE_OFFSET + int(self.site_factor)

        if self.user_id is not None:
            ptr.user_id = user_to_uid(self.user_id)
        if self.group_id is not None:
            ptr.group_id = group_to_gid(self.group_id)

        cstr.fmalloc(&ptr.name, self.name)
        cstr.fmalloc(&ptr.account, self.account)
        cstr.fmalloc(&ptr.wckey, self.wckey)
        cstr.fmalloc(&ptr.array_inx, self.array)
        cstr.fmalloc(&ptr.batch_features, self.batch_constraints)
        cstr.fmalloc(&ptr.cluster_features, self.cluster_constraints)
        cstr.fmalloc(&ptr.comment, self.comment)
        cstr.fmalloc(&ptr.work_dir, self.working_directory)
        cstr.fmalloc(&ptr.features, self.constraints)
        cstr.fmalloc(&ptr.mail_user, self.mail_user)
        cstr.fmalloc(&ptr.mcs_label, self.mcs_label)
        cstr.fmalloc(&ptr.network, self.network)
        cstr.fmalloc(&ptr.qos, self.qos)
        cstr.fmalloc(&ptr.container, self.container)
        cstr.fmalloc(&ptr.container_id, self.container_id)
        cstr.fmalloc(&ptr.std_in, self.standard_in)
        cstr.fmalloc(&ptr.std_out, self.standard_output)
        cstr.fmalloc(&ptr.std_err, self.standard_error)
        cstr.fmalloc(&ptr.tres_per_job, cstr.from_gres_dict(self.gpus, "gpu"))
        cstr.fmalloc(&ptr.tres_per_socket,
                     cstr.from_gres_dict(self.gpus_per_socket, "gpu"))
        cstr.fmalloc(&ptr.tres_per_task,
                     cstr.from_gres_dict(self.gpus_per_task, "gpu"))
        cstr.fmalloc(&ptr.tres_per_node,
                     cstr.from_gres_dict(self.gres_per_node))
        cstr.fmalloc(&ptr.cpus_per_tres,
                     cstr.from_gres_dict(self.cpus_per_gpu, "gpu"))
        cstr.fmalloc(&ptr.admin_comment, self.admin_comment)
        cstr.fmalloc(&ptr.dependency,
                     _parse_dependencies(self.dependencies))
        cstr.fmalloc(&ptr.req_context, self.selinux_context)
        cstr.fmalloc(&ptr.extra, self.extra)
        cstr.from_list(&ptr.clusters, self.clusters)
        cstr.from_list(&ptr.exc_nodes, self.excluded_nodes)
        cstr.from_list(&ptr.req_nodes, self.required_nodes)
        cstr.from_list(&ptr.licenses, self.licenses)
        cstr.from_list(&ptr.partition, self.partitions)
        cstr.from_list(&ptr.reservation, self.reservations)
        cstr.from_dict(&ptr.acctg_freq, self.accounting_gather_frequency)
        ptr.deadline = date_to_timestamp(self.deadline)
        ptr.begin_time = date_to_timestamp(self.begin_time)
        ptr.delay_boot = timestr_to_secs(self.delay_boot_time)
        ptr.time_limit = timestr_to_mins(self.time_limit)
        ptr.time_min = timestr_to_mins(self.time_limit_min)
        ptr.priority = u32(self.priority, zero_is_noval=False)
        ptr.num_tasks = u32(self.ntasks)
        ptr.pn_min_tmp_disk = u32(dehumanize(self.temporary_disk_per_node))
        ptr.cpus_per_task = u16(self.cpus_per_task)
        ptr.sockets_per_node = u16(self.sockets_per_node)
        ptr.cores_per_socket = u16(self.cores_per_socket)
        ptr.ntasks_per_socket = u16(self.ntasks_per_socket)
        ptr.ntasks_per_tres = u16(self.ntasks_per_gpu)
        ptr.ntasks_per_node = u16(self.ntasks_per_node)
        ptr.threads_per_core = u16(self.threads_per_core)
        ptr.ntasks_per_core = u16(self.ntasks_per_core)
        ptr.segment_size = u16(self.segment_size)
        ptr.resv_port_cnt = u16(self.reserved_ports)
        u64_set_bool_flag(&ptr.bitflags, self.spreads_over_nodes,
                          slurm.SPREAD_JOB)
        u64_set_bool_flag(&ptr.bitflags, self.kill_on_invalid_dependency,
                          slurm.KILL_INV_DEP)
        u64_set_bool_flag(&ptr.bitflags, self.use_min_nodes,
                          slurm.USE_MIN_NODES)
        ptr.contiguous = u16_bool(self.requires_contiguous_nodes)
        ptr.kill_on_node_fail = u16_bool(self.kill_on_node_fail)
        ptr.overcommit = u8_bool(self.overcommit)
        ptr.reboot = u16_bool(self.requires_node_reboot)
        ptr.requeue = u16_bool(self.is_requeueable)
        ptr.wait_all_nodes = u16_bool(self.wait_all_nodes)
        ptr.mail_type = mail_type_list_to_int(self.mail_types)
        ptr.profile = acctg_profile_list_to_int(self.profile_types)
        ptr.shared = shared_type_str_to_int(self.resource_sharing)

        if not self.is_update:
            self.ptr.min_nodes, self.ptr.max_nodes = _parse_nodes(self.nodes)
            cstr.fmalloc(&self.ptr.script,
                         _validate_batch_script(self.script, self.script_args))
            self._set_script_args()
            self._set_environment()
            self._set_distribution()

        self._set_memory()
        self._set_open_mode()
        self._set_cpu_frequency()
        self._set_gpu_binding()
        self._set_gres_binding()
        self._set_gres_tasks_per_sharing()
        self._set_min_cpus()
        self._set_core_spec()
        self._set_signal()
        self._set_switches()

        # TODO
        # burst_buffer
        # mem_bind, mem_bind_type?
        # gpu_freq
        # --hint
        # spank_env
        # --propagate for rlimits

        # TODO: maybe pn_min_cpus needed?

    def _set_defaults(self):
        if not self.ntasks:
            self.ntasks = 1
        if not self.cpus_per_task:
            self.cpus_per_task = 1
        if not self.working_directory:
            self.working_directory = str(getcwd())
        if not self.environment:
            # By default, sbatch also exports everything in the users env.
            self.environment = "ALL"

    def _validate_options(self):
        if not self.script:
            raise ValueError("You need to provide a batch script.")

        if (self.memory_per_node and self.memory_per_cpu
                or self.memory_per_gpu and self.memory_per_cpu
                or self.memory_per_node and self.memory_per_gpu):
            raise ValueError("Only one of memory_per_cpu, memory_per_node or "
                             "memory_per_gpu can be set.")

        if (self.ntasks_per_gpu and
                (self.ptr.min_nodes != u32(None) or self.nodes
                or self.gpus_per_task or self.gpus_per_socket
                or self.ntasks_per_node)):
            raise ValueError("ntasks_per_gpu is mutually exclusive with "
                    "nodes, gpus_per_task, gpus_per_socket and "
                    "ntasks_per_node.")

        if self.cpus_per_gpu and self.cpus_per_task:
            raise ValueError("cpus_per_task and cpus_per_gpu "
                             "are mutually exclusive.")

        if (self.cores_reserved_for_system
                and self.threads_reserved_for_system):
            raise ValueError("cores_reserved_for_system is mutually "
                    " exclusive with threads_reserved_for_system.")

    def _set_core_spec(self):
        if self.cores_reserved_for_system:
            self.ptr.core_spec = u16(self.cores_reserved_for_system)
        elif self.threads_reserved_for_system:
            self.ptr.core_spec = u16(self.threads_reserved_for_system)
            self.ptr.core_spec |= slurm.CORE_SPEC_THREAD

    def _set_cpu_frequency(self):
        freq = self.cpu_frequency
        if not freq:
            return None

        # Alternatively support sbatch-like --cpu-freq setting.
        if not isinstance(freq, dict):
            freq = _parse_cpu_freq_str_to_dict(freq)

        freq_min, freq_max, freq_gov = _validate_cpu_freq(freq)
        self.ptr.cpu_freq_min = freq_min
        self.ptr.cpu_freq_max = freq_max
        self.ptr.cpu_freq_gov = freq_gov

    def _set_memory(self):
        if self.memory_per_cpu:
            self.ptr.pn_min_memory = u64(dehumanize(self.memory_per_cpu))
            self.ptr.pn_min_memory |= slurm.MEM_PER_CPU
        elif self.memory_per_node:
            self.ptr.pn_min_memory = u64(dehumanize(self.memory_per_node))
        elif self.memory_per_gpu:
            mem_gpu = u64(dehumanize(self.memory_per_gpu))
            cstr.fmalloc(&self.ptr.mem_per_tres, f"gres:gpu:{mem_gpu}")

    def _set_open_mode(self):
        val = self.log_files_open_mode
        if val == "append":
            self.ptr.open_mode = slurm.OPEN_MODE_APPEND
        elif val == "truncate":
            self.ptr.open_mode = slurm.OPEN_MODE_TRUNCATE

    def _set_script_args(self):
        args = self.script_args
        if not args:
            return None

        if isinstance(args, str):
            sargs = shlex.split(args)
        else:
            sargs = list(args)

        # Script should always first in argv.
        if sargs[0] != self.script:
            sargs.insert(0, self.script)

        self.ptr.argc = len(sargs)
        self.ptr.argv = <char**>try_xmalloc(self.ptr.argc * sizeof(char*))
        if not self.ptr.argv:
            raise MemoryError("xmalloc failed for script_args")

        for idx, opt in enumerate(sargs):
            cstr.fmalloc(&self.ptr.argv[idx], opt)

    def _set_environment(self):
        vals = self.environment
        get_user_env = self.get_user_environment

        # Clear any previous environment set for the Job.
        slurm_env_array_free(self.ptr.environment)
        self.ptr.env_size = 0

        # Allocate a new environment.
        self.ptr.environment = slurm_env_array_create()

        if isinstance(vals, str) or vals is None:
            if vals is None or vals.casefold() == "all":
                # This is the default. Export all current environment
                # variables into the Job.
                slurm_env_array_merge(&self.ptr.environment,
                                      <const char **>slurm.environ)
            elif vals.casefold() == "none":
                # Only env variables starting with "SLURM_" will be exported.
                for var, val in pyenviron.items():
                    if var.startswith("SLURM_"):
                        slurm_env_array_overwrite(&self.ptr.environment,
                                                  var, str(val))
                get_user_env = True
            else:
                # Assume Env-vars were provided sbatch style like a string.
                # Setup all 'SLURM' env vars found first.
                for var, val in pyenviron.items():
                    if var.startswith("SLURM_"):
                        slurm_env_array_overwrite(&self.ptr.environment,
                                                  var, str(val))

                # Merge the provided environment variables from the string in.
                for idx, item in enumerate(vals.split(",")):
                    if idx == 0 and item.casefold() == "all":
                        slurm_env_array_merge(&self.ptr.environment,
                                              <const char **>slurm.environ)
                        continue

                    if not "=" in item:
                        continue

                    var, val = item.split("=", 1)
                    slurm_env_array_overwrite(&self.ptr.environment,
                                              var, str(val))
                get_user_env = True
        else:
            # Here, the user provided an actual dictionary as Input.
            # Setup all 'SLURM' env vars first.
            for var, val in pyenviron.items():
                if var.startswith("SLURM_"):
                    slurm_env_array_overwrite(&self.ptr.environment,
                                              var, str(val))

            # Setup all User selected env vars.
            for var, val in vals.items():
                slurm_env_array_overwrite(&self.ptr.environment,
                                          var, str(val))

        if get_user_env:
            slurm_env_array_overwrite(&self.ptr.environment,
                                      "SLURM_GET_USER_ENV", "1")

        # Calculate Environment size
        while self.ptr.environment and self.ptr.environment[self.ptr.env_size]:
            self.ptr.env_size+=1

    def _set_distribution(self):
        dist=plane = None

        if not self.distribution:
            self.ptr.task_dist = slurm.SLURM_DIST_UNKNOWN
            return None

        if isinstance(self.distribution, int):
            # Assume the user meant to specify the plane size only.
            plane = u16(self.distribution)
        elif isinstance(self.distribution, str):
            # Support sbatch style string input
            dist = TaskDistribution.from_str(self.distribution)
            plane = dist.plane if isinstance(dist.plane, int) else 0

        if plane:
            self.ptr.plane_size = plane
            self.ptr.task_dist = slurm.SLURM_DIST_PLANE
        elif dist is not None:
            self.ptr.task_dist = dist.as_int()

    def _set_gpu_binding(self):
        binding = self.gpu_binding

        if not binding:
            if self.ptr.ntasks_per_tres != u16(None):
                # Set gpu bind implicit to single:ntasks_per_gpu
                binding = f"single:{self.ntasks_per_gpu}"
        else:
            binding = self.gpu_binding.replace("verbose,", "") \
                                      .replace("gpu:", "")
            if "verbose" in self.gpu_binding:
                binding = f"verbose,gpu:{binding}"

        cstr.fmalloc(&self.ptr.tres_bind, binding)

    def _set_min_cpus(self):
        if self.min_cpus_per_node:
            self.ptr.min_cpus = u16(self.min_cpus_per_node)
        elif not self.is_update:
            if self.overcommit:
                self.ptr.min_cpus = max(self.ptr.min_nodes, 1)

            self.ptr.min_cpus = self.ptr.cpus_per_task * self.ptr.num_tasks

    def _set_switches(self):
        vals = self.switches
        if not vals:
            return None

        if not isinstance(vals, dict):
            vals = _parse_switches_str_to_dict(vals)

        self.ptr.req_switch  = u32(kwargs.get("count"))
        self.ptr.wait4switch = timestr_to_secs(kwargs.get("max_wait_time"))

    def _set_signal(self):
        vals = self.signal
        if not vals:
            return None

        if not isinstance(vals, dict):
            vals = _parse_signal_str_to_dict(vals)

        self.ptr.warn_signal = u16(vals.get("signal"))
        self.ptr.warn_time = u16(vals.get("time"), on_noval=60)
        u16_set_bool_flag(&self.ptr.warn_flags,
                bool(vals.get("batch_only")), slurm.KILL_JOB_BATCH)
        u16_set_bool_flag(
                &self.ptr.warn_flags,
                bool(vals.get("allow_reservation_overlap")),
                slurm.KILL_JOB_RESV)

    def _set_gres_binding(self):
        if not self.gres_binding:
            return None

        binding = self.gres_binding.casefold()
        if binding == "enforce-binding":
            self.ptr.bitflags |= slurm.GRES_ENFORCE_BIND
        elif binding == "disable-binding":
            self.ptr.bitflags |= slurm.GRES_DISABLE_BIND

    def _set_gres_tasks_per_sharing(self):
        if not self.gres_tasks_per_sharing:
            return None

        sharing = self.gres_tasks_per_sharing.casefold()
        if sharing == "multiple" or sharing == "multiple-tasks-per-sharing":
            self.ptr.bitflags |= slurm.GRES_MULT_TASKS_PER_SHARING
        elif sharing == "one" or sharing == "one-task-per-sharing":
            self.ptr.bitflags |= slurm.GRES_ONE_TASK_PER_SHARING


def _parse_dependencies(val):
    final = None

    if isinstance(val, str):
        # TODO: Even though everything is checked in the slurmctld, maybe
        # still do some sanity checks here on the input when a string
        # is provided.
        final = val
    elif val is not None:
        satisfy = val.pop("satisfy", "all").casefold()

        if satisfy == "any":
            delim = "?"
        else:
            delim = ","

        final = []
        for condition, vals in val.items():
            if condition == "singleton" and bool(vals):
                final.append("singleton")
                continue

            if not isinstance(vals, list):
                vals = str(vals).split(",")

            vals = [str(s) for s in vals]
            final.append(f"{condition}:{':'.join(vals)}")

        final = delim.join(final)

    return final


def _parse_nodes(vals):
    nmin=nmax = 1

    # Support input like --nodes from sbatch (min-[max])
    if isinstance(vals, dict):
        nmin = u32(vals.get("min", 1), on_noval=1)
        nmax = u32(vals.get("max", 1), on_noval=nmin)
    elif vals is not None:
        v = str(vals).split("-", 1)
        nmin = int(v[0])
        if nmin == 0:
            nmin = 1
        if "-" in str(vals):
            nmax = int(v[1])
        else:
            nmax = nmin

    if not nmax:
        nmax = nmin
    if nmax < nmin:
        raise ValueError("Max Nodecount cannot be less than minimum"
                         " nodecount.")

    return nmin, nmax


def _parse_signal_str_to_dict(vals):
    info = {}
    # This supports input like the --signal option from sbatch
    val_list = re.split("[:@]+", str(vals))

    if len(val_list):
        if ":" in str(vals):
            flags = val_list.pop(0).casefold()

            if "r" in flags:
                info["allow_reservation_overlap"] = True

            if "b" in flags:
                info["batch_only"] = True

        if "@" in str(vals):
            info["time"] = val_list[1]

        info["signal"] = signal_to_num(val_list[0])

    return info


def _parse_switches_str_to_dict(switches_str):
    out = {}
    vals = str(switches_str.split("@"))
    if len(vals) > 1:
        out["max_wait_time"] = timestr_to_secs(vals[1])

    out["count"] = u32(vals[0])

    return out


def _parse_cpu_freq_str_to_dict(freq_str):
    freq_splitted = re.split("[-:]+", str(freq_str))
    freq_len = len(freq_splitted)
    freq = {}

    # Transform cpu-freq string to the individual components.
    if freq_splitted[0].isdigit():
        freq["max"] = freq_splitted[0]
    else:
        if freq_len > 1:
            raise ValueError(
                "Invalid cpu_frequency format: {kwargs}."
                "Governor must be provided as single element or "
                "as last element in the form of min-max:governor. "
            )
        freq["governor"] = freq_splitted[0]

    if freq_len >= 2:
        freq["min"] = freq["max"]
        freq["max"] = freq_splitted[1]

    if freq_len == 3:
        freq["governor"] = freq_splitted[2]

    return freq


def _validate_cpu_freq(freq):
    have_no_range = False
    freq_min = cpu_freq_str_to_int(freq.get("min"))
    freq_max = cpu_freq_str_to_int(freq.get("max"))
    freq_gov = cpu_gov_str_to_int(freq.get("governor"))

    if freq_min != u32(None):
        if freq_max == u32(None):
            freq_max = freq_min
            freq_min = u32(None)
            have_no_range = True
        elif freq_max < freq_min:
            raise ValueError(
                f"min cpu-freq ({freq_min}) must be smaller "
                f"than max cpu-freq ({freq_max})"
            )
    elif freq_max != u32(None) and freq_min == u32(None):
        have_no_range = True

    if have_no_range and freq_gov != u32(None):
        raise ValueError(
            "Setting Governor when specifying only either one "
            "of min or max is not allowed."
        )

    return freq_min, freq_max, freq_gov


def _validate_batch_script(script, args=None):
    try:
        _is_file = Path(script).is_file()
    except (ValueError, OSError):
        _is_file = False
    if _is_file:
        # First assume the caller is passing a path to a script and we try
        # to load it.
        script = Path(script).read_text()
    else:
        if args:
            raise ValueError("Passing arguments to a script is only allowed "
                             "if it was loaded from a file.")

    # Validate the script
    if not script or not len(script):
        raise ValueError("Batch script is empty or none was provided.")
    elif script.isspace():
        raise ValueError("Batch script contains only whitespace.")
    elif not script.startswith("#!"):
        msg = "Not a valid Batch script. "
        msg += "First line must start with '#!',"
        msg += "followed by the path to an interpreter"
        raise ValueError(msg)
    elif "\0" in script:
        msg = "The Slurm Controller does not allow scripts that "
        msg += "contain a NULL character: '\\0'."
        raise ValueError(msg)
    elif "\r\n" in script:
        msg = "Batch script contains DOS line breaks (\\r\\n) "
        msg += "instead of expected UNIX line breaks (\\n)."
        raise ValueError(msg)

    return script
