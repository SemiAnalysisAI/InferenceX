#########################################################################
# sbatch_opt.pyx - utilities to parse #SBATCH options
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

import re
from pathlib import Path

SBATCH_MAGIC = "#SBATCH"


class SbatchOpt():
    def __init__(self, short_opt=None, long_opt=None,
                 our_attr_name=None, attr_param=None, is_boolean=False,
                 has_optional_args=False):
        self.short_opt = short_opt
        self.long_opt = long_opt
        self.our_attr_name = our_attr_name
        self.attr_param = attr_param
        self.is_boolean = is_boolean
        self.has_optional_args = has_optional_args

    def set(self, val, desc, overwrite):
        if self.our_attr_name is None:
            return None

        if getattr(desc, self.our_attr_name) is None or overwrite:
            val = self.attr_param if val is None else val
            setattr(desc, self.our_attr_name, val)


class SbatchOptGresFlags(SbatchOpt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set(self, val, desc, overwrite):
        for flag in val.split(","):
            flag = flag.casefold()

            if flag == "enforce-binding" or flag == "disable-binding":
                if desc.gres_binding is None or overwrite:
                    desc.gres_binding = flag
            elif flag == "one-task-per-sharing" or flag == "multiple-tasks-per-sharing":
                if desc.gres_tasks_per_sharing is None or overwrite:
                    desc.gres_tasks_per_sharing = flag


# Sorted by occurrence in the sbatch manpage - keep in order.
SBATCH_OPTIONS = [
    SbatchOpt("A", "account", "account"),
    SbatchOpt(None, "acctg-freq", "accounting_gather_frequency"),
    SbatchOpt("a", "array", "array"),
    SbatchOpt(None, "batch", "batch_constraints"),
    SbatchOpt(None, "bb", "burst_buffer"),
    SbatchOpt(None, "bbf", "burst_buffer_file"),
    SbatchOpt("b", "begin", "begin_time"),
    SbatchOpt("D", "chdir", "working_directory"),
    SbatchOpt(None, "cluster-constraint", "cluster_constraints"),
    SbatchOpt("M", "clusters", "clusters"),
    SbatchOpt(None, "comment","comment"),
    SbatchOpt("C", "constraint", "constraints"),
    SbatchOpt(None, "container", "container"),
    SbatchOpt(None, "contiguous", "requires_contiguous_nodes"),
    SbatchOpt("S", "core-spec", "cores_reserved_for_system"),
    SbatchOpt(None, "cores-per-socket", "cores_per_socket"),
    SbatchOpt(None, "cpu-freq", "cpu_frequency"),
    SbatchOpt(None, "cpus-per-gpu", "cpus_per_gpu"),
    SbatchOpt("c", "cpus-per-task", "cpus_per_task"),
    SbatchOpt(None, "deadline", "deadline"),
    SbatchOpt(None, "delay-boot", "delay_boot_time"),
    SbatchOpt("d", "dependency", "dependencies"),
    SbatchOpt("m", "distribution", "distribution"),
    SbatchOpt("e", "error", "standard_error"),
    SbatchOpt("x", "exclude", "excluded_nodes"),
    SbatchOpt(None, "exclusive", "resource_sharing", "no"),
    SbatchOpt(None, "export", "environment"),
    SbatchOpt(None, "export-file", None),
    SbatchOpt("B", "extra-node-info", None),
    SbatchOpt(None, "get-user-env", "get_user_environment"),
    SbatchOpt(None, "gid", "group_id"),
    SbatchOpt(None, "gpu-bind", "gpu_binding"),
    SbatchOpt(None, "gpu-freq", None),
    SbatchOpt("G", "gpus", "gpus"),
    SbatchOpt(None, "gpus-per-node", "gpus_per_node"),
    SbatchOpt(None, "gpus-per-socket", "gpus_per_socket"),
    SbatchOpt(None, "gpus-per-socket", "gpus_per_task"),
    SbatchOpt(None, "gres", "gres_per_node"),
    SbatchOptGresFlags(None, "gres-flags"),
    SbatchOpt(None, "hint", None),
    SbatchOpt("H", "hold", "priority", 0),
    SbatchOpt(None, "ignore-pbs", None),
    SbatchOpt("i", "input", "standard_in"),
    SbatchOpt("J", "job-name", "name"),
    SbatchOpt(None, "kill-on-invalid-dep", "kill_on_invalid_dependency"),
    SbatchOpt("L", "licenses", "licenses"),
    SbatchOpt(None, "mail-type", "mail_types"),
    SbatchOpt(None, "mail-user", "mail_user"),
    SbatchOpt(None, "mcs-label", "mcs_label"),
    SbatchOpt(None, "mem", "memory_per_node"),
    SbatchOpt(None, "mem-bind", None),
    SbatchOpt(None, "mem-per-cpu", "memory_per_cpu"),
    SbatchOpt(None, "mem-per-gpu", "memory_per_gpu"),
    SbatchOpt(None, "mincpus", "min_cpus_per_node"),
    SbatchOpt(None, "network", "network"),
    SbatchOpt(None, "nice", "nice"),
    SbatchOpt("k", "no-kill", "kill_on_node_fail", False),
    SbatchOpt(None, "no-requeue", "is_requeueable", False),
    SbatchOpt("F", "nodefile", None),
    SbatchOpt("w", "nodelist", "required_nodes"),
    SbatchOpt("N", "nodes", "nodes"),
    SbatchOpt("n", "ntasks", "ntasks"),
    SbatchOpt(None, "ntasks-per-core", "ntasks_per_core"),
    SbatchOpt(None, "ntasks-per-gpu", "ntasks_per_gpu"),
    SbatchOpt(None, "ntasks-per-node", "ntasks_per_node"),
    SbatchOpt(None, "ntasks-per-socket", "ntasks_per_socket"),
    SbatchOpt(None, "open-mode", "log_files_open_mode"),
    SbatchOpt("o", "output", "standard_output"),
    SbatchOpt("O", "overcommit", "overcommit", True),
    SbatchOpt("s", "oversubscribe", "resource_sharing", "yes"),
    SbatchOpt("p", "partition", "partition"),
    SbatchOpt(None, "prefer", None),
    SbatchOpt(None, "priority", "priority"),
    SbatchOpt(None, "profile", "profile_types"),
    SbatchOpt(None, "propagate", None),
    SbatchOpt("q", "qos", "qos"),
    SbatchOpt(None, "reboot", "requires_node_reboot", True),
    SbatchOpt(None, "requeue", "is_requeueable", True),
    SbatchOpt(None, "reservation", "reservations"),
    SbatchOpt(None, "signal", "signal"),
    SbatchOpt(None, "sockets-per-node", "sockets_per_node"),
    SbatchOpt(None, "spread-job", "spreads_over_nodes", True),
    SbatchOpt(None, "switches", "switches"),
    SbatchOpt(None, "thread-spec", "threads_reserved_for_system"),
    SbatchOpt(None, "threads-per-core", "threads_per_core"),
    SbatchOpt("t", "time", "time_limit"),
    SbatchOpt(None, "time-min", "time_limit_min"),
    SbatchOpt(None, "tmp", "temporary_disk_per_node"),
    SbatchOpt(None, "uid", "user_id"),
    SbatchOpt(None, "use-min-nodes", "use_min_nodes", True),
    SbatchOpt(None, "wait-all-nodes", "wait_all_nodes", True),
    SbatchOpt(None, "wckey", "wckey"),
]


def _parse_line(line):
    # Remove the #SBATCH from the start
    opts = line[len("#SBATCH"):]

    # Ignore possible comments after the options
    opts = opts.split("#")[0].strip()

    # Now the line can be in these forms for example:
    # * -t20 or -t 20
    # * --time=20 or --time 20 or --time20
    if "=" in opts:
        # -t=21 or --time=20
        opts = "=".join(opts.replace("=", " ").split())
        opt, val = opts.split("=")
    elif " " in opts:
        # --time 20 or -t 20
        opts = "=".join(opts.split())
        opt, val = opts.split("=")
    elif any(el.isdigit() for el in opts):
        # -t20 or --time20
        opt, val = list(filter(None, re.split(r'(\d+)', opts)))
    else:
        # Probably a boolean flag, like --exclusive or -O
        opt, val = opts, None

    # Remove "-" or "--" at the front.
    opt = opt[1:]
    if opt[0] == "-":
        # Found second dash.
        opt = opt[1:]

    return opt, val


def _find_opt(opt):
    for sbopt in SBATCH_OPTIONS:
        # Check if we can find the option in our predefined mapping.
        if opt == sbopt.short_opt or opt == sbopt.long_opt:
            return sbopt

    return SbatchOpt()


def _parse_opts_from_batch_script(desc, script, overwrite):
    flags_and_vals = {}

    if not Path(script).is_file():
        raise ValueError("The script path you provided is not valid.")

    script = Path(script).read_text()
    for line in script.splitlines():
        line = line.lstrip()

        if line.startswith(SBATCH_MAGIC):
            flag, val = _parse_line(line)
            opt = _find_opt(flag)
            opt.set(val, desc, overwrite)
