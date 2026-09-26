#########################################################################
# submission.pxd - interface for submitting slurm jobs
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

from pyslurm cimport slurm
from pyslurm.slurm cimport (
    job_desc_msg_t,
    slurm_init_job_desc_msg,
    slurm_free_job_desc_msg,
    submit_response_msg_t,
    slurm_submit_batch_job,
    slurm_free_submit_response_response_msg,
    slurm_env_array_free,
    slurm_env_array_create,
    slurm_env_array_merge,
    slurm_env_array_overwrite,
    slurm_job_share_string,
    xfree,
    try_xmalloc,
)
from pyslurm.utils cimport cstr, ctime
from pyslurm.utils.uint cimport *
from pyslurm.utils.ctime cimport time_t
from pyslurm.core.job.task_dist cimport TaskDistribution


cdef class JobSubmitDescription:
    """Submit Description for a Slurm Job.

    Args:
        **kwargs (Any, optional=None):
            Any valid Attribute this object has

    Attributes:
        name (str):
            Name of the Job, same as -J/--job-name from sbatch.
        account (str):
            Account of the job, same as -A/--account from sbatch.
        user_id (Union[str, int]):
            Run the job as a different User, same as --uid from sbatch.
            This requires root privileges.
            You can both specify the name or numeric uid of the User.
        group_id (Union[str, int]):
            Run the job as a different Group, same as --gid from sbatch.
            This requires root privileges.
            You can both specify the name or numeric gid of the User.
        priority (int):
            Specific priority the Job will receive.
            Same as --priority from sbatch.
            You can achieve the behaviour of sbatch's --hold option by
            specifying a priority of 0.
        site_factor (int):
            Site Factor of the Job. Only used when updating an existing Job.
        wckey (str):
            WCKey to use with the Job, same as --wckey from sbatch.
        array (str):
            Job Array specification, same as -a/--array from sbatch.
        batch_constraints (str):
            Batch Features of a Job, same as --batch from sbatch.
        begin_time (str):
            Defer allocation until the specified time, same as --begin from
            sbatch.
        clusters (Union[list, str]):
            Clusters the job may run on, same as -M/--clusters from sbatch.
        cluster_constraints (str):
            Comma-separated str with cluster constraints for the job.
            This is the same as --cluster-constraint from sbatch.
        comment (str):
            Arbitrary job comment, same as --comment from sbatch.
        admin_comment (str):
            Arbitrary job admin comment.
            Only used when updating an existing job.
        requires_contiguous_nodes (bool):
            Whether allocated Nodes are required to form a contiguous set.
            Same as --contiguous from sbatch.
        cores_reserved_for_system (int):
            Count of cores reserved for system not usable by the Job.
            Same as -S/--core-spec from sbatch.
            Mutually exclusive with `threads_reserved_for_system`.
        threads_reserved_for_system (int):
            Count of threads reserved for system not usable by the Job.
            Same as --thread-spec from sbatch.
            Mutually exclusive with `cores_reserved_for_system`.
        working_directory (str):
            Work directory for the Job. Default is current work-dir from where
            the job was submitted.
            Same as -D/--chdir from sbatch.
        cpu_frequency (Union[dict, str]):
            CPU Frequency for the Job, same as --cpu-freq from sbatch.

            For example, specifying it as a dict:

                cpu_frequency = {
                    "min": "Low",
                    "max": "High",
                    "governor": "UserSpace"
                }

            or like in sbatch with a string. For more info on that, check
            out the sbatch documentation for --cpu-freq.

            If you only want to set a Governor without any min or max, you
            can simply specify it as a standalone string:

                cpu_frequency = "Performance"
                or
                cpu_frequency = {"governor": "Performance"}

            If you want to set a specific, fixed frequency, you can do:

                cpu_frequency = <frequency in kilohertz>
                or either
                cpu_frequency = {"max": <freq>} or cpu_freq = {"min": <freq>}
        nodes (Union[dict, str, int]):
            Amount of nodes needed for the job.
            This is the same as -N/--nodes from sbatch.

            For example, providing min/max nodes as a dict:

                nodes = {
                    "min": 3,
                    "max": 6
                }

            When no range is needed, you can also simply specify it as int:

                nodes = 3

            Other than that, a range can also be specified in a str like with
            sbatch:

                nodes = "1-5"
        deadline (str):
            Deadline specification for the Job, same as --deadline from
            sbatch.
        delay_boot_time (Union[str, int]):
            Delay boot specification for the Job, same as --delay-boot from
            sbatch.
        dependencies (Union[dict, str]):
            Dependencies for the Job, same as -d/--dependency from sbatch.
        excluded_nodes (Union[list, str]):
            Exclude specific nodes for this Job.
            This is the same as -x/--exclude from sbatch.
        required_nodes (Union[list, str]):
            Specific list of nodes required for the Job.
            This is the same as -w/--nodelist from sbatch.
        constraints (str):
            Required node features for the Job.
            This is the same as -C/--constraint from sbatch.
        kill_on_node_fail (bool):
            Should the job get killed if one of the Nodes fails?
            This is the same as -k/--no-kill from sbatch.
        licenses (Union[list, str]):
            A list of licenses for the Job.
            This is the same as -L/--licenses from sbatch.
        mail_user (Union[list, str]):
            List of email addresses for notifications.
            This is the same as --mail-user from sbatch.
        mail_types (Union[list, str]):
            List of mail flags.
            This is the same as --mail-type from sbatch.
        mcs_label (str):
            An MCS Label for the Job.
            This is the same as --mcs-label from sbatch.
        memory_per_cpu (Union[str, int]):
            Memory required per allocated CPU.

            The default unit is in Mebibytes. You are also able to specify
            unit suffixes like K|M|G|T.
            This is the same as --mem-per-cpu from sbatch. This is mutually
            exclusive with memory_per_node and memory_per_gpu.


            Examples:

                # 1 MiB
                memory_per_cpu = 1024

                # 3 GiB
                memory_per_cpu = "3G"
        memory_per_node (Union[str, int]):
            Memory required per whole node.

            The default unit is in Mebibytes. You are also able to specify
            unit suffixes like K|M|G|T.
            This is the same as --mem from sbatch. This is mutually exclusive
            with memory_per_cpu and memory_per_gpu.


            Examples:

                # 1 MiB
                memory_per_node = 1024

                # 3 GiB
                memory_per_node = "3G"
        memory_per_gpu (Union[str, int]):
            Memory required per GPU.

            The default unit is in Mebibytes. You are also able to specify
            unit suffixes like K|M|G|T.
            This is the same as --mem-per-gpu from sbatch. This is mutually
            exclusive with memory_per_node and memory_per_cpu.


            Examples:

                # 1 MiB
                memory_per_gpu = 1024

                # 3 GiB
                memory_per_gpu = "3G"
        network (str):
            Network types for the Job.
            This is the same as --network from sbatch.
        nice (int):
            Adjusted scheduling priority for the Job.
            This is the same as --nice from sbatch.
        log_files_open_mode (str):
            Mode in which standard_output and standard_error log files should be opened.
            This is the same as --open-mode from sbatch.


            Valid options are:

            * `append`
            * `truncate`
        overcommit (bool):
            If the resources should be overcommitted.
            This is the same as -O/--overcommit from sbatch.
        partitions (Union[list, str]):
            A list of partitions the Job may use.
            This is the same as -p/--partition from sbatch.
        accounting_gather_frequency (Union[dict, str]):
            Interval for accounting info to be gathered.
            This is the same as --acctg-freq from sbatch.


            For example, specifying it as a dict:

                accounting_gather_frequency = {
                    "energy"=60,
                    "network"=20,
                }

            or as a single string:

                accounting_gather_frequency = "energy=60,network=20"
        qos (str):
            Quality of Service for the Job.
            This is the same as -q/--qos from sbatch.
        requires_node_reboot (bool):
            Force the allocated nodes to reboot before the job starts.
            This is the same --reboot from sbatch.
        is_requeueable (bool):
            If the Job is eligible for requeuing.
            This is the same as --requeue from sbatch.
        reservations (Union[list, str]):
            A list of possible reservations the Job can use.
            This is the same as --reservation from sbatch.
        script (str):
            Absolute Path or content of the batch script.

            You can specify either a path to a script which will be loaded, or
            you can pass the script as a string.
            If the script is passed as a string, providing arguments to it
            (see `script_args`) is not supported.
        script_args (str):
            Arguments passed to the batch script.
            You can only set arguments if a file path was specified for
            `script`.
        environment (Union[dict, str]):
            Environment variables to be set for the Job.
            This is the same as --export from sbatch.
        resource_sharing (str):
            Controls the resource sharing with other Jobs.
            This property combines functionality of --oversubscribe and
            --exclusive from sbatch.


            Allowed values are are:

            * `oversubscribe` or `yes`:

                The Job allows resources to be shared with other running Jobs.

            * `user`:

                Only sharing resources with other Jobs that have the "user"
                option set is allowed

            * `mcs`:

                Only sharing resources with other Jobs that have the "mcs"
                option set is allowed.

            * `no` or `exclusive`:

                No sharing of resources is allowed. (--exclusive from sbatch)
        distribution (str):
            Task distribution for the Job, same as --distribution from sbatch
        time_limit (str):
            The time limit for the job.
            This is the same as -t/--time from sbatch.
        time_limit_min (str):
            A minimum time limit for the Job.
            This is the same as --time-min from sbatch.
        container (str):
            Path to an OCI container bundle.
            This is the same as --container from sbatch.
        container_id (str):
            Unique name for the Container
            This is the same as --container-id from sbatch.
        cpus_per_task (int):
            The amount of cpus required for each task.

            This is the same as -c/--cpus-per-task from sbatch.
            This is mutually exclusive with `cpus_per_gpu`.
        cpus_per_gpu (int):
            The amount of cpus required for each allocated GPU.

            This is the same as --cpus-per-gpu from sbatch.
            This is mutually exclusive with `cpus_per_task`.
        sockets_per_node (int):
            Restrict Job to nodes with at least this many sockets.
            This is the same as --sockets-per-node from sbatch.
        cores_per_socket (int):
            Restrict Job to nodes with at least this many cores per socket
            This is the same as --cores-per-socket from sbatch.
        threads_per_core (int):
            Restrict Job to nodes with at least this many threads per socket
            This is the same as --threads-per-core from sbatch.
        gpus (Union[dict, str, int]):
            GPUs for the Job to be allocated in total.
            This is the same as -G/--gpus from sbatch.
            Specifying the type of the GPU is optional.


            For example, specifying the GPU counts as a dict:

                gpus = {
                    "tesla": 1,
                    "volta": 5,
                }

            Or, for example, in string format:

                gpus = "tesla:1,volta:5"

            Or, if you don't care about the type of the GPU:

                gpus = 6
        gpus_per_socket (Union[dict, str, int]):
            GPUs for the Job to be allocated per socket.

            This is the same as --gpus-per-socket from sbatch.

            Specifying the type of the GPU is optional. Note that setting
            `gpus_per_socket` requires to also specify sockets_per_node.


            For example, specifying it as a dict:

                gpus_per_socket = {
                    "tesla": 1,
                    "volta": 5,
                }

            Or, for example, in string format:

                gpus_per_socket = "tesla:1,volta:5"

            Or, if you don't care about the type of the GPU:

                gpus_per_socket = 6
        gpus_per_task (Union[dict, str, int]):
            GPUs for the Job to be allocated per task.

            This is the same as --gpus-per-task from sbatch.

            Specifying the type of the GPU is optional. Note that setting
            `gpus_per_task` requires to also specify either one of `ntasks` or
            `gpus`.

            For example, specifying it as a dict:

                gpus_per_task = {
                    "tesla": 1,
                    "volta": 5,
                }

            Or, for example, in string format:

                gpus_per_task = "tesla:1,volta:5"

            Or, if you don't care about the type of the GPU:

                gpus_per_task = 6
        gres_per_node (Union[dict, str]):
            Generic resources to be allocated per node.

            This is the same as --gres from sbatch. You should also use this
            option if you want to specify GPUs per node (--gpus-per-node).
            Specifying the type (by separating GRES name and type with a
            semicolon) is optional.

            For example, specifying it as a dict:

                gres_per_node = {
                    "gpu:tesla": 1,
                    "gpu:volta": 5,
                }

            Or, for example, in string format:

                gres_per_node = "gpu:tesla:1,gpu:volta:5"

            GPU Gres without a specific type:

                gres_per_node = "gpu:6"
        gpu_binding (str):
            Specify GPU binding for the Job.
            This is the same as --gpu-bind from sbatch.
        ntasks (int):
            Maximum amount of tasks for the Job.
            This is the same as -n/--ntasks from sbatch.
        ntasks_per_node (int):
            Amount of tasks to be invoked on each node.
            This is the same as --ntasks-per-node from sbatch.
        ntasks_per_socket (int):
            Maximum amount of tasks to be invoked on each socket.
            This is the same as --ntasks-per-socket from sbatch.
        ntasks_per_core (int):
            Maximum amount of tasks to be invoked on each core.
            This is the same as --ntasks-per-core from sbatch.
        ntasks_per_gpu (int):
            Amount of tasks to be invoked per GPU.
            This is the same as --ntasks-per-socket from sbatch.
        switches (Union[dict, str, int]):
            Maximum amount of leaf switches and wait time desired.

            This can also optionally include a maximum waiting time for these
            switches.
            This is the same as --switches from sbatch.


            For example, specifying it as a dict:

                switches = { "count": 5, "max_wait_time": "00:10:00" }

            Or as a single string (sbatch-style):

                switches = "5@00:10:00"
        signal (Union[dict, str]):
            Warn signal to be sent to the Job.

            This is the same as --signal from sbatch.
            The signal can both be specified with its name, e.g. "SIGKILL", or
            as a number, e.g. 9


            For example, specifying it as a dict:

                signal = {
                    "signal": "SIGKILL",
                    "time": 120
                }

            The above will send a "SIGKILL" signal 120 seconds before the
            Jobs' time limit is reached.

            Or, specifying it as a string (sbatch-style):

                signal = "SIGKILL@120"
        standard_in (str):
            Path to a File acting as standard_in for the batch-script.
            This is the same as -i/--input from sbatch.
        standard_error (str):
            Path to a File acting as standard_error for the batch-script.
            This is the same as -e/--error from sbatch.
        standard_output (str):
            Path to a File to write the Jobs standard_output.
            This is the same as -o/--output from sbatch.
        kill_on_invalid_dependency (bool):
            Kill the job if it has an invalid dependency.
            This is the same as --kill-on-invalid-dep from sbatch.
        spreads_over_nodes (bool):
            Spread the Job over as many nodes as possible.
            This is the same as --spread-job from sbatch.
        use_min_nodes (bool):
            Prefer the minimum amount of nodes specified.
            This is the same as --use-min-nodes from sbatch.
        gres_binding (str):
            Generic resource task binding options.
            This is contained in the --gres-flags option from sbatch.


            Possible values are:

            * `enforce-binding`
            * `disable-binding`
        gres_tasks_per_sharing (str):
            Shared GRES Tasks
            This is contained in the --gres-flags option from sbatch.


            Possible values are:

            * `multiple` or `multiple-tasks-per-sharing`
            * `one` or `one-task-per-sharing`
        temporary_disk_per_node (Union[str, int]):
            Amount of temporary disk space needed per node.

            This is the same as --tmp from sbatch. You can specify units like
            K|M|G|T (multiples of 1024).
            If no unit is specified, the value will be assumed as Mebibytes.


            Examples:

                # 2048 MiB
                tmp_disk_per_node = "2G"

                # 1024 MiB
                tmp_disk_per_node = 1024
        get_user_environment (Union[str, bool, int]):
            TODO
        min_cpus_per_node (str):
            Set the minimum amount of CPUs required per Node.
            This is the same as --mincpus from sbatch.
        wait_all_nodes (bool):
            Controls when the execution of the command begins.

            A value of True means that the Job should begin execution only
            after all nodes in the allocation are ready. Setting it to False,
            the default, means that it is not waited for the nodes to be
            ready. (i.e booted)
        selinux_context (str):
            Request a specific SELinux Context
        segment_size (int):
            When a block topology is used, this defines the size of the
            segments that will be used to create the job allocation
        reserved_ports (int):
            Number of ports that should be reserved for communication
        extra (str):
            Any arbitrary string
    """
    cdef:
        slurm.job_desc_msg_t *ptr
        is_update

    cdef public:
        name
        account
        user_id
        group_id
        priority
        site_factor
        wckey
        array
        batch_constraints
        begin_time
        clusters
        cluster_constraints
        comment
        admin_comment
        requires_contiguous_nodes
        cores_reserved_for_system
        threads_reserved_for_system
        working_directory
        cpu_frequency
        nodes
        deadline
        delay_boot_time
        dependencies
        excluded_nodes
        required_nodes
        constraints
        kill_on_node_fail
        licenses
        mail_user
        mail_types
        mcs_label
        memory_per_cpu
        memory_per_node
        memory_per_gpu
        network
        nice
        log_files_open_mode
        overcommit
        partitions
        profile_types
        accounting_gather_frequency
        qos
        requires_node_reboot
        is_requeueable
        reservations
        script
        script_args
        environment
        resource_sharing
        distribution
        time_limit
        time_limit_min
        container
        container_id
        cpus_per_task
        cpus_per_gpu
        sockets_per_node
        cores_per_socket
        threads_per_core
        gpus
        gpus_per_socket
        gpus_per_task
        gres_per_node
        gpu_binding
        ntasks
        ntasks_per_node
        ntasks_per_socket
        ntasks_per_core
        ntasks_per_gpu
        switches
        signal
        standard_in
        standard_output
        standard_error
        kill_on_invalid_dependency
        spreads_over_nodes
        use_min_nodes
        gres_binding
        gres_tasks_per_sharing
        temporary_disk_per_node
        get_user_environment
        min_cpus_per_node
        wait_all_nodes
        selinux_context
        segment_size
        reserved_ports
        extra
