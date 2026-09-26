#########################################################################
# slurmctld/config.pxd - pyslurm slurmctld config api
#########################################################################
# Copyright (C) 2025 Toni Harzendorf <toni.harzendorf@gmail.com>
#
# Note: Some classes are annotated with additional Copyright notices further
# down
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
    slurm_conf_t,
    slurm_load_ctl_conf,
    slurm_free_ctl_conf,
    slurm_preempt_mode_string,
    slurm_accounting_enforce_string,
    slurm_sprint_cpu_bind_type,
    slurm_ctl_conf_2_key_pairs,
    cpu_bind_type_t,
    list_t,
    xfree,
)
from pyslurm.utils cimport cstr
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from pyslurm.utils.uint cimport (
    u16_parse,
    u32_parse,
    u64_parse,
    u16_parse_bool,
)

from pyslurm.db.util cimport (
    SlurmList,
    SlurmListItem,
)


cdef dict _parse_config_key_pairs(void *ptr, owned=*)


ctypedef struct config_key_pair_t:
    char *name
    char *value


# Documentation for the attributes in the Config class have been largely taken
# from the official slurm.conf overview at:
# https://slurm.schedmd.com/slurm.conf.html
#
# Therefore, the following Copyright notices that slurm.conf has (see
# https://slurm.schedmd.com/slurm.conf.html#SECTION_COPYING), are also listed
# here:
#
# Copyright (C) 2002-2007 The Regents of the University of California. Produced
# at Lawrence Livermore National Laboratory (cf, pyslurm/slurm/SLURM_DISCLAIMER).
# Copyright (C) 2008-2010 Lawrence Livermore National Security.
# Copyright (C) 2010-2022 SchedMD LLC.
cdef class Config:
    """The Slurm Configuration.

    All attributes in this class are read-only.

    Attributes:
        cgroup_config (pyslurm.slurmctld.CgroupConfig):
            The CGroup Configuration data.
        accounting_gather_config (pyslurm.slurmctld.AccountingGatherConfig):
            The Accounting Gather Configuration data.
        mpi_config (pyslurm.slurmctld.MPIConfig):
            The MPI Configuration data.
        accounting_storage_enforce (list[str]):
            List of enforcements on Job submissions.

            {slurm.conf#OPT_AccountingStorageEnforce}
        accounting_storage_backup_host (str):
            Name of the backup machine hosting the Slurm database.

            {slurm.conf#OPT_AccountingStorageBackupHost}
        accounting_storage_external_hosts (list[str]):
            List of external slurmdbds to register with.

            {slurm.conf#OPT_AccountingStorageExternalHost}
        accounting_storage_host (str):
            Name of the machine hosting the slurm database.
            {slurm.conf#OPT_AccountingStorageHost}

        accounting_storage_parameters (dict[str, Union[str, int, bool]]):
            Options for the accounting storage Plugin

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_AccountingStorageParameters}
        accounting_storage_port (int):
            Listening port of the Accounting Database Server

            {slurm.conf#OPT_AccountingStoragePort}
        accounting_storage_tres (list):
            List of configured Resources to track on the Cluster.

            {slurm.conf#OPT_AccountingStorageTRES}
        accounting_storage_type (str):
            The accounting storage type used.

            {slurm.conf#OPT_AccountingStorageType}
        accounting_store_flags (list[str]):
            List of fields that the slurmctld also sends to the accounting
            database.

            {slurm.conf#OPT_AccountingStoreFlags}
        accounting_gather_node_frequency (int):
            Accounting-Gather plugins sampling interval for node accounting.

            {slurm.conf#OPT_AcctGatherNodeFreq}
        accounting_gather_energy_type (str):
            Plugin used for energy consumption accounting.

            {slurm.conf#OPT_AcctGatherEnergyType}
        accounting_gather_interconnect_type (str):
            Plugin used for interconnect network traffic accounting.

            {slurm.conf#OPT_AcctGatherInterconnectType}
        accounting_gather_filesystem_type (str):
            Plugin used for filesystem traffic accounting.

            {slurm.conf#OPT_AcctGatherFilesystemType}
        accounting_gather_profile_type (str):
            Plugin used for detailed job profiling.

            {slurm.conf#OPT_AcctGatherProfileType}
        allow_spec_resource_usage (bool):
            Whether Slurm allows jobs to override the nodes configured
            `CoreSpecCount`

            {slurm.conf#OPT_AllowSpecResourcesUsage}
        auth_alt_types (list[str]):
            List of alternative authentication plugins the slurmctld permits.

            {slurm.conf#OPT_AuthAltTypes}
        auth_alt_parameters (dict[str, Union[str, int, bool]]):
            Options for the alternative authentication plugins.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_AuthAltParameters}
        auth_info (list[str]):
            List of additional information used for authentication of
            communication between Slurm daemons.

            {slurm.conf#OPT_AuthInfo}
        auth_type (str):
            Primary authentication method for communications between Slurm
            components.

            {slurm.conf#OPT_AuthType}
        batch_start_timeout (int):
            The maximum time (in seconds) that a batch job is permitted for
            launching before being considered missing and releasing the
            allocation.

            {slurm.conf#OPT_BatchStartTimeout}
        bcast_exclude_paths (list[str]):
            List of absolute directory paths to be excluded when
            autodetecting and broadcasting executable shared object dependencies
            through `sbcast` or `srun --bcast`.

            {slurm.conf#OPT_BcastExclude}
        bcast_parameters (dict[str, Union[int, str, bool]]:
            Options for `sbcast` and `srun --bcast` behaviour.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_BcastParameters}
        burst_buffer_type (str):
            Plugin used to manage burst buffers.

            {slurm.conf#OPT_BurstBufferType}
        slurmctld_boot_time (int):
            Timestamp of when the slurmctld last booted.
        certmgr_parameters (str):
            List of options for the certmgr Plugin.
        certmgr_type (str):
            Plugin used for certmgr mechanism.
        cli_filter_plugins (list[str]):
            List of CLI Filter plugins to use.

            {slurm.conf#OPT_CliFilterPlugins}
        cluster_name (str):
            Name of the Cluster.

            {slurm.conf#OPT_ClusterName}
        communication_parameters (dict[str, Union[str, int]]):
            Communication options for Cluster daemons.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_CommunicationParameters}
        complete_wait_time (int):
            The time to wait, in seconds, when any job is in the COMPLETING state
            before any additional jobs are scheduled.

            {slurm.conf#OPT_CompleteWait}
        default_cpu_frequency_governor (str):
            Default CPU governor to use when a Job has not specified the
            `--cpu-freq` option.

            {slurm.conf#OPT_CpuFreqDef}
        cpu_frequency_governors (list[str]):
            List of CPU Governors allowed to be set on Job submission.

            {slurm.conf#OPT_CpuFreqGovernors}
        credential_type (str):
            Cryptographic signature tool to be used when creating job step
            credentials.

            {slurm.conf#OPT_CredType}
        data_parser_parameters (str):
            Default value to apply for `data_parser` plugin parameters.

            {slurm.conf#OPT_DataParserParameters}
        debug_flags (list[str]):
            List of DebugFlags currently set for Daemons.

            {slurm.conf#OPT_DebugFlags}
        default_memory_per_cpu (int):
            Default real memory size available per allocated CPU in Mebibytes.

            {slurm.conf#OPT_DefMemPerCPU}
        default_memory_per_node (int):
            Default real memory size available per allocated Node in Mebibytes.

            {slurm.conf#OPT_DefMemPerNode}
        dependency_parameters (dict[str, Union[str, int, bool]]):
            List of parameters for dependencies.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_DependencyParameters}
        disable_root_jobs (bool):
            Whether root can submit Jobs or not.

            {slurm.conf#OPT_DisableRootJobs}
        eio_timeout (int):
            The number of seconds srun waits for slurmstepd to close the TCP/IP
            connection used to relay data between the user application and srun
            when the user application terminates.

            {slurm.conf#OPT_EioTimeout}
        enforce_partition_limits (str):
            Controls which Limits are enforced on Partition level.

            {slurm.conf#OPT_EnforcePartLimits}
        epilog (list[str]):
            List of Epilog scripts in use that are executed as root on every
            node when a Job completes.

            {slurm.conf#OPT_Epilog}
        epilog_timeout (int):
            The interval in seconds Slurm waits for Epilog before terminating
            them.

            {slurm.conf#OPT_EpilogTimeout}
        epilog_msg_time (int):
            The number of microseconds that the slurmctld daemon requires to
            process an epilog completion message from the slurmd daemons.

            {slurm.conf#OPT_EpilogMsgTime}
        epilog_slurmctld (list[str]):
            List of Epilog scripts in use that are executed by slurmctld at job
            allocation.

            {slurm.conf#OPT_EpilogSlurmctld}
        fair_share_dampening_factor (int):
            Dampen the effect of exceeding a user or group's fair share of
            allocated resources.

            {slurm.conf#OPT_FairShareDampeningFactor}
        federation_parameters (dict[str, Union[str, int, bool]]):
            Options for Federations

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_FederationParameters}
        first_job_id (int):
            The job id to be used for the first job submitted.

            {slurm.conf#OPT_FirstJobId}
        get_environment_timeout (int):
            How long a Job waits (in seconds) to load the Users environment
            before attempting to load it from a cache file.

            {slurm.conf#OPT_GetEnvTimeout}
        gres_types (list[str]):
            List of generic resources to be managed.

            {slurm.conf#OPT_GresTypes}
        group_update_force (bool):
            Whether user group membership information is updated periodically,
            even if there are no changes to `/etc/group`.

            {slurm.conf#OPT_GroupUpdateForce}
        group_update_time (int):
            How frequently information about user group membership is updated,
            and how longer it is cached (in seconds).

            {slurm.conf#OPT_GroupUpdateTime}
        default_gpu_frequency (str):
            Default GPU frequency to use when running a job step if it has not
            been explicitly set using the --gpu-freq option.

            {slurm.conf#OPT_GpuFreqDef}
        hash_plugin (str):
            Type of hash plugin used for network communication.

            {slurm.conf#OPT_HashPlugin}
        hash_value (str):
            Current configuration hash value (hex).
        health_check_interval (int):
            Interval in seconds between executions of `HealthCheckProgram`

            {slurm.conf#OPT_HealthCheckInterval}
        health_check_node_state (list[str]):
            List of node states which are eligible to execute
            `HealthCheckProgram`.

            {slurm.conf#OPT_HealthCheckNodeState}
        health_check_program (str):
            Pathname of a script that is periodally executed as root user on
            all compute nodes.

            {slurm.conf#OPT_HealthCheckProgram}
        inactive_limit (int):
            The interval, in seconds, after which a non-responsive job
            allocation command (e.g. `srun` or `salloc`) will result in the job
            being terminated.

            {slurm.conf#OPT_InactiveLimit}
        interactive_step_options (str):
            When `LaunchParameters=use_interactive_step` is enabled, launching
            salloc will automatically start an srun process with
            `interactive_step_options` to launch a terminal on a node in the job
            allocation.

            {slurm.conf#OPT_InteractiveStepOptions}
        job_accounting_gather_type (str):
            The job accounting gather plugin used to collect usage information
            about Jobs.

            {slurm.conf#OPT_JobAcctGatherType}
        job_accounting_gather_frequency (dict[str, int]):
            The job accounting and profiling sampling intervals.

            {slurm.conf#OPT_JobAcctGatherFrequency}
        job_accounting_gather_parameters (list[str]):
            Arbitrary paramerers for `job_accounting_gather_type`

            {slurm.conf#OPT_JobAcctGatherParams}
        job_completion_host (str):
            Name of the machine hosting the job completion database.

            {slurm.conf#OPT_JobCompHost}
        job_completion_location (str):
            Sets a string which has different meaning depending on
            `job_completion_type`

            {slurm.conf#OPT_JobCompLoc}
        job_completion_parameters (dict[str, Union[str, int, bool]]):
            Arbitrary text passed to the Job completion plugin.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_JobCompParams}
        job_completion_port (int):
            The listening port of the job completion database server.

            {slurm.conf#OPT_JobCompPort}
        job_completion_type (str):
            Job completion logging mechanism type

            {slurm.conf#OPT_JobCompType}
        job_completion_user (str):
            User account user for accessing the job completion database.

            {slurm.conf#OPT_JobCompUser}
        namespace_plugin (str):
            Plugin used for job isolation through Linux namespaces.

            {slurm.conf#OPT_NamespaceType}
        job_file_append (bool):
            This option controls what to do if a job's output or error file
            exist when the job is started. If `True`, then append to the
            existing file. `False`, which is the default, means any existing
            files are truncated.

            {slurm.conf#OPT_JobFileAppend}
        job_requeue (bool):
            Whether jobs are requeuable by default

            {slurm.conf#OPT_JobRequeue}
        job_submit_plugins (list[str]):
            Site specific list of plugins used for setting default job
            parameters and/or logging events

            {slurm.conf#OPT_JobSubmitPlugins}
        kill_on_bad_exit (bool):
            Whether a step will be terminated immediately if any task is
            crashed or aborted.

            {slurm.conf#OPT_KillOnBadExit}
        kill_wait_time (int):
            The interval, in seconds, given to a job's processes between the
            `SIGTERM` and `SIGKILL` signals upon reaching its time limit.

            {slurm.conf#OPT_KillWait}
        launch_parameters (list[str]):
            Options for the job launch plugin.

            {slurm.conf#OPT_LaunchParameters}
        licenses (dict[str, int]):
            Licenses that can be allocated to jobs.

            {slurm.conf#OPT_Licenses}
        log_time_format (str):
            Format of the timestamp in slurmctld and slurmd log-files.

            {slurm.conf#OPT_LogTimeFormat}
        mail_domain (str):
            Domain name to qualify usernames if email address is not explicitly
            given with the `--mail-user` option.

            {slurm.conf#OPT_MailDomain}
        mail_program (str):
            Pathname to the program used to send emails per user request

            {slurm.conf#OPT_MailProg}
        max_array_size (int):
            Maximum job array task index value allowed.

            {slurm.conf#OPT_MaxArraySize}
        max_batch_requeue (int):
            Maximum number of times a batch job may be automatically requeued
            before being marked as `JobHeldAdmin`.

            {slurm.conf#OPT_MaxBatchRequeue}
        max_dbd_msgs (int):
            Maximum number of messages the Slurm controllers queues before
            starting to drop them when the slurmdbd is down.

            {slurm.conf#OPT_MaxDBDMsgs}
        max_job_count (int):
            Maximum number of jobs slurmctld can have in memory at one time.

            {slurm.conf#OPT_MaxJobCount}
        max_job_id (int):
            Highest job ID possible for Jobs that will be assigned
            automatically on submission.

            {slurm.conf#OPT_MaxJobId}
        max_memory_per_cpu (int):
            Maximum real memory size available per allocated CPU in Mebibytes.

            {slurm.conf#OPT_MaxMemPerCPU}
        max_memory_per_node (int):
            Maximum real memory size available per allocated Node in Mebibytes.

            {slurm.conf#OPT_MaxMemPerNode}
        max_node_count (int):
            Maximum count of nodes which may exist in the slurmctld.

            {slurm.conf#OPT_MaxNodeCount}
        max_step_count (int):
            Maximum number of Steps that any Job can initiate.

            {slurm.conf#OPT_MaxStepCount}
        max_tasks_per_node (int):
            Maximum number of tasks Slurm will allow for a job step to spawn on
            a single node.

            {slurm.conf#OPT_MaxTasksPerNode}
        mcs_plugin (str):
            Associate a security label to jobs, for resource sharing among jobs
            with the same label.

            {slurm.conf#OPT_MCSPlugin}
        mcs_parameters (list[str]):
            Parameters for the MCS Plugin.

            {slurm.conf#OPT_MCSParameters}
        metrics_type (str):
            Name of the Metrics plugin used.

            {slurm.conf#OPT_MetricsType}
        min_job_age (int):
            Minimum age (in seconds) of a completed Job before its record is
            cleared from slurmctlds memory.

            {slurm.conf#OPT_MinJobAge}
        mpi_default (str):
            Default type of MPI that will be used.

            {slurm.conf#OPT_MpiDefault}
        mpi_parameters (dict[str, Union[str, int, bool]]):
            Parameters for MPI.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_MpiParams}
        message_timeout (int):
            Time permitted for a round-trip communication to complete in
            seconds.

            {slurm.conf#OPT_MessageTimeout}
        next_job_id (int):
            Next Job-ID that will be assigned.
        node_features_plugins (list[str]):
            Plugins to be used for support of node features which can change
            through time.

            {slurm.conf#OPT_NodeFeaturesPlugins}
        over_time_limit (int):
            Number of minutes by which a job can exceed its time limit before
            being canceled.

            {slurm.conf#OPT_OverTimeLimit}
        plugin_dirs (list[str]):
            List of paths where Slurm looks for plugins.

            {slurm.conf#OPT_PluginDir}
        plugin_stack_config (str):
            Location of the config file for Slurm stackable plugins.

            {slurm.conf#OPT_PlugStackConfig}
        preempt_exempt_time (int):
            Minimum run time for all jobs before they can be considered for
            preemption.

            {slurm.conf#OPT_PreemptExemptTime}
        preempt_mode (str):
            Mechanism used to preempt jobs or enable gang scheduling.

            {slurm.conf#OPT_PreemptMode}
        preempt_parameters (dict[str, Union[str, int, bool]]):
            Options for the Preempt Plugin.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_PreemptParameters}
        preempt_type (str):
            Plugin used to identify which jobs can be preempted.

            {slurm.conf#OPT_PreemptMode}
        prep_parameters (list[str]):
            Parameters passed to the PrEpPlugins.

            {slurm.conf#OPT_PrEpParamrters}
        prep_plugins (list[str]):
            List of PrEp Plugins to be used.

            {slurm.conf#OPT_PrEpPlugins}
        priority_decay_half_life (int):
            Controls how long (in seconds) prior resource use is considered in
            determining how over- or under-serviced an association is.

            {slurm.conf#OPT_PriorityDecayHalfLife}
        priority_calc_period (int):
            Period (in minutes) in which the half-life decay will be
            re-calculated.

            {slurm.conf#OPT_PriorityCalcPeriod}
        priority_favor_small (bool):
            Whether small jobs should be given preferential scheduling
            priority.

            {slurm.conf#OPT_PriorityFavorSmall}
        priority_flags (list[str]):
            List of flags that modify priority behaviour.

            {slurm.conf#OPT_PriorityFlags}
        priority_max_age (int):
            Job age (in seconds) that is needed before receiving the maximum
            age factor in computing priority.

            {slurm.conf#OPT_PriorityMaxAge}
        priority_parameters (str):
            Arbitrary string used by the `priority_type` plugin.

            {slurm.conf#OPT_PriorityParameters}
        priority_usage_reset_period (str):
            At this interval the usage of associations will be reset to 0.

            {slurm.conf#OPT_PriorityUsageResetPeriod}
        priority_type (str):
            Specifies the plugin to be used in establishing a job's scheduling
            priority.

            {slurm.conf#OPT_PriorityType}
        priority_weight_age (int):
            An integer value that sets the degree to which the queue wait time
            component contributes to the job's priority.

            {slurm.conf#OPT_PriorityWeightAge}
        priority_weight_assoc (int):
            An integer value that sets the degree to which the association
            component contributes to the job's priority.

            {slurm.conf#OPT_PriorityWeightAssoc}
        priority_weight_fair_share (int):
            An integer value that sets the degree to which the fair-share
            component contributes to the job's priority.

            {slurm.conf#OPT_PriorityWeightFairShare}
        priority_weight_job_size (int):
            An integer value that sets the degree to which the job size
            component contributes to the job's priority.

            {slurm.conf#OPT_PriorityWeightJobSize}
        priority_weight_partition (int):
            Partition factor used by priority/multifactor plugin in calculating
            job priority.

            {slurm.conf#OPT_PriorityWeightPartition}
        priority_weight_qos (int):
            An integer value that sets the degree to which the Quality Of
            Service component contributes to the job's priority

            {slurm.conf#OPT_PriorityWeightQOS}
        priority_weight_tres (dict[str, int]):
            TRES Types and weights that sets the degree that each TRES Type
            contributes to the job's priority.

            {slurm.conf#OPT_PriorityWeightTRES}
        private_data (list[str]):
            Defines what type of information is hidden from regular users.

            {slurm.conf#OPT_PrivateData}
        proctrack_type (str):
            Identifies the plugin to be used for process tracking on a job step
            basis.

            {slurm.conf#OPT_ProctrackType}
        prolog (list[str]):
            List of pathnames of programs for the slurmd to execute whenever
            it is asked to run a job step from a new job allocation.

            {slurm.conf#OPT_Prolog}
        prolog_timeout (int):
            The interval in seconds Slurm waits for Prolog and Epilog before
            terminating them.

            {slurm.conf#OPT_PrologTimeout}
        prolog_slurmctld (list[str]):
            List of pathnames of programs for the slurmctld daemon to execute
            before granting a new job allocation.

            {slurm.conf#OPT_PrologSlurmctld}
        propagate_prio_process (int):
            Controls the scheduling priority (nice value) of user spawned
            tasks.

            {slurm.conf#OPT_PropagatePrioProcess}
        prolog_flags (list[str]):
            Flags to control the Prolog behavior.

            {slurm.conf#OPT_PrologFlags}
        propagate_resource_limits (list[str]):
            List of resource limit names that are propagated to the Job
            environment.

            {slurm.conf#OPT_PropagateResourceLimits}
        propagate_resource_limits_except (list[str]):
            List of resource limit names that are excluded from propagation to
            the Job environment.

            {slurm.conf#OPT_PropagateResourceLimitsExcept}
        reboot_program (str):
            Program to be executed on each compute node to reboot it.

            {slurm.conf#OPT_RebootProgram}
        reconfig_flags (list[str]):
            List of flags to control various actions that may be taken when a
            reconfigure command is issued (for example with `scontrol
            reconfig`).

            {slurm.conf#OPT_ReconfigFlags}
        requeue_exit (str):
            Enables automatic requeue for batch jobs which exit with the
            specified values.

            {slurm.conf#OPT_RequeueExit}
        requeue_exit_hold (str):
            Enables automatic requeue for batch jobs which exit with the
            specified values, with these jobs being held until released
            manually by the user.

            {slurm.conf#OPT_RequeueExitHold}
        resume_fail_program (str):
            The program that will be executed when nodes fail to resume to by
            `resume_timeout`.

            {slurm.conf#OPT_ResumeFailProgram}
        resume_program (str):
            Program that will be executed when a node in power save mode is
            assigned work to perform.

            {slurm.conf#OPT_ResumeProgram}
        resume_rate (int):
            Number of nodes per minute that will be restored from power save
            mode to normal operation by `resume_program`.

            {slurm.conf#OPT_ResumeRate}
        resume_timeout (int):
            Maximum time permitted (in seconds) between when a node resume
            request is issued and when the node is actually available for use.

            {slurm.conf#OPT_ResumeTimeout}
        reservation_epilog (str):
            Pathname of a program for the slurmctld to execute when a
            reservation ends.

            {slurm.conf#OPT_ResvEpilog}
        reservation_over_run (int):
            Describes how long (in minutes) a job already running in a
            reservation should be permitted to execute after the end time of
            the reservation has been reached

            {slurm.conf#OPT_ResvOverRun}
        reservation_prolog (str):
            Pathname of a program for the slurmctld to execute when a
            reservation begins.

            {slurm.conf#OPT_ResvProlog}
        return_to_service (int):
            Controls when a `DOWN` node will be returned to service

            {slurm.conf#OPT_ReturnToService}
        scheduler_log_file (str):
            Pathname of the scheduling event logging file.

            {slurm.conf#OPT_SlurmSchedLogFile}
        scheduler_logging_enabled (bool):
            The initial level of scheduling event logging.

            {slurm.conf#OPT_SlurmSchedLogLevel}
        scheduler_parameters (dict[str, Union[str, int, bool]]):
            List of options for the `scheduler_type` plugin.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_SchedulerParameters}
        scheduler_time_slice (int):
            Number of seconds in each time slice when gang scheduling is
            enabled.

            {slurm.conf#OPT_SchedulerTimeSlice}
        scheduler_type (str):
            Identifies the type of scheduler to be used.

            {slurm.conf#OPT_SchedulerType}
        scron_parameters (list[str]):
            Parameters for scron.

            {slurm.conf#OPT_ScronParameters}
        select_type (str):
            Identifies the type of resource selection algorithm to be used.

            {slurm.conf#OPT_SelectType}
        select_type_parameters (list[str]):
            Parameters passed to the `select_type` plugin.

            {slurm.conf#OPT_SelectTypeParameters}
        priority_site_factor_plugin (str):
            Specifies an optional plugin to be used alongside
            "priority/multifactor", which is meant to initially set and
            continuously update the SiteFactor priority factor.

            {slurm.conf#OPT_PrioritySiteFactorPlugin}
        priority_site_factor_parameters (str):
            Arbitrary string used by the PrioritySiteFactorPlugin plugin.

            {slurm.conf#OPT_PrioritySiteFactorParameters}
        slurm_conf_path (str):
            Path of the current slurm.conf file used.
        slurm_user_id (int):
            UID of the `slurm_user_Name`
        slurm_user_name (str):
            Name of the Slurm User
        slurmd_user_id (int):
            UID of the `slurmd_user_name`
        slurmd_user_name (str):
            Name of the User slurmd runs as.
        slurmctld_log_level (str):
            The level of detail to provide `slurmctld` daemon's logs.

            {slurm.conf#OPT_SlurmctldDebug}
        slurmctld_log_file (str):
            Pathname of a file into which the `slurmctld` daemon's logs are
            written.

            {slurm.conf#OPT_SlurmctldLogFile}
        slurmctld_pid_file (str):
            Pathname of a file into which the `slurmctld` daemon may write its
            process id.

            {slurm.conf#OPT_SlurmctldPidFile}
        slurmctld_port (str):
            Port number where `slurmctld` listens to for work.
            Note that this can also be a port range.

            {slurm.conf#OPT_SlurmctldPort}
        slurmctld_primary_off_program (str):
            This program is executed when a `slurmctld` daemon running as the
            primary server becomes a backup server.

            {slurm.conf#OPT_SlurmctldPrimaryOffProg}
        slurmctld_primary_on_program (str):
            This program is executed when a `slurmctld` daemon running as a
            backup server becomes the primary server.

            {slurm.conf#OPT_SlurmctldPrimaryOnProg}
        slurmctld_syslog_level (str):
            Level of detail that the `slurmctld` logs to the syslog.

            {slurm.conf#OPT_SlurmctldSyslogDebug}
        slurmctld_timeout (int):
            The interval, in seconds, that the backup controller waits for the
            primary controller to respond before assuming control.

            {slurm.conf#OPT_SlurmctldTimeout}
        slurmctld_parameters (dict[str, Union[str, int, bool]]):
            Options set for the `slurmctld`.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_SlurmctldParameters}
        slurmd_log_level (str):
            Level of detail `slurmd` is logging.

            {slurm.conf#OPT_SlurmdDebug}
        slurmd_log_file (str):
            Pathname of the file where `slurmd` writes logs to.

            {slurm.conf#OPT_SlurmdLogFile}
        slurmd_parameters (dict[str, Union[str, int, bool]]):
            Parameters for the `slurmd`.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_SlurmdParameters}
        slurmd_pid_file (str):
            Pathname of a file into which the `slurmd` daemon may write its
            process id.

            {slurm.conf#OPT_SlurmdPidFile}
        slurmd_port (int):
            Port number where `slurmd` listens to for work.

            {slurm.conf#OPT_SlurmdPort}
        slurmd_spool_directory (str):
            Pathname of a directory into which the `slurmd` daemon's state
            information and batch job script information are written.

            {slurm.conf#OPT_SlurmdSpoolDir}
        slurmd_syslog_level (str):
            Level of detail that the `slurmd` logs to the syslog.

            {slurm.conf#OPT_SlurmdSyslogDebug}
        slurmd_timeout (int):
            The interval, in seconds, that `slurmctld` waits for `slurmd` to
            respond before configuring that node's state to `DOWN`.

            {slurm.conf#OPT_SlurmdTimeout}
        srun_epilog (str):
            Pathname of an executable to be run by `srun` following the
            completion of a job step.

            {slurm.conf#OPT_SrunEpilog}
        srun_port_range (str):
            Ports `srun` creates to communicate with the `slurmctld`, the
            `slurmstepd` and to handle the application I/O.

            {slurm.conf#OPT_SrunPortRange}
        srun_prolog (str):
            Pathname of an executable to be run by `srun` prior to the launch
            of a job step.

            {slurm.conf#OPT_SrunProlog}
        state_save_location (str):
            Pathname of a directory where `slurmctld` saves its state.

            {slurm.conf#OPT_StateSaveLocation}
        suspend_exclude_nodes (str):
            Specifies the nodes which are to not be placed in power save mode,
            even if the node remains idle for an extended period of time.

            {slurm.conf#OPT_SuspendExcNodes}
        suspend_exclude_partitions (str):
            Specifies the partitions whose nodes are to not be placed in power
            save mode, even if the node remains idle for an extended period of
            time.

            {slurm.conf#OPT_SuspendExcParts}
        suspend_exclude_states (list[str]):
            Specifies node states that are not to be powered down
            automatically.

            {slurm.conf#OPT_SuspendExcStates}
        suspend_program (str):
            Program that will be executed when a node remains idle for an
            extended period of time.

            {slurm.conf#OPT_SuspendProgram}
        suspend_rate (int):
            Number of nodes per minute that are placed into power save mode.

            {slurm.conf#OPT_SuspendRate}
        suspend_time (int):
            Nodes which remain idle or down for this number of seconds will be
            placed into power save mode.

            {slurm.conf#OPT_SuspendTime}
        suspend_timeout (int):
            Maximum time permitted (in seconds) between when a node suspend
            request is issued and when the node is shutdown.

            {slurm.conf#OPT_SuspendTimeout}
        switch_type (str):
            Identifies the type of switch or interconnect used for application
            communications.

            {slurm.conf#OPT_SwitchType}
        switch_parameters (dict[str, Union[str, int, bool]]):
            Optional parameters for the switch plugin.

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_SwitchParameters}
        task_epilog (str):
            Pathname of a program to be executed as the slurm job's owner after
            termination of each task.

            {slurm.conf#OPT_TaskEpilog}
        task_plugin (str):
            Identifies the type of task launch plugin, typically used to
            provide resource management within a node.

            {slurm.conf#OPT_TaskPlugin}
        task_plugin_parameters (list[str]):
            Optional Parameters for `task_plugin`.

            {slurm.conf#OPT_TaskPluginParam}
        task_prolog (str):
            Pathname of a program to be executed as the slurm job's owner prior
            to initiation of each task.

            {slurm.conf#OPT_TaskProlog}
        tls_parameters (list[str]):
            Parameters for `tls_type`.
        tls_type (str):
            TLS Plugin used.
        tcp_timeout (int):
            Time permitted for TCP connection to be established.

            {slurm.conf#OPT_TCPTimeout}
        temporary_filesystem (str):
            Pathname of the file system available to user jobs for temporary
            storage.

            {slurm.conf#OPT_TmpFS}
        topology_parameters (dict[str, Union[str, int, bool]]):
            List of network topology options

            If a value in this dict is `True`, it means this parameter does not
            have any additional options specified, and is just an "enabled"
            option.

            {slurm.conf#OPT_TopologyParam}
        topology_plugin (str):
            Identifies the plugin to be used for determining the network
            topology and optimizing job allocations to minimize network
            contention.

            {slurm.conf#OPT_TopologyPlugin}
        tree_width (int):
            Specifies the width of the virtual network tree `slurmd` uses for
            communication.

            {slurm.conf#OPT_TreeWidth}
        unkillable_step_program (str):
            Program that will be executed when the processes in a job step are
            determined unkillable.

            {slurm.conf#OPT_UnkillableStepProgram}
        unkillable_step_timeout (int):
            The length of time, in seconds, that Slurm will wait before
            deciding that processes in a job step are unkillable.

            {slurm.conf#OPT_UnkillableStepTimeout}
        track_wckey (bool):
            Whether WCKeys are tracked or not.

            {slurm.conf#OPT_TrackWCKey}
        use_pam (bool):
            Whether PAM (Pluggable Authentication Modules for Linux) will be
            enabled or not.

            {slurm.conf#OPT_UsePAM}
        version (str):
            Version as returned by the `slurmctld`.
        virtual_memory_size_factor (int):
            Specifies the job's or job step's virtual memory limit as a
            percentage of its real memory limit.

            {slurm.conf#OPT_VSizeFactor}
        wait_time (int):
            Specifies how many seconds the srun command should by default wait
            after the first task terminates before terminating all remaining
            tasks.

            {slurm.conf#OPT_WaitTime}
        x11_parameters (list[str]):
            Parameters for Slurm's built-in X11 forwarding implementation.

            {slurm.conf#OPT_X11Parameters}
    """
    cdef:
        slurm_conf_t *ptr
        CgroupConfig _cgroup_config
        AccountingGatherConfig _accounting_gather_config
        MPIConfig _mpi_config


# Documentation for the attributes in the MPIConfig class have
# been largely taken from the official mpi.conf overview at:
# https://slurm.schedmd.com/mpi.conf.html
#
# Therefore, the following Copyright notices that mpi.conf has (see
# https://slurm.schedmd.com/mpi.conf.html#SECTION_COPYING), are also
# listed here:
#
# Copyright (C) 2022 SchedMD LLC.
cdef class MPIConfig:
    """Slurm MPI Config (`mpi.conf`)

    Attributes:
        pmix_cli_tmp_dir_base (str):
            Directory to have PMIx use for temporary files.

            {mpi.conf#OPT_PMIxCliTmpDirBase}
        pmix_coll_fence (str):
            Defines the type of fence to use for collecting inter-node data.

            {mpi.conf#OPT_PMIxCollFence}
        pmix_debug (bool):
            Whether debug logging for the PMIx Plugin is enabled or not.

            {mpi.conf#OPT_PMIxDebug}
        pmix_direct_conn (bool):
            Whether direct launching of tasks is enabled or not.

            {mpi.conf#OPT_PMIxDirectConn}
        pmix_direct_conn_early (bool):
            Whether early connection to a parent node are allowed or not.

            {mpi.conf#OPT_PMIxDirectConnEarly}
        pmix_direct_conn_ucx (bool):
            Whether PMIx is allowed to use UCX for communication.

            {mpi.conf#OPT_PMIxDirectConnUCX}
        pmix_direct_same_arch (bool):
            Whether additional communication optimizations are enabled when
            `pmix_direct_conn` is also set to `True`, also assuming all nodes
            of the job have the same architecture.

            {mpi.conf#OPT_PMIxDirectSameArch}
        pmix_environment (dict[str, Union[str, int]]):
            Environment variables to bet set in the Job environment, used by
            PMIx.

            {mpi.conf#OPT_PMIxEnv}
        pmix_fence_barrier (bool):
            Whether to fence inter-node communication for data collection.

            {mpi.conf#OPT_PMIxFenceBarrier}
        pmix_net_devices_ucx (str):
            Type of network device to use for communication.

            {mpi.conf#OPT_PMIxNetDevicesUCX}
        pmix_timeout (int):
            The maximum time (in seconds) allowed for communication between
            hosts to take place.

            {mpi.conf#OPT_PMIxTimeout}
        pmix_tls_ucx (list[str]):
            List of values for the UCX_TLS variable which restrict the
            transports to use.

            {mpi.conf#OPT_PMIxTlsUCX}
    """
    cdef public:
        pmix_cli_tmp_dir_base
        pmix_coll_fence
        pmix_debug
        pmix_direct_conn
        pmix_direct_conn_early
        pmix_direct_conn_ucx
        pmix_direct_same_arch
        pmix_environment
        pmix_fence_barrier
        pmix_net_devices_ucx
        pmix_timeout
        pmix_tls_ucx

    @staticmethod
    cdef MPIConfig from_ptr(void *ptr)


# Documentation for the attributes in the CgroupConfig class have
# been largely taken from the official cgroup.conf overview at:
# https://slurm.schedmd.com/cgroup.conf.html
#
# Therefore, the following Copyright notices that cgroup.conf has (see
# https://slurm.schedmd.com/cgroup.conf.html#SECTION_COPYING), are also
# listed here:
#
# Copyright (C) 2010-2012 Lawrence Livermore National Security. (cf,
# pyslurm/slurm/SLURM_DISCLAIMER).
# Copyright (C) 2010-2022 SchedMD LLC.
cdef class CgroupConfig:
    """Slurm Cgroup Config (`cgroup.conf`)

    Attributes:
        mountpoint (str):
            Specifies the PATH under which cgroup controllers should be
            mounted.

            {cgroup.conf#OPT_CgroupMountpoint}
        plugin (str):
            Specifies the plugin to be used when interacting with the cgroup
            subsystem.

            {cgroup.conf#OPT_CgroupPlugin}
        systemd_timeout (int):
            Maximum time (in milliseconds) that Slurm will wait for the slurmd
            scope to be ready before failing.

            {cgroup.conf#OPT_SystemdTimeout}
        ignore_systemd (bool):
            If `True`, it will avoid any call to dbus and contact with systemd,
            and cgroup hierarchy preparation is done manually. Only for
            `cgroup/v2`

            {cgroup.conf#OPT_IgnoreSystemd}
        ignore_systemd_on_failure (bool):
            Similar to `ignore_systemd`, but only in the case that a dbus call
            does not succeed. Only for `cgroup/v2`.

            {cgroup.conf#OPT_IgnoreSystemdOnFailure}
        enable_controllers (bool):
            When enabled, `slurmd` gets the available controllers from root`s
            cgroup.controllers file located in `mountpoint`.

            {cgroup.conf#OPT_EnableControllers}
        allowed_ram_space (int):
            Constrains the job/step cgroup RAM to this percentage of the
            allocated memory.

            {cgroup.conf#OPT_AllowedRAMSpace}
        allowed_swap_space (float):
            Constrain the job cgroup swap space to this percentage of the
            allocated memory.

            {cgroup.conf#OPT_AllowedSwapSpace}
        constrain_cores (bool):
            When `True`, then constrain allowed cores to the subset of
            allocated resources.

            {cgroup.conf#OPT_ConstrainCores}
        constrain_devices (bool):
            When `True`, then constrain the job's allowed devices based on GRES
            allocated resources.

            {cgroup.conf#OPT_ConstrainDevices}
        constrain_ram_space (bool):
            When `True`, then constrain the job's RAM usage by setting the
            memory soft limit to the allocated memory and the hard limit to the
            allocated memory * `allowed_ram_space`.

            {cgroup.conf#OPT_ConstrainRAMSpace}
        constrain_swap_space (bool):
            When `True`, then constrain the job's swap space usage.

            {cgroup.conf#OPT_ConstrainSwapSpace}
        max_ram_percent (float):
            Upper bound in percent of total RAM (configured RealMemory of the
            node) on the RAM constraint for a job.

            {cgroup.conf#OPT_MaxRAMPercent}
        max_swap_percent (float):
            Upper bound (in percent of total RAM, configured RealMemory of the
            node) on the amount of RAM+Swap that may be used for a job.

            {cgroup.conf#OPT_MaxSwapPercent}
        memory_swappiness (float):
            Configures the kernel's priority for swapping out anonymous pages
            verses file cache pages for the job cgroup. Only for `cgroup/v1`.
            A value of `-1.0` means that the kernel's default swappiness value
            will be used.

            {cgroup.conf#OPT_MemorySwappiness}
        min_ram_space (int):
            Lower bound (in Mebibytes) on the memory limits defined by
            `allowed_ram_space` and `allowed_swap_space`.

            {cgroup.conf#OPT_MinRAMSpace}
        signal_children_processes (bool):
            When `True`, then send signals (for cancelling, suspending,
            resuming, etc.) to all children processes in a job/step.

            {cgroup.conf#OPT_SignalChildrenProcesses}
    """
    cdef public:
        mountpoint
        plugin
        systemd_timeout
        ignore_systemd
        ignore_systemd_on_failure
        enable_controllers

        allowed_ram_space
        allowed_swap_space
        constrain_cores
        constrain_devices
        constrain_ram_space
        constrain_swap_space
        max_ram_percent
        max_swap_percent
        memory_swappiness
        min_ram_space

        signal_children_processes

    @staticmethod
    cdef CgroupConfig from_ptr(void *ptr)


# Documentation for the attributes in the AccountingGatherConfig class have
# been largely taken from the official acct_gather.conf overview at:
# https://slurm.schedmd.com/acct_gather.conf.html
#
# Therefore, the following Copyright notices that acct_gather.conf has (see
# https://slurm.schedmd.com/acct_gather.conf.html#SECTION_COPYING), are also
# listed here:
#
# Copyright (C) 2012-2013 Bull.
# Copyright (C) 2012-2022 SchedMD LLC.
cdef class AccountingGatherConfig:
    """Slurm Accounting Gather Config (`acct_gather.conf`)

    Attributes:
        energy_ipmi_frequency (int):
            Number of seconds between BMC access samples or XCC samples,
            depending on the plugin used.

            {acct_gather.conf#OPT_EnergyIPMIFrequency}
        energy_ipmi_calc_adjustment (bool):
            When `True`, the consumption between the last BMC access sample and
            a step consumption update is approximated to get more accurate task
            consumption.

            {acct_gather.conf#OPT_EnergyIPMICalcAdjustment}
        energy_ipmi_power_sensors (str):
            IDs of the sensors to used.

            {acct_gather.conf#OPT_EnergyIPMIPowerSensors}
        energy_ipmi_user_name (str):
            BMC Username

            {acct_gather.conf#OPT_EnergyIPMIUsername}
        energy_ipmi_password (str):
            BMC Password

            {acct_gather.conf#OPT_EnergyIPMIPassword}
        energy_ipmi_timeout (int):
            Timeout, in seconds, for initializing the IPMI XCC context for a
            new gathering thread. Default is 10 seconds.

            {acct_gather.conf#OPT_EnergyIPMITimeout}
        profile_hdf5_dir (str):
            Path to the shared folder into which the `acct_gather_profile`
            plugin will write detailed data.

            {acct_gather.conf#OPT_ProfileHDF5Dir}
        profile_hdf5_default (list[str]):
            List of data types to be collected for each job submission.

            {acct_gather.conf#OPT_ProfileHDF5Default}
        profile_influxdb_database (str):
            InfluxDB v1.x database name where profiling information is to be
            written. InfluxDB v2.x bucket name where profiling information is
            to be written.

            {acct_gather.conf#OPT_ProfileInfluxDBDatabase}
        profile_influxdb_default (list[str]):
            List of data types to be collected for each job submission.

            {acct_gather.conf#OPT_ProfileInfluxDBDefault}
        profile_influxdb_host (str):
            The hostname of the machine where the InfluxDB instance is executed
            and the port used by the HTTP API.

            {acct_gather.conf#OPT_ProfileInfluxDBHost}
        profile_influxdb_password (str):
            Password for `profile_influxdb_user`

            {acct_gather.conf#OPT_ProfileInfluxDBPass}
        profile_influxdb_rtpolicy (str):
            The InfluxDB v1.x retention policy name for the database configured
            in ProfileInfluxDBDatabase option. The InfluxDB v2.x retention
            policy bucket name for the database configured in
            ProfileInfluxDBDatabase option.

            {acct_gather.conf#OPT_ProfileInfluxDBRTPolicy}
        profile_influxdb_user (str):
            InfluxDB username that should be used to gain access to the
            database configured in `profile_influxdb_database`.

            {acct_gather.conf#OPT_ProfileInfluxDBRTUser}
        profile_influxdb_timeout (int):
            The maximum time in seconds that an HTTP query to the InfluxDB
            server can take.

            {acct_gather.conf#OPT_ProfileInfluxDBTimeout}
        infiniband_ofed_port (int):
            Represents the port number of the local Infiniband card that we are
            willing to monitor.

            {acct_gather.conf#OPT_InfinibandOFEDPort}
        sysfs_interfaces (list[str]):
            List of interface names to collect statistics from.

            {acct_gather.conf#OPT_SysfsInterfaces}
    """
    cdef public:
        energy_ipmi_frequency
        energy_ipmi_calc_adjustment
        energy_ipmi_power_sensors
        energy_ipmi_user_name
        energy_ipmi_password
        energy_ipmi_timeout

        profile_hdf5_dir
        profile_hdf5_default

        profile_influxdb_database
        profile_influxdb_default
        profile_influxdb_host
        profile_influxdb_password
        profile_influxdb_rtpolicy
        profile_influxdb_user
        profile_influxdb_timeout

        infiniband_ofed_port

        sysfs_interfaces

    @staticmethod
    cdef AccountingGatherConfig from_ptr(void *ptr)
