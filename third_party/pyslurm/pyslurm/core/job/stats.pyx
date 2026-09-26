#########################################################################
# job/stats.pyx - interface to retrieve slurm job realtime stats
#########################################################################
# Copyright (C) 2024 Toni Harzendorf <toni.harzendorf@gmail.com>
# Copyright (C) 2008 Lawrence Livermore National Security.
#
#########################################################################
# The main logic for this file's code was taken from:
# https://github.com/SchedMD/slurm/blob/42a05b1bb4a7719944ee26adc6bc5d73e2d36823/src/sstat/sstat.c#L109
#
# The code has been modified a bit and translated to Cython syntax. Slurm's
# sstat was originally written by Morris Jette <jette1@llnl.gov>
#
# Slurm is licensed under the GNU General Public License. For the full text of
# Slurm's License, please see here: pyslurm/slurm/SLURM_LICENSE
#
#########################################################################
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

from pyslurm.core.error import verify_rpc
from pyslurm.utils.helpers import nodelist_to_range_str


cdef load_single(JobStep step):
    cdef:
        # jobacctinfo_t is the opaque data type provided in slurm.h
        # jobacctinfo is the actual (partial) re-definition of the jobacctinfo
        # type
        #
        # This is because we need to have access to jobacct.tres_list,
        # otherwise we cannot call jobacctinfo_aggregate. Also we want to have
        # the values of user_cpu_sec and sys_cpu_sec.
        jobacctinfo_t *total_jobacct = NULL
        jobacctinfo *stat_jobacct = NULL

        job_step_stat_t *step_stat = NULL
        job_step_stat_response_msg_t *stat_resp = NULL
        assoc_mgr_lock_t locks
        slurmdb_step_rec_t db_step
        SlurmList stats_list
        SlurmListItem stat_list_ptr
        char *usage_tmp = NULL
        int rc = slurm.SLURM_SUCCESS
        int ntasks = 0
        list nodes = []

    rc = slurm_job_step_stat(&step.ptr.step_id, NULL,
                             step.ptr.start_protocol_ver, &stat_resp)
    if rc != slurm.SLURM_SUCCESS:
        slurm_job_step_stat_response_msg_free(stat_resp)
        if rc == slurm.ESLURM_INVALID_JOB_ID:
            return None
        else:
            verify_rpc(rc)

    memset(&locks, 0, sizeof(assoc_mgr_lock_t))
    memset(&db_step, 0, sizeof(slurmdb_step_rec_t))
    memset(&db_step.stats, 0, sizeof(slurmdb_stats_t))

    stats_list = SlurmList.wrap(stat_resp.stats_list, owned=False)
    for stat_list_ptr in stats_list:
        step_stat = <job_step_stat_t*>stat_list_ptr.data
        # Casting jobacctinfo_t to jobacctinfo... hoping this is sane to do
        stat_jobacct = <jobacctinfo*>step_stat.jobacct

        if not step_stat.step_pids or not step_stat.step_pids.node_name:
            continue

        node = cstr.to_unicode(step_stat.step_pids.node_name)
        if step_stat.step_pids.pid_cnt > 0:
            for i in range(step_stat.step_pids.pid_cnt):
                if node not in step.pids:
                    step.pids[node] = []

                step.pids[node].append(step_stat.step_pids.pid[i])

        nodes.append(node)
        ntasks += step_stat.num_tasks
        if step_stat.jobacct:
            if not assoc_mgr_tres_list and stat_jobacct.tres_list:
                locks.tres = WRITE_LOCK
                assoc_mgr_lock(&locks)
                assoc_mgr_post_tres_list(stat_jobacct.tres_list)
                assoc_mgr_unlock(&locks)
                stat_jobacct.tres_list = NULL

            if not total_jobacct:
                total_jobacct = jobacctinfo_create(NULL)

            jobacctinfo_aggregate(total_jobacct, step_stat.jobacct)

            db_step.user_cpu_sec += stat_jobacct.user_cpu_sec
            db_step.user_cpu_usec += stat_jobacct.user_cpu_usec
            db_step.sys_cpu_sec += stat_jobacct.sys_cpu_sec
            db_step.sys_cpu_usec += stat_jobacct.sys_cpu_usec

    if total_jobacct:
        jobacctinfo_2_stats(&db_step.stats, total_jobacct)
        jobacctinfo_destroy(total_jobacct)

    if ntasks:
        db_step.stats.act_cpufreq /= <double>ntasks

        usage_tmp = db_step.stats.tres_usage_in_ave
        db_step.stats.tres_usage_in_ave = slurmdb_ave_tres_usage(usage_tmp, ntasks)
        xfree(usage_tmp)

        usage_tmp = db_step.stats.tres_usage_out_ave
        db_step.stats.tres_usage_out_ave = slurmdb_ave_tres_usage(usage_tmp, ntasks)
        xfree(usage_tmp)

    step.stats = JobStepStatistics.from_ptr(
            &db_step,
            nodes,
            step.cpus,
            step.run_time if step.run_time else 0,
            is_live=True,
    )

    slurm_job_step_stat_response_msg_free(stat_resp)
    slurmdb_free_slurmdb_stats_members(&db_step.stats)
