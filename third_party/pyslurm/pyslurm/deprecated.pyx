# cython: profile=False
# cython: language_level=3
# cython: auto_pickle=False
import os
import re
import sys
import time as p_time

from socket import gethostname
from collections import defaultdict
from pwd import getpwnam, getpwuid

from libc.errno cimport errno, EAGAIN
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.stdint cimport int64_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport strlen, strcpy, memset, memcpy
from posix.unistd cimport getuid, getgid
from cpython cimport bool

cdef extern from 'stdio.h':
    ctypedef struct FILE
    cdef FILE *stdout

cdef extern from 'Python.h':
    cdef FILE *PyFile_AsFile(object file)

cdef extern from 'time.h' nogil:
    ctypedef long time_t
    double difftime(time_t time1, time_t time2)
    time_t time(time_t *t)

cdef extern from "sys/wait.h" nogil:
    int WIFSIGNALED(int status)
    int WTERMSIG(int status)
    int WEXITSTATUS(int status)
    int WIFEXITED(int status)

cdef extern from "<sys/types.h>" nogil:
    ctypedef long id_t

cdef extern from "<sys/resource.h>" nogil:
    enum: PRIO_PROCESS
    int getpriority(int, id_t)

#cdef extern from *:
#    # deprecated backwards compatibility declaration
#    ctypedef char*  const_char_ptr  "const char*"
#    ctypedef char** const_char_pptr "const char**"

cdef extern from "alps_cray.h" nogil:
    cdef int ALPS_CRAY_SYSTEM

import builtins as __builtin__

from pyslurm cimport slurm
from pyslurm.slurm cimport xmalloc
import pyslurm.core.job

include "pydefines/slurm_errno_defines.pxi"
include "pydefines/slurm_errno_enums.pxi"

include "pydefines/slurm_defines.pxi"
include "pydefines/slurm_enums.pxi"

include "pydefines/slurmdb_defines.pxi"
include "pydefines/slurmdb_enums.pxi"

#
# Slurm Macros as Cython inline functions
#

cdef inline SLURM_VERSION_NUMBER():
    return slurm.SLURM_VERSION_NUMBER

cdef inline SLURM_VERSION_MAJOR(a):
    return ((a >> 16) & 0xff)

cdef inline SLURM_VERSION_MINOR(a):
    return ((a >> 8) & 0xff)

cdef inline SLURM_VERSION_MICRO(a):
    return (a & 0xff)

cdef inline SLURM_VERSION_NUM(a):
    return (((SLURM_VERSION_MAJOR(a)) << 16) +
            ((SLURM_VERSION_MINOR(a)) << 8) +
            (SLURM_VERSION_MICRO(a)))

DEF MAX_RETRIES = 15

# SLURM_ID_HASH
# Description:
#   Creates a hash of a Slurm JOBID and STEPID
#   The JOB STEP ID is in the top 32 bits of the hash with the job id occupying
#   the lower 32 bits.
#
#   IN  _jobid -- SLURM's JOB ID (uint32_t)
#   IN  _stepid -- SLURM's JOB STEP ID (uint32_t)
#   RET id_hash -- (uint64_t)


cdef inline SLURM_ID_HASH(_jobid, _stepid):
    return <uint64_t>(<uint64_t>_stepid << 32 + _jobid)

cdef inline SLURM_ID_HASH_JOB_ID(hash_id):
    return <uint32_t>(hash_id & 0x00000000FFFFFFFF)

cdef inline SLURM_ID_HASH_STEP_ID(hash_id):
    return <uint32_t>(hash_id >> 32)


# Convert a hash ID to its legacy (pre-17.11) equivalent
# Used for backward compatibility for Cray PMI

cdef inline SLURM_ID_HASH_LEGACY(hash_id):
    return ((hash_id >> 32) * 10000000000 + (hash_id & 0x00000000FFFFFFFF))


# Helpers
cdef inline listOrNone(char* value, sep_char):
    if value is NULL:
        return []

    if not sep_char:
        return value.decode("UTF-8", "replace")

    if sep_char == '':
        return value.decode("UTF-8", "replace")

    return value.decode("UTF_8", "replace").split(sep_char)

cdef inline listOfStrings(char **value):
    l = []
    i = 0
    if value != NULL:
        while value[i] != NULL:
            l.append(stringOrNone(value[i], ''))
            i += 1

    return(tuple(l))

cdef inline stringOrNone(char* value, value2):
    if value is NULL:
        if value2 is '':
            return None
        return value2
    return value.decode("UTF-8", "replace")


cdef inline int16orNone(uint16_t value):
    if value is NO_VAL16:
        return None
    else:
        return value


cdef inline int32orNone(uint32_t value):
    if value is NO_VAL:
        return None
    else:
        return value


cdef inline int64orNone(uint64_t value):
    if value is NO_VAL64:
        return None
    else:
        return value


cdef inline int16orUnlimited(uint16_t value, return_type):
    if value is INFINITE16:
        if return_type is "int":
            return None
        else:
            return "UNLIMITED"
    else:
        if return_type is "int":
            return value
        else:
            return str(value)


#
# Defined job states
#

cdef inline IS_JOB_PENDING(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_PENDING)

cdef inline IS_JOB_RUNNING(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_RUNNING)

cdef inline IS_JOB_SUSPENDED(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_SUSPENDED)

cdef inline IS_JOB_COMPLETE(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_COMPLETE)

cdef inline IS_JOB_CANCELLED(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_CANCELLED)

cdef inline IS_JOB_FAILED(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_FAILED)

cdef inline IS_JOB_TIMEOUT(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_TIMEOUT)

cdef inline IS_JOB_NODE_FAILED(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_NODE_FAIL)

cdef inline IS_JOB_DEADLINE(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_DEADLINE)

cdef inline IS_JOB_OOM(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) == JOB_OOM)

cdef inline IS_JOB_POWERING_UP_NODE(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_STATE_BASE)

#
# Derived job states
#

cdef inline IS_JOB_COMPLETING(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_COMPLETING)

cdef inline IS_JOB_CONFIGURING(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_CONFIGURING)

cdef inline IS_JOB_STARTED(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) > JOB_PENDING)

cdef inline IS_JOB_FINISHED(slurm.slurm_job_info_t _X):
    return ((_X.job_state & JOB_STATE_BASE) > JOB_SUSPENDED)

cdef inline IS_JOB_COMPLETED(slurm.slurm_job_info_t _X):
    return (IS_JOB_FINISHED(_X) and ((_X.job_state & JOB_COMPLETING) == 0))

cdef inline IS_JOB_RESIZING(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_RESIZING)

cdef inline IS_JOB_REQUEUED(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_REQUEUE)

cdef inline IS_JOB_FED_REQUEUED(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_REQUEUE_FED)

cdef inline IS_JOB_REVOKED(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_REVOKED)

cdef inline IS_JOB_SIGNALING(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_SIGNALING)

cdef inline IS_JOB_STAGE_OUT(slurm.slurm_job_info_t _X):
    return (_X.job_state & JOB_STAGE_OUT)

#
# Defined node states
#

cdef inline IS_NODE_UNKNOWN(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_UNKNOWN

cdef inline IS_NODE_DOWN(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_DOWN

cdef inline IS_NODE_IDLE(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_IDLE

cdef inline IS_NODE_ALLOCATED(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_ALLOCATED

cdef inline IS_NODE_ERROR(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_ERROR

cdef inline IS_NODE_MIXED(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_MIXED

cdef inline IS_NODE_FUTURE(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_BASE) == NODE_STATE_FUTURE

#
# Derived node states
#

cdef inline IS_NODE_CLOUD(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_CLOUD)

cdef inline IS_NODE_DRAIN(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_DRAIN)

cdef inline IS_NODE_DRAINING(slurm.node_info_t _X):
    return ((_X.node_state & NODE_STATE_DRAIN) and
            (IS_NODE_ALLOCATED(_X) or IS_NODE_MIXED(_X)))

cdef inline IS_NODE_DRAINED(slurm.node_info_t _X):
    return (IS_NODE_DRAIN(_X) and not IS_NODE_DRAINING(_X))

cdef inline IS_NODE_COMPLETING(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_COMPLETING)

cdef inline IS_NODE_NO_RESPOND(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_NO_RESPOND)

cdef inline IS_NODE_POWERED_DOWN(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_POWERED_DOWN)

cdef inline IS_NODE_POWERING_DOWN(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_POWERING_DOWN)

cdef inline IS_NODE_FAIL(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_FAIL)

cdef inline IS_NODE_POWERING_UP(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_POWERING_UP)

cdef inline IS_NODE_MAINT(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_MAINT)

cdef inline IS_NODE_REBOOT_REQUESTED(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_REBOOT_REQUESTED)

cdef inline IS_NODE_REBOOT_ISSUED(slurm.node_info_t _X):
    return (_X.node_state & NODE_STATE_REBOOT_ISSUED)

ctypedef struct config_key_pair_t:
    char *name
    char *value


#
# Cython Wrapper Functions
#


def get_controllers():
    """Get information about slurm controllers.

    Returns:
        (tuple): Name of primary controller, Name of backup controllers
    """
    cdef:
        slurm.slurm_conf_t *slurm_ctl_conf_ptr = NULL
        slurm.time_t Time = <slurm.time_t>NULL
        int apiError = 0
        int errCode = slurm.slurm_load_ctl_conf(Time, &slurm_ctl_conf_ptr)
        uint32_t length = 0

    if errCode != 0:
        apiError = slurm_get_errno()
        raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)

    control_machs = []
    if slurm_ctl_conf_ptr is not NULL:

        if slurm_ctl_conf_ptr.control_machine is not NULL:
            length = slurm_ctl_conf_ptr.control_cnt
            for index in range(length):
                primary = stringOrNone(slurm_ctl_conf_ptr.control_machine[index], '')
                control_machs.append(primary)

        slurm.slurm_free_ctl_conf(slurm_ctl_conf_ptr)

    return control_machs


def is_controller(Host=None):
    """Return slurm controller status for host.

    Args:
        Host (str): Name of host to check

    Returns:
        (str): None, "primary" or "backup"
    """
    control_machs = get_controllers()
    if not Host:
        Host = gethostname()

    index = control_machs.index(Host)
    if index == -1:
        return None
    if index == 0:
        return 'primary'
    if index > 0:
        return 'backup'


def slurm_api_version():
    """Return the slurm API version number.

    Returns:
        (tuple): A tuple of version_major, version_minor, version_micro
    """
    cdef long version = slurm.SLURM_VERSION_NUMBER

    return (SLURM_VERSION_MAJOR(version),
            SLURM_VERSION_MINOR(version),
            SLURM_VERSION_MICRO(version))


def slurm_load_slurmd_status():
    """Issue RPC to get and load the status of Slurmd daemon.

    Returns:
        (str): Slurmd information
    """
    cdef:
        dict Status = {}, Status_dict = {}
        slurm.slurmd_status_t *slurmd_status = NULL
        int errCode = slurm.slurm_load_slurmd_status(&slurmd_status)

    if errCode == slurm.SLURM_SUCCESS:
        hostname = stringOrNone(slurmd_status.hostname, '')
        Status_dict['actual_boards'] = slurmd_status.actual_boards
        Status_dict['booted'] = slurmd_status.booted
        Status_dict['actual_cores'] = slurmd_status.actual_cores
        Status_dict['actual_cpus'] = slurmd_status.actual_cpus
        Status_dict['actual_real_mem'] = slurmd_status.actual_real_mem
        Status_dict['actual_sockets'] = slurmd_status.actual_sockets
        Status_dict['actual_threads'] = slurmd_status.actual_threads
        Status_dict['actual_tmp_disk'] = slurmd_status.actual_tmp_disk
        Status_dict['hostname'] = hostname
        Status_dict['last_slurmctld_msg'] = slurmd_status.last_slurmctld_msg
        Status_dict['pid'] = slurmd_status.pid
        Status_dict['slurmd_debug'] = slurmd_status.slurmd_debug
        Status_dict['slurmd_logfile'] = stringOrNone(slurmd_status.slurmd_logfile, '')
        Status_dict['step_list'] = stringOrNone(slurmd_status.step_list, '')
        Status_dict['version'] = stringOrNone(slurmd_status.version, '')

        Status[hostname] = Status_dict

    slurm.slurm_free_slurmd_status(slurmd_status)

    return Status


cpdef long slurm_get_rem_time(uint32_t JobID=0) except? -1:
    """Get the remaining time in seconds for a slurm job step.

    Args:
        JobID (int): The job id.

    Returns:
        int: Remaining time in seconds or -1 on error
    """
    cdef int apiError = 0
    cdef long errCode = slurm.slurm_get_rem_time(JobID)

    if errCode != 0:
        apiError = slurm_get_errno()
        raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)

    return errCode


cpdef time_t slurm_get_end_time(uint32_t JobID=0) except? -1:
    """Get the end time in seconds for a slurm job step.

    Args:
        JobID (int): The job id.

    Returns:
        int: Remaining time in seconds or -1 on error
    """
    cdef time_t EndTime = -1
    cdef int apiError = 0
    cdef int errCode = slurm.slurm_get_end_time(JobID, &EndTime)

    if errCode != 0:
        apiError = slurm_get_errno()
        raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)

    return EndTime


cpdef int slurm_job_node_ready(uint32_t JobID=0) except? -1:
    """Return if a node could run a slurm job now if dispatched.

    Args:
        JobID (int): The job id.

    Returns:
        int: Node ready code.
    """
    cdef int apiError = 0
    cdef int errCode = slurm.slurm_job_node_ready(JobID)

    return errCode


def slurm_pid2jobid(uint32_t JobPID=0):
    """Get the slurm job id from a process id.

    Args:
        JobPID (int): Job process id

    Returns:
        int: 0 for success or a slurm error code
    """
    cdef:
        uint32_t JobID = 0
        int apiError = 0
        int errCode = slurm.slurm_pid2jobid(JobPID, &JobID)

    if errCode != 0:
        apiError = slurm_get_errno()
        raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)

    return errCode, JobID


#
# Slurm Error Class
#


class SlurmError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(slurm.slurm_strerror(self.value))


#
# Slurm Error Functions
#


def slurm_get_errno():
    """Return the slurm error as set by a slurm API call.

    Returns:
        (int): Current slurm error number
    """
    return errno


def slurm_strerror(int Errno=0):
    """Return slurm error message represented by a given slurm error number.

    Args:
        Errno (int): slurm error number.

    Returns:
        (str): slurm error string
    """
    cdef char* errMsg = slurm.slurm_strerror(Errno)

    return "%s" % errMsg


def slurm_perror(char* Msg=''):
    """Print to standard error the supplied header.

    Header is followed by a colon, followed by a text description of the last
    Slurm error code generated.

    Args:
        Msg (str): slurm program error String
    """
    slurm.slurm_perror(Msg)


#
# Hostlist Class
#


cdef class hostlist:
    """Wrapper for Slurm hostlist functions."""

    cdef slurm.hostlist_t *hl

    def __cinit__(self):
        self.hl = NULL

    def __dealloc__(self):
        self.destroy()

    def create(self, hostnames=None):
        if not hostnames:
            self.hl = slurm.slurm_hostlist_create(NULL)
        else:
            b_hostnames = hostnames.encode("UTF-8")
            self.hl = slurm.slurm_hostlist_create(b_hostnames)
        if not self.hl:
            raise ValueError("No memory")
        else:
            return True

    def destroy(self):
        if self.hl is not NULL:
            slurm.slurm_hostlist_destroy(self.hl)
            self.hl = NULL

    def count(self):
        return slurm.slurm_hostlist_count(self.hl)

    def get_list(self):
        """Get the list of hostnames composing the hostlist.

        For example with a hostlist created with "tux[1-3]" -> [ 'tux1',
        tux2', 'tux3' ].

        Returns:
            (list): The list of hostnames in case of success or None on error.
        """
        cdef:
            slurm.hostlist_t *hlist = NULL
            char *hostlist_s = NULL
            char *tmp_str = NULL
            list host_list = None
            unsigned int nb_hosts
            unsigned int host_index

        py_string = ''

        if self.hl is not NULL:
            # make a copy of self.hl since slurm.slurm_hostlist_shift() is destructive.
            tmp_str = slurm.slurm_hostlist_ranged_string_xmalloc(self.hl)
            if tmp_str is not NULL:
                hlist = slurm.slurm_hostlist_create(tmp_str)
                nb_hosts = slurm.slurm_hostlist_count(hlist)
                host_list = []
                for host_index in range(nb_hosts):
                    hostlist_s = slurm.slurm_hostlist_shift(hlist)
                    py_string = hostlist_s
                    free(hostlist_s)
                    host_list.append(py_string)

                slurm.xfree(tmp_str)
                slurm.slurm_hostlist_destroy(hlist)

        return host_list

    def get(self):
        cdef:
            char *hostlist_s = NULL
            char *tmp_str = NULL

        py_string = None
        if self.hl is not NULL:
            tmp_str = slurm.slurm_hostlist_ranged_string_xmalloc(self.hl)
            if tmp_str is not NULL:
                hostlist_s = <char *>malloc(strlen(tmp_str) + 1)
                strcpy(hostlist_s, tmp_str)
                py_string = hostlist_s
                free(hostlist_s)
                slurm.xfree(tmp_str)

        return py_string

    def ranged_string(self):
        if self.hl is not NULL:
            return stringOrNone(slurm.slurm_hostlist_ranged_string_xmalloc(self.hl), '')

    def find(self, hostname):
        if self.hl is not NULL:
            b_hostname = hostname.encode("UTF-8")
            return slurm.slurm_hostlist_find(self.hl, b_hostname)

    def pop(self):
        if self.hl is not NULL:
            return stringOrNone(slurm.slurm_hostlist_shift(self.hl), '')

    def shift(self):
        return self.pop()

    def push(self, hosts):
        if self.hl is not NULL:
            b_hosts = hosts.encode("UTF-8")
            return slurm.slurm_hostlist_push(self.hl, b_hosts)

    def push_host(self, host):
        if self.hl is not NULL:
            b_host = host.encode("UTF-8")
            return slurm.slurm_hostlist_push_host(self.hl, b_host)

    def uniq(self):
        if self.hl is not NULL:
            slurm.slurm_hostlist_uniq(self.hl)


#
# Trigger Get/Set/Update Class
#


cdef class trigger:

    def set(self, dict trigger_dict):
        """Set or create a slurm trigger.

        Args:
            trigger_dict (dict): A populated dictionary of trigger information

        Returns:
            (int): 0 for success or -1 for error, and the slurm error code is
                set appropriately.
        """
        cdef:
            slurm.trigger_info_t trigger_set
            int errCode = -1

        slurm.slurm_init_trigger_msg(&trigger_set)

        if 'jobid' in trigger_dict:
            JobId = trigger_dict['jobid']
            trigger_set.res_type = TRIGGER_RES_TYPE_JOB  # 1

            if isinstance(JobId, int):
                JobId = str(JobId)

            b_JobId = JobId.encode("UTF-8")
            trigger_set.res_id = b_JobId

            if 'fini' in trigger_dict:
                trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_FINI  # 0x0010
            if 'offset' in trigger_dict:
                trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_TIME  # 0x0008

        elif 'node' in trigger_dict:
            trigger_set.res_type = TRIGGER_RES_TYPE_NODE
            if trigger_dict['node'] == '':
                trigger_set.res_id = '*'
            else:
                b_node = trigger_dict['node'].encode("UTF-8")
                trigger_set.res_id = b_node

        trigger_set.offset = 0x8000
        if 'offset' in trigger_dict:
            trigger_set.offset = trigger_set.offset + trigger_dict['offset']

        b_program = trigger_dict['program'].encode("UTF-8")
        trigger_set.program = b_program

        event = trigger_dict['event']
        if event == 'burst_buffer':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_BURST_BUFFER

        if event == 'drained':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_DRAINED    # 0x0100

        if event == 'down':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_DOWN       # 0x0002

        if event == 'fail':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_FAIL       # 0x0004

        if event == 'up':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_UP         # 0x0001

        if event == 'idle':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_IDLE       # 0x0080

        if event == 'reconfig':
            trigger_set.trig_type = trigger_set.trig_type | TRIGGER_TYPE_RECONFIG   # 0x0020

        while slurm.slurm_set_trigger(&trigger_set):
            slurm.slurm_perror('slurm_set_trigger')
            # EAGAIN
            if slurm_get_errno() != 11:
                errCode = slurm_get_errno()
                return errCode

            p_time.sleep(5)

        return 0

    def get(self):
        """Get the information on slurm triggers.

        Returns:
            (dict): Dictionary, where keys are the trigger IDs
        """
        cdef:
            slurm.trigger_info_msg_t *trigger_get = NULL
            int errCode = slurm.slurm_get_triggers(&trigger_get)
            dict Triggers = {}, Trigger_dict

        if errCode == 0:
            for record in trigger_get.trigger_array[:trigger_get.record_count]:
                trigger_id = record.trig_id

                Trigger_dict = {}
                Trigger_dict['flags'] = record.flags
                Trigger_dict['trig_id'] = trigger_id
                Trigger_dict['res_type'] = record.res_type
                Trigger_dict['res_id'] = stringOrNone(record.res_id, '')
                Trigger_dict['trig_type'] = record.trig_type
                Trigger_dict['offset'] = record.offset - 0x8000
                Trigger_dict['user_id'] = record.user_id
                Trigger_dict['program'] = stringOrNone(record.program, '')

                Triggers[trigger_id] = Trigger_dict

            slurm.slurm_free_trigger_msg(trigger_get)

        return Triggers

    def clear(self, TriggerID=0, UserID=slurm.NO_VAL, ID=0):
        """Clear or remove a slurm trigger.

        Args:
            TriggerID (str): Trigger Identifier
            UserID (str): User Identifier
            ID (str): Job Identifier

        Returns:
            (int): 0 for success or a slurm error code
        """
        cdef:
            slurm.trigger_info_t trigger_clear
            int errCode

        if not (TriggerID or UserID or ID):
            raise ValueError("One of `TriggerID` or `UserID` or `ID` must be provided.")

        trigger_clear.trig_id = TriggerID
        trigger_clear.user_id = UserID

        if ID:
            trigger_clear.res_type = TRIGGER_RES_TYPE_JOB  # 1
            b_job_id = str(ID).encode("UTF-8")
            trigger_clear.res_id = b_job_id

        errCode = slurm.slurm_clear_trigger(&trigger_clear)

        if errCode != slurm.SLURM_SUCCESS:
            raise ValueError(stringOrNone(slurm.slurm_strerror(errCode), ''), errCode)

        return errCode


#
# QOS Class
#


cdef class qos:
    """Access/update slurm QOS information."""

    cdef:
        void *dbconn
        dict _QOSDict
        slurm.list_t *_QOSList

    def __cinit__(self):
        self.dbconn = <void *>NULL
        self._QOSDict = {}

    def __dealloc__(self):
        self.__destroy()

    cdef __destroy(self):
        """QOS Destructor method."""
        self._QOSDict = {}

    def load(self):
        """Load slurm QOS information."""
        self.__load()

    cdef int __load(self) except? -1:
        """Load slurm QOS list."""
        cdef:
            slurm.slurmdb_qos_cond_t *new_qos_cond = NULL
            int apiError = 0
            void* dbconn = slurm.slurmdb_connection_get(NULL)
            slurm.list_t *QOSList = slurm.slurmdb_qos_get(dbconn, new_qos_cond)

        if QOSList is NULL:
            apiError = slurm_get_errno()
            raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)
        else:
            self._QOSList = QOSList

        slurm.slurmdb_connection_close(&dbconn)
        return 0

    def lastUpdate(self):
        """Return last time (sepoch seconds) the QOS data was updated.

        Returns:
            int: epoch seconds
        """
        return self._lastUpdate

    def ids(self):
        """Return the QOS IDs from retrieved data.

        Returns:
            (dict): Dictionary of QOS IDs
        """
        return self._QOSDict.keys()

    def get(self):
        """Get slurm QOS information.

        Returns:
            (dict): Dictionary whose key is the QOS ID
        """
        self.__load()
        self.__get()

        return self._QOSDict

    cdef __get(self):
        cdef:
            slurm.list_t *qos_list = NULL
            slurm.list_itr_t *iters = NULL
            int i = 0
            int listNum = 0
            dict Q_dict = {}

        if self._QOSList is not NULL:
            listNum = slurm.slurm_list_count(self._QOSList)
            iters = slurm.slurm_list_iterator_create(self._QOSList)

            for i in range(listNum):
                qos = <slurm.slurmdb_qos_rec_t *>slurm.slurm_list_next(iters)
                name = stringOrNone(qos.name, '')

                # QOS infos
                QOS_info = {}

                if name:
                    QOS_info['description'] = stringOrNone(qos.description, '')
                    QOS_info['flags'] = qos.flags
                    QOS_info['grace_time'] = qos.grace_time
                    QOS_info['grp_jobs'] = qos.grp_jobs
                    QOS_info['grp_submit_jobs'] = qos.grp_submit_jobs
                    QOS_info['grp_tres'] = stringOrNone(qos.grp_tres, '')
                    # QOS_info['grp_tres_ctld']
                    QOS_info['grp_tres_mins'] = stringOrNone(qos.grp_tres_mins, '')
                    # QOS_info['grp_tres_mins_ctld']
                    QOS_info['grp_tres_run_mins'] = stringOrNone(qos.grp_tres_run_mins, '')
                    # QOS_info['grp_tres_run_mins_ctld']
                    QOS_info['grp_wall'] = qos.grp_wall
                    QOS_info['max_jobs_pu'] = qos.max_jobs_pu
                    QOS_info['max_submit_jobs_pu'] = qos.max_submit_jobs_pu
                    QOS_info['max_tres_mins_pj'] = stringOrNone(qos.max_tres_mins_pj, '')
                    # QOS_info['max_tres_min_pj_ctld']
                    QOS_info['max_tres_pj'] = stringOrNone(qos.max_tres_pj, '')
                    # QOS_info['max_tres_min_pj_ctld']
                    QOS_info['max_tres_pn'] = stringOrNone(qos.max_tres_pn, '')
                    # QOS_info['max_tres_min_pn_ctld']
                    QOS_info['max_tres_pu'] = stringOrNone(qos.max_tres_pu, '')
                    # QOS_info['max_tres_min_pu_ctld']
                    QOS_info['max_tres_run_mins_pu'] = stringOrNone(
                        qos.max_tres_run_mins_pu, '')

                    QOS_info['max_wall_pj'] = qos.max_wall_pj
                    QOS_info['min_tres_pj'] = stringOrNone(qos.min_tres_pj, '')
                    # QOS_info['min_tres_pj_ctld']
                    QOS_info['name'] = name
                    # QOS_info['*preempt_bitstr'] =
                    # QOS_info['preempt_list'] = qos.preempt_list

                    QOS_info['priority'] = qos.priority
                    QOS_info['usage_factor'] = qos.usage_factor
                    QOS_info['usage_thres'] = qos.usage_thres

                    # NB - Need to add code to decode types of grp_tres_ctld (uint64t list) etc

                if name:
                    Q_dict[name] = QOS_info

            slurm.slurm_list_iterator_destroy(iters)
            slurm.slurm_list_destroy(self._QOSList)

        self._QOSDict = Q_dict

#
# slurmdbd jobs Class
#
cdef class slurmdb_jobs:
    """Access Slurmdbd Jobs information."""
    cdef:
        void* db_conn
        slurm.slurmdb_job_cond_t *job_cond

    def __cinit__(self):
        self.job_cond = <slurm.slurmdb_job_cond_t *>xmalloc(sizeof(slurm.slurmdb_job_cond_t))
        self.db_conn = slurm.slurmdb_connection_get(NULL)

    def __dealloc__(self):
        slurm.xfree(self.job_cond)
        slurm.slurmdb_connection_close(&self.db_conn)

    def get(self, jobids=[], userids=[], starttime=0, endtime=0, flags = None,
            db_flags = None, clusters = []):
        """Get Slurmdb information about some jobs.

        Input formats for start and end times:
            *   today or tomorrow
            *   midnight, noon, teatime (4PM)
            *   HH:MM [AM|PM]
            *   MMDDYY or MM/DD/YY or MM.DD.YY
            *   YYYY-MM-DD[THH[:MM[:SS]]]
            *   now + count [minutes | hours | days | weeks]
            *

        Invalid time input results in message to stderr and return value of
        zero.

        Args:
            jobids (list): Ids of the jobs to search. Defaults to all jobs.
            starttime (int, optional): Select jobs eligible after this
                timestamp
            endtime (int, optional): Select jobs eligible before this
                timestamp
            userids (list): List of userids
            flags (int): Flags
            db_flags (int): DB Flags
            clusters (list): List of clusters

        Returns:
            (dict): Dictionary whose key is the JOBS ID
        """
        cdef:
            int i = 0
            int listNum = 0
            int apiError = 0
            dict J_dict = {}
            slurm.list_t *JOBSList
            slurm.list_itr_t *iters = NULL


        if clusters:
            self.job_cond.cluster_list = slurm.slurm_list_create(NULL)
            for _cluster in clusters:
                _cluster = _cluster.encode("UTF-8")
                slurm.slurm_addto_char_list_with_case(self.job_cond.cluster_list, _cluster, False)

        if db_flags:
            if isinstance(db_flags, int):
                self.job_cond.db_flags = db_flags
        else:
            self.job_cond.db_flags = slurm.SLURMDB_JOB_FLAG_NOTSET

        if flags:
            if isinstance(flags, int):
                self.job_cond.flags = flags

        if jobids:
            self.job_cond.step_list = slurm.slurm_list_create(NULL)
            for _jobid in jobids:
                if isinstance(_jobid, int):
                    _jobid = str(_jobid).encode("UTF-8")
                else:
                    _jobid = _jobid.encode("UTF-8")
                slurm.slurm_addto_step_list(self.job_cond.step_list, _jobid)

        if userids:
            self.job_cond.userid_list = slurm.slurm_list_create(NULL)
            for _userid in userids:
                if isinstance(_userid, int):
                    _userid = str(_userid).encode("UTF-8")
                else:
                    _userid = _userid.encode("UTF-8")
                slurm.slurm_addto_char_list_with_case(self.job_cond.userid_list, _userid, False)

        if starttime:
            self.job_cond.usage_start = slurm.slurm_parse_time(starttime, 1)
            errno = slurm_get_errno()
            if errno == slurm.ESLURM_INVALID_TIME_VALUE:
                raise ValueError(slurm.slurm_strerror(errno), errno)

        if endtime:
            self.job_cond.usage_end = slurm.slurm_parse_time(endtime, 1)
            errno = slurm_get_errno()
            if errno == slurm.ESLURM_INVALID_TIME_VALUE:
                raise ValueError(slurm.slurm_strerror(errno), errno)

        JOBSList = slurm.slurmdb_jobs_get(self.db_conn, self.job_cond)

        if JOBSList is NULL:
            apiError = slurm_get_errno()
            raise ValueError(slurm.slurm_strerror(apiError), apiError)

        listNum = slurm.slurm_list_count(JOBSList)
        iters = slurm.slurm_list_iterator_create(JOBSList)

        for i in range(listNum):
            job = <slurm.slurmdb_job_rec_t *>slurm.slurm_list_next(iters)

            JOBS_info = {}
            if job is not NULL:
                jobid = job.jobid
                JOBS_info['account'] = stringOrNone(job.account, '')
                JOBS_info['alloc_nodes'] = job.alloc_nodes
                JOBS_info['array_job_id'] = job.array_job_id
                JOBS_info['array_max_tasks'] = job.array_max_tasks
                JOBS_info['array_task_id'] = job.array_task_id
                JOBS_info['array_task_str'] = stringOrNone(job.array_task_str, '')
                JOBS_info['associd'] = job.associd
                JOBS_info['blockid'] = stringOrNone(job.blockid, '')
                JOBS_info['cluster'] = stringOrNone(job.cluster, '')
                JOBS_info['constraints'] = stringOrNone(job.constraints, '')
                JOBS_info['container'] = stringOrNone(job.container, '')
                JOBS_info['derived_ec'] = job.derived_ec
                JOBS_info['derived_es'] = stringOrNone(job.derived_es, '')
                JOBS_info['elapsed'] = job.elapsed
                JOBS_info['eligible'] = job.eligible
                JOBS_info['end'] = job.end
                JOBS_info['env'] = stringOrNone(job.env, '')
                JOBS_info['exitcode'] = job.exitcode
                JOBS_info['gid'] = job.gid
                JOBS_info['jobid'] = job.jobid
                JOBS_info['jobname'] = stringOrNone(job.jobname, '')
                JOBS_info['partition'] = stringOrNone(job.partition, '')
                JOBS_info['nodes'] = stringOrNone(job.nodes, '')
                JOBS_info['priority'] = job.priority
                JOBS_info['qosid'] = job.qosid
                JOBS_info['req_cpus'] = job.req_cpus

                if job.req_mem & slurm.MEM_PER_CPU:
                    JOBS_info['req_mem'] = job.req_mem & (~slurm.MEM_PER_CPU)
                    JOBS_info['req_mem_per_cpu'] = True
                else:
                    JOBS_info['req_mem'] = job.req_mem
                    JOBS_info['req_mem_per_cpu'] = False

                JOBS_info['requid'] = job.requid
                JOBS_info['resvid'] = job.resvid
                JOBS_info['resv_name'] = stringOrNone(job.resv_name,'')
                JOBS_info['script'] = stringOrNone(job.script,'')
                JOBS_info['show_full'] = job.show_full
                JOBS_info['start'] = job.start
                JOBS_info['state'] = job.state
                JOBS_info['state_str'] = stringOrNone(slurm.slurm_job_state_string(job.state), '')

                # TRES are reported as strings in the format `TRESID=value` where TRESID is one of:
                # TRES_CPU=1, TRES_MEM=2, TRES_ENERGY=3, TRES_NODE=4, TRES_BILLING=5, TRES_FS_DISK=6, TRES_VMEM=7, TRES_PAGES=8
                # Example: '1=0,2=745472,3=0,6=1949,7=7966720,8=0'

                # add job steps
                JOBS_info['steps'] = {}
                step_dict = JOBS_info['steps']

                stepsNum = slurm.slurm_list_count(job.steps)
                stepsIter = slurm.slurm_list_iterator_create(job.steps)
                for i in range(stepsNum):
                    step = <slurm.slurmdb_step_rec_t *>slurm.slurm_list_next(stepsIter)
                    step_info = {}
                    if step is not NULL:
                        step_id = step.step_id.step_id

                        step_info['container'] = stringOrNone(step.container, '')
                        step_info['elapsed'] = step.elapsed
                        step_info['end'] = step.end
                        step_info['exitcode'] = step.exitcode

                        # Don't add this unless you want to create an endless recursive structure
                        # step_info['job_ptr'] = JOBS_Info # job's record

                        step_info['nnodes'] = step.nnodes
                        step_info['nodes'] = stringOrNone(step.nodes, '')
                        step_info['ntasks'] = step.ntasks
                        step_info['pid_str'] = stringOrNone(step.pid_str, '')
                        step_info['req_cpufreq_min'] = step.req_cpufreq_min
                        step_info['req_cpufreq_max'] = step.req_cpufreq_max
                        step_info['req_cpufreq_gov'] = step.req_cpufreq_gov
                        step_info['requid'] = step.requid
                        step_info['start'] = step.start
                        step_info['state'] = step.state
                        step_info['state_str'] = stringOrNone(slurm.slurm_job_state_string(step.state), '')

                        # TRES are reported as strings in the format `TRESID=value` where TRESID is one of:
                        # TRES_CPU=1, TRES_MEM=2, TRES_ENERGY=3, TRES_NODE=4, TRES_BILLING=5, TRES_FS_DISK=6, TRES_VMEM=7, TRES_PAGES=8
                        # Example: '1=0,2=745472,3=0,6=1949,7=7966720,8=0'
                        step_info['stats'] = {}
                        stats = step_info['stats']
                        stats['act_cpufreq'] = step.stats.act_cpufreq
                        stats['consumed_energy'] = step.stats.consumed_energy
                        stats['tres_usage_in_max'] = stringOrNone(step.stats.tres_usage_in_max, '')
                        stats['tres_usage_in_max_nodeid'] = stringOrNone(step.stats.tres_usage_in_max_nodeid, '')
                        stats['tres_usage_in_max_taskid'] = stringOrNone(step.stats.tres_usage_in_max_taskid, '')
                        stats['tres_usage_in_min'] = stringOrNone(step.stats.tres_usage_in_min, '')
                        stats['tres_usage_in_min_nodeid'] = stringOrNone(step.stats.tres_usage_in_min_nodeid, '')
                        stats['tres_usage_in_min_taskid'] = stringOrNone(step.stats.tres_usage_in_min_taskid, '')
                        stats['tres_usage_in_tot'] = stringOrNone(step.stats.tres_usage_in_tot, '')
                        stats['tres_usage_out_ave'] = stringOrNone(step.stats.tres_usage_out_ave, '')
                        stats['tres_usage_out_max'] = stringOrNone(step.stats.tres_usage_out_max, '')
                        stats['tres_usage_out_max_nodeid'] = stringOrNone(step.stats.tres_usage_out_max_nodeid, '')
                        stats['tres_usage_out_max_taskid'] = stringOrNone(step.stats.tres_usage_out_max_taskid, '')
                        stats['tres_usage_out_min'] = stringOrNone(step.stats.tres_usage_out_min, '')
                        stats['tres_usage_out_min_nodeid'] = stringOrNone(step.stats.tres_usage_out_min_nodeid, '')
                        stats['tres_usage_out_min_taskid'] = stringOrNone(step.stats.tres_usage_out_min_taskid, '')
                        stats['tres_usage_out_tot'] = stringOrNone(step.stats.tres_usage_out_tot, '')
                        step_info['stepid'] = step_id
                        step_info['stepname'] = stringOrNone(step.stepname, '')
                        step_info['submit_line'] = stringOrNone(step.submit_line, '')
                        step_info['suspended'] = step.suspended
                        step_info['sys_cpu_sec'] = step.sys_cpu_sec
                        step_info['sys_cpu_usec'] = step.sys_cpu_usec
                        step_info['task_dist'] = step.task_dist
                        step_info['tot_cpu_sec'] = step.tot_cpu_sec
                        step_info['tot_cpu_usec'] = step.tot_cpu_usec
                        step_info['user_cpu_sec'] = step.user_cpu_sec
                        step_info['user_cpu_usec'] = step.user_cpu_usec

                        step_dict[step_id] = step_info

                slurm.slurm_list_iterator_destroy(stepsIter)

                JOBS_info['submit'] = job.submit
                JOBS_info['submit_line'] = stringOrNone(job.submit_line,'')
                JOBS_info['suspended'] = job.suspended
                JOBS_info['sys_cpu_sec'] = job.sys_cpu_sec
                JOBS_info['sys_cpu_usec'] = job.sys_cpu_usec
                JOBS_info['timelimit'] = job.timelimit
                JOBS_info['tot_cpu_sec'] = job.tot_cpu_sec
                JOBS_info['tot_cpu_usec'] = job.tot_cpu_usec
                JOBS_info['tres_alloc_str'] = stringOrNone(job.tres_alloc_str,'')
                JOBS_info['tres_req_str'] = stringOrNone(job.tres_req_str,'')
                JOBS_info['uid'] = job.uid
                JOBS_info['used_gres'] = stringOrNone(job.used_gres, '')
                JOBS_info['user'] = stringOrNone(job.user,'')
                JOBS_info['user_cpu_sec'] = job.user_cpu_sec
                JOBS_info['user_cpu_usec'] = job.user_cpu_usec
                JOBS_info['wckey'] = stringOrNone(job.wckey, '')
                JOBS_info['wckeyid'] = job.wckeyid
                JOBS_info['work_dir'] = stringOrNone(job.work_dir, '')
                J_dict[jobid] = JOBS_info

        slurm.slurm_list_iterator_destroy(iters)
        slurm.slurm_list_destroy(JOBSList)
        if clusters:
            slurm.slurm_list_destroy(self.job_cond.cluster_list)
        if userids:
            slurm.slurm_list_destroy(self.job_cond.userid_list)
        return J_dict

#
# slurmdbd Reservations Class
#
cdef class slurmdb_reservations:
    """Access Slurmdbd reservations information."""
    cdef:
        void *dbconn
        slurm.slurmdb_reservation_cond_t *reservation_cond

    def __cinit__(self):
        self.reservation_cond = <slurm.slurmdb_reservation_cond_t *>xmalloc(sizeof(slurm.slurmdb_reservation_cond_t))

    def __dealloc__(self):
        slurm.slurmdb_destroy_reservation_cond(self.reservation_cond)

    def set_reservation_condition(self, start_time, end_time):
        """Limit the next get() call to reservations that start after and
        before a certain time.

        Args:
            start_time (int): Select reservations that start after this
                unix timestamp
            end_time (int): Select reservations that end before this unix
                timestamp
        """
        if self.reservation_cond == NULL:
            self.reservation_cond = <slurm.slurmdb_reservation_cond_t *>xmalloc(sizeof(slurm.slurmdb_reservation_cond_t))

        if self.reservation_cond != NULL:
            self.reservation_cond.with_usage = 1
            self.reservation_cond.time_start = <slurm.time_t>start_time
            self.reservation_cond.time_end = <slurm.time_t>end_time
        else:
            raise MemoryError()

    def get(self):
        """Get slurm reservations information.

        Returns:
            (dict): Dictionary whose keys are the reservations ids
        """
        cdef:
            slurm.list_t *reservation_list
            slurm.list_itr_t *iters = NULL
            slurm.slurmdb_reservation_rec_t *reservation
            int i = 0
            int j = 0
            int listNum
            slurm.list_t *_resvList

        Reservation_dict = {}
        reservation_list = slurm.slurmdb_reservations_get(self.dbconn, self.reservation_cond)

        if reservation_list is not NULL:
            listNum = slurm.slurm_list_count(reservation_list)
            iters = slurm.slurm_list_iterator_create(reservation_list)

            for i in range(listNum):
                reservation = <slurm.slurmdb_reservation_rec_t *>slurm.slurm_list_next(iters)
                Reservation_rec_dict = {}

                if reservation is not NULL:
                    reservation_id = reservation.id
                    Reservation_rec_dict['name'] = stringOrNone(reservation.name, '')
                    Reservation_rec_dict['nodes'] = stringOrNone(reservation.nodes, '')
                    Reservation_rec_dict['node_index'] = stringOrNone(reservation.node_inx, '')
                    Reservation_rec_dict['associations'] = stringOrNone(reservation.assocs, '')
                    Reservation_rec_dict['cluster'] = stringOrNone(reservation.cluster, '')
                    Reservation_rec_dict['tres_str'] = stringOrNone(reservation.tres_str, '')
                    Reservation_rec_dict['reservation_id'] = reservation.id
                    Reservation_rec_dict['time_start'] = reservation.time_start
                    Reservation_rec_dict['time_start_prev'] = reservation.time_start_prev
                    Reservation_rec_dict['time_end'] = reservation.time_end
                    Reservation_rec_dict['flags'] = reservation.flags

                    if reservation.tres_list != NULL:
                        num_tres = slurm.slurm_list_count(reservation.tres_list)
                        tres_iters = slurm.slurm_list_iterator_create(reservation.tres_list)
                        tres_dict = {}
                        Reservation_rec_dict['num_tres'] = num_tres

                        for j in range(num_tres):
                            tres = <slurm.slurmdb_tres_rec_t *>slurm.slurm_list_next(tres_iters)
                            if tres is not NULL:
                                tmp_tres_dict = {}
                                tres_id = tres.id
                                tmp_tres_dict['name'] = stringOrNone(tres.name,'')
                                tmp_tres_dict['type'] = stringOrNone(tres.type,'')
                                tmp_tres_dict['rec_count'] = tres.rec_count
                                tmp_tres_dict['count'] = tres.count
                                tmp_tres_dict['tres_id'] = tres.id
                                tmp_tres_dict['alloc_secs'] = tres.alloc_secs
                                tres_dict[tres_id] = tmp_tres_dict

                        Reservation_rec_dict['tres_list'] = tres_dict
                        slurm.slurm_list_iterator_destroy(tres_iters)

                    Reservation_dict[reservation_id] = Reservation_rec_dict

            slurm.slurm_list_iterator_destroy(iters)
            slurm.slurm_list_destroy(reservation_list)

        return Reservation_dict

#
# slurmdbd clusters Class
#
cdef class slurmdb_clusters:
    """Access Slurmdbd Clusters information."""
    cdef:
        void *db_conn
        slurm.slurmdb_cluster_cond_t *cluster_cond

    def __cinit__(self):
        self.cluster_cond = <slurm.slurmdb_cluster_cond_t *>xmalloc(sizeof(slurm.slurmdb_cluster_cond_t))
        slurm.slurmdb_init_cluster_cond(self.cluster_cond, 0)
        self.db_conn = slurm.slurmdb_connection_get(NULL)

    def __dealloc__(self):
        slurm.slurmdb_destroy_cluster_cond(self.cluster_cond)
        slurm.slurmdb_connection_close(&self.db_conn)

    def set_cluster_condition(self, start_time, end_time):
        """Limit the next get() call to clusters that existed after and before
        a certain time.

        Args:
            start_time (int): Select clusters that existed after this unix
                timestamp
            end_time (int): Select clusters that existed before this unix
                timestamp
        """
        if self.cluster_cond == NULL:
            self.cluster_cond = <slurm.slurmdb_cluster_cond_t *>xmalloc(sizeof(slurm.slurmdb_cluster_cond_t))

        if self.cluster_cond != NULL:
            slurm.slurmdb_init_cluster_cond(self.cluster_cond, 0)
            self.cluster_cond.with_deleted = 1
            self.cluster_cond.with_usage = 1
            self.cluster_cond.usage_start = <slurm.time_t>start_time
            self.cluster_cond.usage_end = <slurm.time_t>end_time
        else:
            raise MemoryError()

    def get(self):
        """Get slurm clusters information.

        Returns:
            (dict): Dictionary whose keys are the clusters ids
        """
        cdef:
            slurm.list_t *clusters_list
            slurm.list_itr_t *iters = NULL
            slurm.slurmdb_cluster_rec_t *cluster = NULL
            int rc = slurm.SLURM_SUCCESS
            int i = 0
            int j = 0
            int listNum

        Cluster_dict = {}
        cluster_list = slurm.slurmdb_clusters_get(self.db_conn, self.cluster_cond)

        if cluster_list is not NULL:
            listNum = slurm.slurm_list_count(cluster_list)
            iters = slurm.slurm_list_iterator_create(cluster_list)

            for i in range(listNum):
                cluster = <slurm.slurmdb_cluster_rec_t *>slurm.slurm_list_next(iters)
                Cluster_rec_dict = {}

                if cluster is not NULL:
                    cluster_name = stringOrNone(cluster.name, '')
                    Cluster_rec_dict['name'] = cluster_name
                    Cluster_rec_dict['nodes'] = stringOrNone(cluster.nodes, '')
                    Cluster_rec_dict['control_host'] = stringOrNone(cluster.control_host, '')
                    Cluster_rec_dict['tres'] = stringOrNone(cluster.tres_str, '')
                    Cluster_rec_dict['control_port'] = cluster.control_port
                    Cluster_rec_dict['rpc_version'] = cluster.rpc_version
                    Cluster_rec_dict['flags'] = cluster.flags
                    Cluster_rec_dict['dimensions'] = cluster.dimensions
                    Cluster_rec_dict['classification'] = cluster.classification

                    if cluster.accounting_list != NULL:
                        num_acct = slurm.slurm_list_count(cluster.accounting_list)
                        acct_iters = slurm.slurm_list_iterator_create(cluster.accounting_list)
                        acct_dict = {}
                        Cluster_rec_dict['num_acct'] = num_acct

                        for j in range(num_acct):
                            acct_tres = <slurm.slurmdb_cluster_accounting_rec_t *>slurm.slurm_list_next(acct_iters)
                            if acct_tres is not NULL:
                                acct_tres_dict = {}
                                acct_tres_rec = <slurm.slurmdb_tres_rec_t>acct_tres.tres_rec
                                acct_tres_id = acct_tres_rec.id

                                if (acct_tres_rec.name is not NULL):
                                    acct_tres_dict['name'] = stringOrNone(acct_tres_rec.name,'')
                                if (acct_tres_rec.type is not NULL):
                                    acct_tres_dict['type'] = stringOrNone(acct_tres_rec.type,'')

                                acct_tres_dict['rec_count'] = acct_tres_rec.rec_count
                                acct_tres_dict['count'] = acct_tres_rec.count
                                acct_tres_dict['alloc_secs'] = acct_tres.alloc_secs
                                acct_tres_dict['down_secs'] = acct_tres.down_secs
                                acct_tres_dict['idle_secs'] = acct_tres.idle_secs
                                acct_tres_dict['plan_secs'] = acct_tres.plan_secs
                                acct_tres_dict['pdown_secs'] = acct_tres.pdown_secs
                                acct_tres_dict['over_secs'] = acct_tres.over_secs
                                acct_tres_dict['period_start'] = acct_tres.period_start
                                acct_dict[acct_tres_id] = acct_tres_dict

                        Cluster_rec_dict['accounting'] = acct_dict
                        slurm.slurm_list_iterator_destroy(acct_iters)

                    Cluster_dict[cluster_name] = Cluster_rec_dict

            slurm.slurm_list_iterator_destroy(iters)
            slurm.slurm_list_destroy(cluster_list)

        return Cluster_dict

#
# slurmdbd Events Class
#
cdef class slurmdb_events:
    """Access Slurmdbd events information."""
    cdef:
        void *dbconn
        slurm.slurmdb_event_cond_t *event_cond

    def __cinit__(self):
        self.event_cond = <slurm.slurmdb_event_cond_t *>xmalloc(sizeof(slurm.slurmdb_event_cond_t))

    def __dealloc__(self):
        slurm.slurmdb_destroy_event_cond(self.event_cond)

    def set_event_condition(self, start_time, end_time):
        """Limit the next get() call to conditions that existed after and
        before a certain time.

        Args:
            start_time (int): Select conditions that existed after this unix timestamp
            end_time (int): Select conditions that existed before this unix timestamp
        """
        if self.event_cond == NULL:
            self.event_cond = <slurm.slurmdb_event_cond_t *>xmalloc(sizeof(slurm.slurmdb_event_cond_t))

        if self.event_cond != NULL:
            ##self.event_cond.with_usage = 1
            self.event_cond.period_start = <slurm.time_t>start_time
            self.event_cond.period_end = <slurm.time_t>end_time
        else:
            raise MemoryError()

    def get(self):
        """Get slurm events information.

        Returns:
            (dict): Dictionary whose keys are the events ids
        """
        cdef:
            slurm.list_t *event_list
            slurm.list_itr_t *iters = NULL
            slurm.slurmdb_event_rec_t *event = NULL
            int i = 0
            int listNum = 0

        Event_dict = {}
        event_list = slurm.slurmdb_events_get(self.dbconn, self.event_cond)

        if event_list is not NULL:
            listNum = slurm.slurm_list_count(event_list)
            iters = slurm.slurm_list_iterator_create(event_list)

            for i in range(listNum):
                event = <slurm.slurmdb_event_rec_t *>slurm.slurm_list_next(iters)
                event_rec_dict = {}

                if event is not NULL:
                    event_id = event.period_start
                    event_rec_dict['cluster'] = stringOrNone(event.cluster, '')
                    event_rec_dict['cluster_nodes'] = stringOrNone(event.cluster_nodes, '')
                    event_rec_dict['node_name'] = stringOrNone(event.node_name, '')
                    event_rec_dict['reason'] = stringOrNone(event.reason, '')
                    event_rec_dict['tres_str'] = stringOrNone(event.tres_str, '')
                    event_rec_dict['event_type'] = event.event_type
                    event_rec_dict['time_start'] = event.period_start
                    event_rec_dict['time_end'] = event.period_end
                    event_rec_dict['tres_str'] = event.tres_str
                    event_rec_dict['state'] = event.state
                    event_rec_dict['reason_uid'] = event.reason_uid

                    Event_dict[event_id] = event_rec_dict

            slurm.slurm_list_iterator_destroy(iters)
            slurm.slurm_list_destroy(event_list)

        return Event_dict

#
# SlurmDB Reports (sreport)
#

cdef class slurmdb_reports:
    """Access Slurmdbd reports."""
    cdef:
        void *db_conn
        slurm.slurmdb_assoc_cond_t *assoc_cond

    def __cinit__(self):
        self.assoc_cond = <slurm.slurmdb_assoc_cond_t *>xmalloc(sizeof(slurm.slurmdb_assoc_cond_t))

    def __dealloc__(self):
        slurm.slurmdb_destroy_assoc_cond(self.assoc_cond)

    def report_cluster_account_by_user(self, starttime=None,
                                       endtime=None):
        """sreport cluster AccountUtilizationByUser

        Args:
            starttime (Union[str, int]): Start time
            endtime (Union[str, int]): Start time

        Returns:
            (dict): sreport information.
        """
        cdef:
            slurm.list_t *slurmdb_report_cluster_list = NULL
            slurm.list_itr_t *itr = NULL
            slurm.list_itr_t *cluster_itr = NULL
            slurm.list_itr_t *tres_itr = NULL
            slurm.slurmdb_cluster_cond_t cluster_cond
            slurm.slurmdb_report_assoc_rec_t *slurmdb_report_assoc = NULL
            slurm.slurmdb_report_cluster_rec_t *slurmdb_report_cluster = NULL
            slurm.slurmdb_tres_rec_t *tres
            time_t start_time
            time_t end_time
            int i
            int j

        slurm.slurmdb_init_cluster_cond(&cluster_cond, 0)
        self.assoc_cond.flags = slurm.ASSOC_COND_FLAG_SUB_ACCTS

        if starttime:
            self.assoc_cond.usage_start = slurm.slurm_parse_time(starttime, 1)

        if endtime:
            self.assoc_cond.usage_end = slurm.slurm_parse_time(endtime, 1)

        start_time = self.assoc_cond.usage_start
        end_time = self.assoc_cond.usage_end
        slurm.slurmdb_report_set_start_end_time(&start_time, &end_time)
        self.assoc_cond.usage_start = start_time
        self.assoc_cond.usage_end = end_time

        self.assoc_cond.flags |= slurm.ASSOC_COND_FLAG_WITH_USAGE
        self.assoc_cond.flags |= slurm.ASSOC_COND_FLAG_WITH_DELETED

        slurmdb_report_cluster_list = slurm.slurmdb_report_cluster_account_by_user(
            self.db_conn, self.assoc_cond
        )

        if slurmdb_report_cluster_list is NULL:
            slurm.slurmdb_destroy_assoc_cond(self.assoc_cond)
            slurm.slurm_list_destroy(slurmdb_report_cluster_list)
            slurmdb_report_cluster_list = NULL
            sys.exit(0)

        cluster_itr = slurm.slurm_list_iterator_create(slurmdb_report_cluster_list)
        Cluster_dict = {}

        for i in range(slurm.slurm_list_count(slurmdb_report_cluster_list)):
            slurmdb_report_cluster = <slurm.slurmdb_report_cluster_rec_t *>slurm.slurm_list_next(cluster_itr)
            cluster_name = stringOrNone(slurmdb_report_cluster.name, '')
            Cluster_dict[cluster_name] = {}
            itr = slurm.slurm_list_iterator_create(slurmdb_report_cluster.assoc_list)

            for j in range(slurm.slurm_list_count(slurmdb_report_cluster.assoc_list)):
                slurmdb_report_assoc = <slurm.slurmdb_report_assoc_rec_t *>slurm.slurm_list_next(itr)
                Assoc_dict = {}
                Assoc_dict["account"] = stringOrNone(slurmdb_report_assoc.acct, '')
                Assoc_dict["cluster"] = stringOrNone(slurmdb_report_assoc.cluster, '')
                Assoc_dict["parent_account"] = stringOrNone(slurmdb_report_assoc.parent_acct, '')
                Assoc_dict["user"] = stringOrNone(slurmdb_report_assoc.user, '')
                Assoc_dict["tres_list"] = []
                tres_itr = slurm.slurm_list_iterator_create(slurmdb_report_assoc.tres_list)

                for k in range(slurm.slurm_list_count(slurmdb_report_assoc.tres_list)):
                    tres = <slurm.slurmdb_tres_rec_t *>slurm.slurm_list_next(tres_itr)
                    Tres_dict = {}
                    Tres_dict["alloc_secs"] = <int>tres.alloc_secs
                    Tres_dict["rec_count"] = tres.rec_count
                    Tres_dict["count"] = <int>tres.count
                    Tres_dict["id"] = tres.id
                    Tres_dict["name"] = stringOrNone(tres.name, '')
                    Tres_dict["type"] = stringOrNone(tres.type, '')
                    Assoc_dict["tres_list"].append(Tres_dict)

                Cluster_dict[cluster_name] = Assoc_dict
                slurm.slurm_list_iterator_destroy(tres_itr)

            slurm.slurm_list_iterator_destroy(itr)

        slurm.slurm_list_iterator_destroy(cluster_itr)
        slurm.slurm_list_destroy(slurmdb_report_cluster_list)
        slurmdb_report_cluster_list = NULL

        return Cluster_dict

#
# Helper functions to convert numerical States
#


cdef inline dict __get_licenses(char *licenses):
    """Returns a dict of licenses from the slurm license string.

    Args:
        licenses (str): String containing license information

    Returns:
        dict: Dictionary of licenses and associated value.
    """
    if (licenses is NULL):
        return {}

    cdef:
        dict licDict = {}
        int i = 0
        list alist = listOrNone(licenses, ',')
        int listLen = len(alist)

    if alist:
        for i in range(listLen):
            value = 1
            try:
                key, value = alist[i].split(':')
            except:
                key = alist[i]
            licDict["%s" % key] = value

    return licDict


def get_trigger_res_type(uint16_t inx):
    """Returns a string that represents the slurm trigger res type.

    Args:
        ResType (int): Slurm trigger res state
            * TRIGGER_RES_TYPE_JOB        1
            * TRIGGER_RES_TYPE_NODE       2
            * TRIGGER_RES_TYPE_SLURMCTLD  3
            * TRIGGER_RES_TYPE_SLURMDBD   4
            * TRIGGER_RES_TYPE_DATABASE   5
            * TRIGGER_RES_TYPE_OTHER      7

    Returns:
        (str): Trigger reservation state string
    """
    return __get_trigger_res_type(inx)

cdef inline object __get_trigger_res_type(uint16_t ResType):
    rtype = 'unknown'

    if ResType == TRIGGER_RES_TYPE_JOB:
        rtype = 'job'
    elif ResType == TRIGGER_RES_TYPE_NODE:
        rtype = 'node'
    elif ResType == TRIGGER_RES_TYPE_SLURMCTLD:
        rtype = 'slurmctld'
    elif ResType == TRIGGER_RES_TYPE_SLURMDBD:
        rtype = 'slurmbdb'
    elif ResType == TRIGGER_RES_TYPE_DATABASE:
        rtype = 'database'
    elif ResType == TRIGGER_RES_TYPE_OTHER:
        rtype = 'other'

    return "%s" % rtype


def get_trigger_type(uint32_t inx):
    """Returns a string that represents the state of the slurm trigger.

    Args:
        TriggerType (int): Slurm trigger type
            * TRIGGER_TYPE_UP                 0x00000001
            * TRIGGER_TYPE_DOWN               0x00000002
            * TRIGGER_TYPE_FAIL               0x00000004
            * TRIGGER_TYPE_TIME               0x00000008
            * TRIGGER_TYPE_FINI               0x00000010
            * TRIGGER_TYPE_RECONFIG           0x00000020
            * TRIGGER_TYPE_IDLE               0x00000080
            * TRIGGER_TYPE_DRAINED            0x00000100
            * TRIGGER_TYPE_PRI_CTLD_FAIL      0x00000200
            * TRIGGER_TYPE_PRI_CTLD_RES_OP    0x00000400
            * TRIGGER_TYPE_PRI_CTLD_RES_CTRL  0x00000800
            * TRIGGER_TYPE_PRI_CTLD_ACCT_FULL 0x00001000
            * TRIGGER_TYPE_BU_CTLD_FAIL       0x00002000
            * TRIGGER_TYPE_BU_CTLD_RES_OP     0x00004000
            * TRIGGER_TYPE_BU_CTLD_AS_CTRL    0x00008000
            * TRIGGER_TYPE_PRI_DBD_FAIL       0x00010000
            * TRIGGER_TYPE_PRI_DBD_RES_OP     0x00020000
            * TRIGGER_TYPE_PRI_DB_FAIL        0x00040000
            * TRIGGER_TYPE_PRI_DB_RES_OP      0x00080000
            * TRIGGER_TYPE_BURST_BUFFER       0x00100000

    Returns:
        (str): Trigger state string
    """
    return __get_trigger_type(inx)

cdef inline object __get_trigger_type(uint32_t TriggerType):
    rtype = 'unknown'

    if TriggerType == TRIGGER_TYPE_UP:
        rtype = 'up'
    elif TriggerType == TRIGGER_TYPE_DOWN:
        rtype = 'down'
    elif TriggerType == TRIGGER_TYPE_FAIL:
        rtype = 'fail'
    elif TriggerType == TRIGGER_TYPE_TIME:
        rtype = 'time'
    elif TriggerType == TRIGGER_TYPE_FINI:
        rtype = 'fini'
    elif TriggerType == TRIGGER_TYPE_RECONFIG:
        rtype = 'reconfig'
    elif TriggerType == TRIGGER_TYPE_IDLE:
        rtype = 'idle'
    elif TriggerType == TRIGGER_TYPE_DRAINED:
        rtype = 'drained'
    elif TriggerType == TRIGGER_TYPE_PRI_CTLD_FAIL:
        rtype = 'primary_slurmctld_failure'
    elif TriggerType == TRIGGER_TYPE_PRI_CTLD_RES_OP:
        rtype = 'primary_slurmctld_resumed_operation'
    elif TriggerType == TRIGGER_TYPE_PRI_CTLD_RES_CTRL:
        rtype = 'primary_slurmctld_resumed_control'
    elif TriggerType == TRIGGER_TYPE_PRI_CTLD_ACCT_FULL:
        rtype = 'primary_slurmctld_acct_buffer_full'
    elif TriggerType == TRIGGER_TYPE_BU_CTLD_FAIL:
        rtype = 'backup_ctld_failure'
    elif TriggerType == TRIGGER_TYPE_BU_CTLD_RES_OP:
        rtype = 'backup_ctld_resumed_operation'
    elif TriggerType == TRIGGER_TYPE_BU_CTLD_AS_CTRL:
        rtype = 'backup_ctld_assumed_control'
    elif TriggerType == TRIGGER_TYPE_PRI_DBD_FAIL:
        rtype = 'primary_slurmdbd_failure'
    elif TriggerType == TRIGGER_TYPE_PRI_DBD_RES_OP:
        rtype = 'primary_slurmdbd_resumed_operation'
    elif TriggerType == TRIGGER_TYPE_PRI_DB_FAIL:
        return 'primary_database_failure'
    elif TriggerType == TRIGGER_TYPE_PRI_DB_RES_OP:
        rtype = 'primary_database_resumed_operation'
    elif TriggerType == TRIGGER_TYPE_BURST_BUFFER:
        rtype = 'burst_buffer'

    return "%s" % rtype


def get_rm_partition_state(int inx):
    """Returns a string that represents the partition state.

    Args:
        inx (int): Slurm partition state

    Returns:
        (str): Partition state string
    """
    return __get_rm_partition_state(inx)


cdef inline object __get_rm_partition_state(int inx):
    cdef list state = [
        'Free',
        'Configuring',
        'Ready',
        'Busy',
        'Deallocating',
        'Error',
        'Nav'
    ]

    rm_part_state = 'Unknown'
    try:
        rm_part_state = state[inx]
    except:
        pass

    return "%s" % rm_part_state


cdef inline object __get_partition_state(int inx, int extended=0):
    """Returns a string that represents the state of the partition.

    Args:
        inx (int): Slurm partition type
        extended (int): extended flag

    Returns:
        str: Partition state
    """
    cdef:
        int drain_flag = (inx & 0x0200)
        int comp_flag = (inx & 0x0400)
        int no_resp_flag = (inx & 0x0800)
        int power_flag = (inx & 0x1000)

    inx = (inx & 0x00ff)

    state = '?'

    if (drain_flag):
        if (comp_flag or (inx == 4)):
            state = 'Draining'
            if (no_resp_flag and extended):
                state = 'Draining*'
        else:
            state = 'Drained'
            if (no_resp_flag and extended):
                state = 'Drained*'
        return state

    if (inx == 1):
        state = 'Down'
        if (no_resp_flag and extended):
            state = 'Down*'
    elif (inx == 3):
        state = 'Allocated'
        if (no_resp_flag and extended):
            state = 'Allocated*'
        elif (comp_flag and extended):
            state = 'Allocated+'
        elif (comp_flag):
            state = 'Completing'
            if (no_resp_flag and extended):
                state = 'Completing*'
    elif (inx == 2):
        state = 'Idle'
        if (no_resp_flag and extended):
            state = 'Idle*'
        elif (power_flag and extended):
            state = 'Idle~'
    elif (inx == 0):
        state = 'Unknown'
        if (no_resp_flag and extended):
            state = 'Unknown*'

    return "%s" % state


def get_partition_mode(uint16_t flags=0, uint16_t max_share=0):
    """Returns a string represents the state of the partition mode.

    Args:
        flags (int): Flags
        max_share (int): Max share

    Returns:
        (dict): Partition mode dict
    """
    return __get_partition_mode(flags, max_share)

cdef inline dict __get_partition_mode(uint16_t flags=0, uint16_t max_share=0):
    cdef:
        dict mode = {}
        uint16_t force = max_share & SHARED_FORCE
        uint16_t val = max_share & (~SHARED_FORCE)

    if (flags & PART_FLAG_DEFAULT):
        mode['Default'] = 1
    else:
        mode['Default'] = 0

    if (flags & PART_FLAG_HIDDEN):
        mode['Hidden'] = 1
    else:
        mode['Hidden'] = 0

    if (flags & PART_FLAG_NO_ROOT):
        mode['DisableRootJobs'] = 1
    else:
        mode['DisableRootJobs'] = 0

    if (flags & PART_FLAG_ROOT_ONLY):
        mode['RootOnly'] = 1
    else:
        mode['RootOnly'] = 0

    if val == 0:
        mode['Shared'] = "EXCLUSIVE"
    elif force:
        mode['Shared'] = "FORCED:" + str(val)
    elif val == 1:
        mode['Shared'] = "NO"
    else:
        mode['Shared'] = "YES:" + str(val)

    if (flags & PART_FLAG_LLN):
        mode['LLN'] = 1
    else:
        mode['LLN'] = 0

    if (flags & PART_FLAG_EXCLUSIVE_USER):
        mode['ExclusiveUser'] = 1
    else:
        mode['ExclusiveUser'] = 0

    return mode


def epoch2date(epochSecs):
    """Convert epoch secs to a python time string.

    Args:
        epochSecs (int): Seconds since epoch

    Returns:
        (str): Date str
    """
    try:
        dateTime = p_time.gmtime(epochSecs)
        return "%s" % p_time.strftime("%a %b %d %H:%M:%S %Y", dateTime)
    except:
        pass


class Dict(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, Dict)

    def __repr__(self):
        return dict.__repr__(self)


#
# Slurm Controller License Class
#


cdef class licenses:
    """Access slurm controller license information."""

    cdef:
        slurm.license_info_msg_t *_msg
        slurm.time_t _lastUpdate
        uint16_t _ShowFlags
        dict _licDict

    def __cinit__(self):
        self._msg = NULL
        self._ShowFlags = slurm.SHOW_ALL
        self._lastUpdate = <time_t> NULL

    def __dealloc__(self):
        """Free the memory allocated by load licenses method."""
        pass

    def lastUpdate(self):
        """Return last time (epoch seconds) license data was updated.

        Returns:
            (int): Epoch seconds
        """
        return self._lastUpdate

    def ids(self):
        """Return the current license names from retrieved license data.

        This method calls slurm_load_licenses to retrieve license information
        from the controller.  slurm_free_license_info_msg is used to free the
        license message buffer.

        Returns:
            (dict): Dictionary of licenses
        """
        cdef:
            int rc
            int apiError
            uint32_t i
            list all_licenses

        rc = slurm.slurm_load_licenses(<time_t> NULL, &self._msg,
                                       self._ShowFlags)

        if rc == slurm.SLURM_SUCCESS:
            all_licenses = []
            self._lastUpdate = self._msg.last_update

            for i in range(self._msg.num_lic):
                all_licenses.append(self._msg.lic_array[i].name)
            slurm.slurm_free_license_info_msg(self._msg)
            self._msg = NULL
            return all_licenses
        else:
            apiError = slurm_get_errno()
            raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)

    def get(self):
        """Get full license information from the slurm controller.

        This method calls slurm_load_licenses to retrieve license information
        from the controller.  slurm_free_license_info_msg is used to free the
        license message buffer.

        Returns:
            (dict): Dictionary whose key is the license name
        """
        cdef:
            int rc
            int apiError
            dict License_dict

        rc = slurm.slurm_load_licenses(<time_t> NULL, &self._msg,
                                       self._ShowFlags)

        if rc == slurm.SLURM_SUCCESS:
            self._licDict = {}
            self._lastUpdate = self._msg.last_update

            for record in self._msg.lic_array[:self._msg.num_lic]:
                License_dict = {}
                license_name = stringOrNone(record.name, '')
                License_dict["total"] = record.total
                License_dict["in_use"] = record.in_use
                License_dict["available"] = record.available
                License_dict["remote"] = record.remote
                self._licDict[license_name] = License_dict
            slurm.slurm_free_license_info_msg(self._msg)
            self._msg = NULL
            return self._licDict
        else:
            apiError = slurm_get_errno()
            raise ValueError(stringOrNone(slurm.slurm_strerror(apiError), ''), apiError)
