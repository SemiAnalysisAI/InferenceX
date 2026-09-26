# cython: c_string_type=unicode, c_string_encoding=default
# cython: language_level=3

from libcpp cimport bool
from cpython.version cimport PY_MAJOR_VERSION

from posix.unistd cimport (
    uid_t,
    pid_t,
    gid_t,
)

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

from libc.string cimport (
    strlen,
    memcpy,
)


cdef extern from '<netinet/in.h>' nogil:
    ctypedef struct sockaddr_storage:
        pass


cdef extern from '<stdio.h>' nogil:
    ctypedef struct FILE
    cdef FILE *stdout


cdef extern from '<time.h>' nogil:
    ctypedef long time_t
    double difftime(time_t time1, time_t time2)
    time_t time(time_t *t)


cdef extern from '<Python.h>' nogil:
    cdef FILE *PyFile_AsFile(object file)
    char *__FILE__
    cdef int __LINE__
    char *__FUNCTION__


cdef extern from '<pthread.h>' nogil:
    ctypedef struct pthread_mutex_t:
        pass

    ctypedef struct pthread_cond_t:
        pass

    ctypedef struct pthread_t:
        pass


cdef extern from *:
    ctypedef struct slurm_job_credential
    ctypedef struct switch_jobinfo
    ctypedef struct select_jobinfo
    ctypedef struct select_nodeinfo
    ctypedef struct jobacctinfo
    ctypedef struct allocation_msg_thread
    ctypedef struct sbcast_cred
    ctypedef struct hostlist
    ctypedef struct xlist
    ctypedef struct listIterator
    ctypedef struct slurm_step_ctx_struct
    ctypedef struct slurm_ctl_conf_t


# Header definitions
include "slurm_version.h.pxi"
include "slurm_errno.h.pxi"
include "slurm.h.pxi"
include "slurmdb.h.pxi"

# Any other definitions which are not directly in
# the header files, but exported in libslurm.so
include "extra.pxi"

# Any self defined helper functions.
# Just keeping them around here for now. Ideally they shouldn't be
# within this slurm c-api package, and should be defined somewhere else.
include "helpers.pxi"
