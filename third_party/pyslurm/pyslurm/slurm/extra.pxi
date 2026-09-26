#
# Structs that are not in the Slurm headers, which need to be redefined
# in order to implement certain features.
#
# For example: to communicate with the slurmctld directly in order
# to retrieve the actual batch-script as a string.
#
# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/persist_conn.h#L53
ctypedef enum persist_conn_type_t:
    PERSIST_TYPE_NONE = 0
    PERSIST_TYPE_DBD
    PERSIST_TYPE_FED
    PERSIST_TYPE_HA_CTL
    PERSIST_TYPE_HA_DBD
    PERSIST_TYPE_ACCT_UPDATE

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/persist_conn.h#L62
ctypedef struct persist_msg_t:
    void *conn
    void *data
    uint16_t msg_type

ctypedef int (*_persist_conn_t_callback_proc)(void *arg, persist_msg_t *msg, buf_t **out_buffer)

ctypedef void (*_persist_conn_t_callback_fini)(void *arg)

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/persist_conn.h#L68
ctypedef struct persist_conn_t:
    void *auth_cred
    uid_t auth_uid
    gid_t auth_gid
    bool auth_ids_set
    _persist_conn_t_callback_proc callback_proc
    _persist_conn_t_callback_fini callback_fini
    char *cluster_name
    time_t comm_fail_time
    uint16_t my_port
    uint16_t flags
    bool inited
    persist_conn_type_t persist_type
    uid_t r_uid
    char *rem_host
    uint16_t rem_port
    time_t *shutdown
    pthread_t thread_id
    int timeout
    void *tls_conn
    slurm_trigger_callbacks_t trigger_callbacks
    uint16_t version

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/pack.h#L68
ctypedef struct buf_t:
    uint32_t magic
    char *head
    uint32_t size
    uint32_t processed
    bool mmaped
    bool shadow

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/slurm_protocol_defs.h#L761
ctypedef struct return_code_msg_t:
    uint32_t return_code

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/slurm_protocol_defs.h#L432
ctypedef struct job_id_msg_t:
    slurm_step_id_t step_id
    uint16_t show_flags

# https://github.com/SchedMD/slurm/blob/slurm-24-05-3-1/src/common/msg_type.h#L45
# Only partially defined - not everything needed at the moment.
ctypedef enum slurm_msg_type_t:
    REQUEST_SHARE_INFO    = 2022
    REQUEST_BATCH_SCRIPT  = 2051
    RESPONSE_BATCH_SCRIPT = 2052
    RESPONSE_SLURM_RC     = 8001

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/slurm_protocol_defs.h#L240
ctypedef struct forward_t:
    slurm_node_alias_addrs_t alias_addrs
    uint16_t cnt
    uint16_t init
    char *nodelist
    uint32_t timeout
    uint16_t tree_width
    uint16_t tree_depth

# https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/slurm_protocol_defs.h#L269
ctypedef struct forward_struct_t:
    slurm_node_alias_addrs_t *alias_addrs
    char *buf
    int buf_len
    uint16_t fwd_cnt
    pthread_mutex_t forward_mutex
    pthread_cond_t notify
    list_t *ret_list
    uint32_t timeout


cdef extern from *:
    """
    typedef struct conmgr_fd_s conmgr_fd_t; \
    """
    ctypedef struct conmgr_fd_t


# https://github.com/SchedMD/slurm/blob/slurm-25-11-4-1/src/common/slurm_protocol_defs.h
ctypedef struct slurm_msg_t:
    slurm_addr_t address
    void *auth_cred
    int auth_index
    uid_t auth_uid
    gid_t auth_gid
    bool auth_ids_set
    uid_t restrict_uid
    bool restrict_uid_set
    uint32_t body_offset
    buf_t *buffer
    void *pcon
    void *conmgr_con
    void *data
    uint16_t flags
    uint8_t hash_index
    char *tls_cert
    void *conn
    bool conn_is_mtls
    uint16_t msg_type
    uint16_t protocol_version
    forward_t forward
    forward_struct_t *forward_struct
    slurm_addr_t orig_addr
    list_t *ret_list

# https://github.com/SchedMD/slurm/blob/fe82218def7b57f5ecda9222e80662ebbb6415f8/src/common/slurm_protocol_defs.c#L865
cdef extern void slurm_free_return_code_msg(return_code_msg_t *msg)

# https://github.com/SchedMD/slurm/blob/2354049372e503af3217f94d65753abc440fa178/src/common/slurm_protocol_api.h#L440
cdef extern int slurm_send_recv_controller_msg(slurm_msg_t *request_msg,
                                        slurm_msg_t *response_msg,
                                        slurmdb_cluster_rec_t *comm_cluster_rec)

# https://github.com/SchedMD/slurm/blob/fe82218def7b57f5ecda9222e80662ebbb6415f8/src/common/slurm_protocol_defs.c#L168
cdef extern void slurm_msg_t_init(slurm_msg_t *msg)

# https://github.com/SchedMD/slurm/blob/master/src/common/job_resources.h
ctypedef struct job_resources:
    bitstr_t *core_bitmap
    bitstr_t *core_bitmap_used
    uint32_t  cpu_array_cnt
    uint16_t *cpu_array_value
    uint32_t *cpu_array_reps
    uint16_t *cpus
    uint16_t *cpus_used
    uint16_t *cores_per_socket
    uint16_t  cr_type
    uint64_t *memory_allocated
    uint64_t *memory_used
    uint32_t  next_step_node_inx
    uint32_t  nhosts
    bitstr_t *node_bitmap
    uint32_t  node_req
    char	 *nodes
    uint32_t  ncpus
    uint32_t *sock_core_rep_count
    uint16_t *sockets_per_node
    uint16_t *tasks_per_node
    uint16_t  threads_per_core
    uint8_t   whole_node

#
# TRES
#
ctypedef enum tres_types_t:
    TRES_CPU = 1
    TRES_MEM
    TRES_ENERGY
    TRES_NODE
    TRES_BILLING
    TRES_FS_DISK
    TRES_VMEM
    TRES_PAGES
    TRES_STATIC_CNT

# Global Environment
cdef extern char **environ

# Local slurm config
cdef extern slurm_conf_t slurm_conf

#
# Slurm Memory routines
# We simply use the macros from xmalloc.h - more convenient
#

cdef extern from "pyslurm/slurm/xmalloc.h" nogil:
    void xfree(void *__p)
    void *xmalloc(size_t __sz)
    void *try_xmalloc(size_t __sz)

cdef extern void slurm_xfree_ptr(void *)

#
# Slurm xstring functions
#

cdef extern char *slurm_xstrdup(const char *str)

#
# Slurm time functions
#

cdef extern void slurm_secs2time_str(time_t time, char *string, int size)
cdef extern void slurm_mins2time_str(time_t time, char *string, int size)
cdef extern int slurm_time_str2mins(const char *string)
cdef extern int slurm_time_str2secs(const char *string)
cdef extern void slurm_make_time_str(time_t *time, char *string, int size)
cdef extern time_t slurm_parse_time(char *time_str, int past)

#
# Slurm Job functions
#

cdef extern void slurm_free_job_desc_msg(job_desc_msg_t *msg)
cdef extern void slurm_free_job_info(job_info_t *job)
cdef extern void slurm_free_job_info_members(job_info_t *job)
cdef extern void slurm_free_job_step_info_response_msg(job_step_info_response_msg_t *msg)
cdef extern void slurm_free_job_step_info_members(job_step_info_t *msg)
cdef extern char *slurm_job_state_string(uint16_t inx)
cdef extern char *slurm_job_state_reason_string(int inx)
cdef extern char *slurm_job_share_string(uint16_t shared)
cdef extern void slurm_free_update_step_msg(step_update_request_msg_t *msg)

#
# Slurm Node functions
#

cdef extern char *slurm_node_state_string_complete(uint32_t inx)
cdef extern void slurm_free_update_node_msg(update_node_msg_t *msg)
cdef extern void slurm_free_node_info_members(node_info_t *node)

#
# Slurm environment functions

cdef extern void slurm_env_array_merge(char ***dest_array, const char **src_array)
cdef extern char **slurm_env_array_create()
cdef extern int slurm_env_array_overwrite(char ***array_ptr, const char *name, const char *value)
cdef extern void slurm_env_array_free(char **env_array)

#
# Misc
#

cdef extern char *slurm_preempt_mode_string (uint16_t preempt_mode)
cdef extern uint16_t slurm_preempt_mode_num (const char *preempt_mode)
cdef extern char *slurm_node_state_string (uint32_t inx)
cdef extern char *slurm_step_layout_type_name (task_dist_states_t task_dist)
cdef extern char *slurm_reservation_flags_string (reserve_info_t *resv_ptr)
cdef extern void slurm_free_stats_response_msg (stats_info_response_msg_t *msg)
cdef extern int slurm_addto_char_list_with_case(list_t *char_list, char *names, bool lower_case_noralization)
cdef extern int slurm_addto_step_list(list_t *step_list, char *names)
cdef extern int slurmdb_report_set_start_end_time(time_t *start, time_t *end)
cdef extern uint16_t slurm_get_track_wckey()
cdef extern void slurm_sprint_cpu_bind_type(char *str, cpu_bind_type_t cpu_bind_type)
cdef extern void slurm_accounting_enforce_string(uint16_t enforce, char *str, int str_len)

# Slurm bit functions

cdef extern bitstr_t *slurm_bit_alloc(bitoff_t nbits)
cdef extern void slurm_bit_set(bitstr_t *b, bitoff_t bit)
cdef extern int slurm_bit_test(bitstr_t *b, bitoff_t bit)
cdef extern char *slurm_bit_fmt(char *str, int32_t len, bitstr_t *b)
cdef extern void slurm_bit_free(bitstr_t **b)


cdef extern from *:
    """
    #define bit_free(__b) slurm_bit_free((bitstr_t **)&(__b))
    #define FREE_NULL_BITMAP(_X)    \
    do {                            \
        if (_X)                     \
                bit_free(_X);       \
        _X = NULL;                  \
    } while(0)                      \
    """
    void bit_free(bitstr_t *_X)
    void FREE_NULL_BITMAP(bitstr_t *_X)

cdef extern char *slurm_hostlist_deranged_string_xmalloc(hostlist_t *hl)

#
# slurmdb functions
#

cdef extern void slurmdb_job_cond_def_start_end(slurmdb_job_cond_t *job_cond)
cdef extern uint64_t slurmdb_find_tres_count_in_string(char *tres_str_in, int id)
cdef extern slurmdb_job_rec_t *slurmdb_create_job_rec()
cdef extern void slurmdb_init_assoc_rec(slurmdb_assoc_rec_t *assoc, bool free_it)
cdef extern void slurmdb_init_tres_cond(slurmdb_tres_cond_t *tres, bool free_it)

#
# Slurm Partition functions
#

cdef extern void slurm_free_update_part_msg(update_part_msg_t *msg)
cdef extern void slurm_free_partition_info_members(partition_info_t *node)

#
# Slurmctld stuff
#

cdef extern char *debug_flags2str(uint64_t debug_flags)
cdef extern int debug_str2flags(const char* debug_flags, uint64_t *flags_out)
cdef extern char *parse_part_enforce_type_2str(uint16_t type)
cdef extern char *health_check_node_state_str(uint32_t node_state)
cdef extern char *priority_flags_string(uint16_t priority_flags)
cdef extern char* prolog_flags2str(uint16_t prolog_flags)
cdef extern uint16_t prolog_str2flags(char *prolog_flags)
cdef extern char *log_num2string(uint16_t inx)
cdef extern uint16_t log_string2num(const char *name)
cdef extern char *private_data_string(uint16_t private_data, char *str, int str_len)
cdef extern char *reconfig_flags2str(uint16_t reconfig_flags)
cdef extern uint16_t reconfig_str2flags(char *reconfig_flags)
cdef extern char *select_type_param_string(uint16_t select_type_param)
cdef extern char *job_defaults_str(list_t *in_list)
