
# EXPOSE slurmdb.h ENUMS TO PYTHON SPACE

# enum slurmdb_admin_level_t

SLURMDB_ADMIN_NOTSET                    = slurm.SLURMDB_ADMIN_NOTSET
SLURMDB_ADMIN_ADMIN_NONE                = slurm.SLURMDB_ADMIN_NONE
SLURMDB_ADMIN_OPERATOR                  = slurm.SLURMDB_ADMIN_OPERATOR
SLURMDB_ADMIN_SUPER_USER                = slurm.SLURMDB_ADMIN_SUPER_USER

# end enum slurmdb_admin_level_t

# enum slurmdb_classification_type_t

SLURMDB_CLASS_NONE                      = slurm.SLURMDB_CLASS_NONE
SLURMDB_CLASS_CAPABILITY                = slurm.SLURMDB_CLASS_CAPABILITY
SLURMDB_CLASS_CAPACITY                  = slurm.SLURMDB_CLASS_CAPACITY
SLURMDB_CLASS_CAPAPACITY                = slurm.SLURMDB_CLASS_CAPAPACITY

# end enum slurmdb_classification_type_t

# enum slurmdb_event_type_t

SLURMDB_EVENT_ALL                       = slurm.SLURMDB_EVENT_ALL
SLURMDB_EVENT_CLUSTER                   = slurm.SLURMDB_EVENT_CLUSTER
SLURMDB_EVENT_NODE                      = slurm.SLURMDB_EVENT_NODE

# end enum slurmdb_event_type_t

# enum slurmdb_problem_type_t

SLURMDB_PROBLEM_NOT_SET                 = slurm.SLURMDB_PROBLEM_NOT_SET
SLURMDB_PROBLEM_ACCT_NO_ASSOC           = slurm.SLURMDB_PROBLEM_ACCT_NO_ASSOC
SLURMDB_PROBLEM_ACCT_NO_USERS           = slurm.SLURMDB_PROBLEM_ACCT_NO_USERS
SLURMDB_PROBLEM_USER_NO_ASSOC           = slurm.SLURMDB_PROBLEM_USER_NO_ASSOC
SLURMDB_PROBLEM_USER_NO_UID             = slurm.SLURMDB_PROBLEM_USER_NO_UID

# end enum slurmdb_problem_type_t

# enum slurmdb_report_sort_t

SLURMDB_REPORT_SORT_TIME                = slurm.SLURMDB_REPORT_SORT_TIME
SLURMDB_REPORT_SORT_NAME                = slurm.SLURMDB_REPORT_SORT_NAME

# end enum slurmdb_report_sort_t

# enum slurmdb_report_time_format_t

SLURMDB_REPORT_TIME_SECS                = slurm.SLURMDB_REPORT_TIME_SECS
SLURMDB_REPORT_TIME_MINS                = slurm.SLURMDB_REPORT_TIME_MINS
SLURMDB_REPORT_TIME_HOURS               = slurm.SLURMDB_REPORT_TIME_HOURS
SLURMDB_REPORT_TIME_PERCENT             = slurm.SLURMDB_REPORT_TIME_PERCENT
SLURMDB_REPORT_TIME_SECS_PER            = slurm.SLURMDB_REPORT_TIME_SECS_PER
SLURMDB_REPORT_TIME_MINS_PER            = slurm.SLURMDB_REPORT_TIME_MINS_PER
SLURMDB_REPORT_TIME_HOURS_PER           = slurm.SLURMDB_REPORT_TIME_HOURS_PER

# end enum slurmdb_report_time_format_t

# enum slurmdb_resource_type_t

SLURMDB_RESOURCE_NOTSET                 = slurm.SLURMDB_RESOURCE_NOTSET
SLURMDB_RESOURCE_LICENSE                = slurm.SLURMDB_RESOURCE_LICENSE

# end enum slurmdb_resource_type_t

# enum slurmdb_update_type_t

SLURMDB_UPDATE_NOTSET                   = slurm.SLURMDB_UPDATE_NOTSET
SLURMDB_ADD_USER                        = slurm.SLURMDB_ADD_USER
SLURMDB_ADD_ASSOC                       = slurm.SLURMDB_ADD_ASSOC
SLURMDB_ADD_COORD                       = slurm.SLURMDB_ADD_COORD
SLURMDB_MODIFY_USER                     = slurm.SLURMDB_MODIFY_USER
SLURMDB_MODIFY_ASSOC                    = slurm.SLURMDB_MODIFY_ASSOC
SLURMDB_REMOVE_USER                     = slurm.SLURMDB_REMOVE_USER
SLURMDB_REMOVE_ASSOC                    = slurm.SLURMDB_REMOVE_ASSOC
SLURMDB_REMOVE_COORD                    = slurm.SLURMDB_REMOVE_COORD
SLURMDB_ADD_QOS                         = slurm.SLURMDB_ADD_QOS
SLURMDB_REMOVE_QOS                      = slurm.SLURMDB_REMOVE_QOS
SLURMDB_MODIFY_QOS                      = slurm.SLURMDB_MODIFY_QOS
SLURMDB_ADD_WCKEY                       = slurm.SLURMDB_ADD_WCKEY
SLURMDB_REMOVE_WCKEY                    = slurm.SLURMDB_REMOVE_WCKEY
SLURMDB_MODIFY_WCKEY                    = slurm.SLURMDB_MODIFY_WCKEY
SLURMDB_ADD_CLUSTER                     = slurm.SLURMDB_ADD_CLUSTER
SLURMDB_REMOVE_CLUSTER                  = slurm.SLURMDB_REMOVE_CLUSTER
SLURMDB_REMOVE_ASSOC_USAGE              = slurm.SLURMDB_REMOVE_ASSOC_USAGE
SLURMDB_ADD_RES                         = slurm.SLURMDB_ADD_RES
SLURMDB_REMOVE_RES                      = slurm.SLURMDB_REMOVE_RES
SLURMDB_MODIFY_RES                      = slurm.SLURMDB_MODIFY_RES
SLURMDB_ADD_TRES                        = slurm.SLURMDB_ADD_TRES
SLURMDB_UPDATE_FEDS                     = slurm.SLURMDB_UPDATE_FEDS

# end enum slurmdb_update_type_t

# enum cluster_fed_states

CLUSTER_FED_STATE_NA                    = slurm.CLUSTER_FED_STATE_NA
CLUSTER_FED_STATE_ACTIVE                = slurm.CLUSTER_FED_STATE_ACTIVE
CLUSTER_FED_STATE_INACTIVE              = slurm.CLUSTER_FED_STATE_INACTIVE

# end enum cluster_fed_states














