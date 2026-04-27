/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_SERVICE_COLL_H_
#define UCC_SERVICE_COLL_H_

#include "ucc/api/ucc.h"
#include "components/tl/ucc_tl.h"

typedef struct ucc_service_coll_req {
    ucc_coll_task_t *task;
    ucc_team_t      *team;
    void *           data;
    ucc_subset_t     subset;
} ucc_service_coll_req_t;

ucc_status_t ucc_service_allreduce(ucc_team_t *team, void *sbuf, void *rbuf,
                                   ucc_datatype_t dt, size_t count,
                                   ucc_reduction_op_t op, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req);

ucc_status_t ucc_service_allgather(ucc_team_t *team, void *sbuf, void *rbuf,
                                   size_t msgsize, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req);

ucc_status_t ucc_service_bcast(ucc_team_t *team, void *buf, size_t msgsize,
                               ucc_rank_t root, ucc_subset_t subset,
                               ucc_service_coll_req_t **req);

ucc_status_t ucc_service_coll_test(ucc_service_coll_req_t *req);

ucc_status_t ucc_service_coll_finalize(ucc_service_coll_req_t *req);

ucc_status_t ucc_internal_oob_init(ucc_team_t *team, ucc_subset_t subset,
                                   ucc_team_oob_coll_t *oob);

void ucc_internal_oob_finalize(ucc_team_oob_coll_t *oob);

ucc_status_t ucc_collective_finalize_internal(ucc_coll_task_t *task);

/**
 * Create datatype validation schedule for rooted collectives
 *
 * This function checks if datatype validation is needed and creates a schedule
 * with validation logic if required. If validation is not needed, returns the
 * original task unchanged.
 *
 * @param team The UCC team
 * @param task The actual collective task (already created by TL/CL)
 * @param status_out If non-NULL, set to the error status when returning NULL
 * @return Schedule with validation (as ucc_coll_task_t*), or original task, or NULL on error
 */
ucc_coll_task_t* ucc_service_dt_check(ucc_team_t *team, ucc_coll_task_t *task,
                                      ucc_status_t *status_out);

/**
 * Create a dummy "actual" task for use with ucc_service_dt_check when the
 * regular ucc_coll_init path failed with UCC_ERR_NOT_SUPPORTED (typically
 * because some rank locally has a non-predefined datatype that the TL
 * cannot run).
 *
 * The dummy carries enough metadata for ucc_service_dt_check to construct
 * the validation schedule:
 *   - bargs (coll_type, root, flags, src/dst info, asymmetric scratch info)
 *     copied from the supplied bargs, used by ucc_service_dt_check to pick
 *     the right datatype/memory type to validate.
 *   - base team (task->team) - taken as the first CL team of the user team,
 *     used by ucc_schedule_init and the wrapper ucc_coll_task_init calls
 *     inside ucc_service_dt_check (only needs context access, not a coll
 *     plan, since the dummy never executes).
 *
 * The dummy's post() returns UCC_ERR_NOT_SUPPORTED defensively. In practice
 * it is never invoked because the cross-rank validation always fails when
 * this dummy is in play (the rank with non-predefined dt seeds
 * UCC_ERR_NOT_SUPPORTED into the min/max reduce), so
 * ucc_dt_check_actual_wrapper_post short-circuits before calling post().
 * Its finalize (ucc_dummy_finalize) returns the task to the stub mempool.
 *
 * @param team   The UCC team handle (must be UCC_TEAM_ACTIVE so cl_teams[0]
 *               exists)
 * @param bargs  Base coll args with full populated args, scratch info
 * @return Dummy task on success, NULL on allocation/init failure
 */
ucc_coll_task_t* ucc_dt_check_create_dummy_task(ucc_team_t *team,
                                                ucc_base_coll_args_t *bargs);

#endif
