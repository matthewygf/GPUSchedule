import math
from core import util


# TODO: GANDIVA, TOPOLOGY
def ms_yarn_placement(infrastructure, next_job):
    gpu_demand = next_job.gpus
    success = False
    try_alloc_ms = try_cross_node_alloc_ms if gpu_demand > infrastructure.num_gpu_p_node else try_single_node_alloc_ms
    nodes, success = try_alloc_ms(infrastructure, next_job)
    return nodes, success


placement_algorithms = {
    'yarn': ms_yarn_placement
}


def schedule_fifo(placement_algo, infrastructure, jobs_manager, delta):
    """NOTE: First in first out, does not preempt or migrate"""
    # F in F out, get the first job from the queue
    next_job = jobs_manager.pop(delta)
    if next_job is None:
        return
    assert next_job.is_waiting()
    nodes, success = placement_algo(infrastructure, next_job)
    if success:
        return nodes, next_job, success
    else:
        next_job.pending_time += delta
        jobs_manager.insert(next_job)
        return nodes, next_job, success


scheduling_algorithms = {
    'fifo': schedule_fifo,
    # 'sf': schedule_smallest_first
}


def try_cross_node_alloc_ms(infrastructure, job):
    """
    From Tiresias:
    try get gpus from multiple nodes
        [ need gpus / gpu_p_node ] nodes, and one node with [need_gpu % gpu_p_node]
    if can't find, give up, and return False
    """
    # if someone decide to have 5 gpus but we have 4 per node,
    # we assigned 2 full node.
    least_num_full_nodes = math.ceil(job.gpus / infrastructure.num_gpu_p_node)

    nodes_assigned = {}
    to_be_assigned = job.tasks.copy()
    num_full_tasks = len(job.tasks)
    assigned_task = {}
    for n_id, node in iter(infrastructure.nodes.items()):
        if not node.is_free(): continue

        if len(assigned_task) == len(to_be_assigned): break

        # this is checking how many nodes can fit the job current remaining tasks.
        ps_tasks_can_fit, worker_tasks_can_fit = node.can_fit_num_task(to_be_assigned)

        ps_count = 0
        worker_count = 0
        pop_t = None
        for k, v in iter(job.tasks.items()):
            if k in assigned_task:
                continue

            if 'ps' in k and ps_count <= ps_tasks_can_fit:
                pop_t = to_be_assigned[k]
                ps_count += 1
            elif 'worker' in k and worker_count <= worker_tasks_can_fit:
                pop_t = to_be_assigned[k]
                worker_count += 1
            else:
                continue

            if pop_t is not None:
                result = node.try_reserve_and_placed_task(pop_t)
                if not result:
                    # we didn't actually placed anything if it was false.
                    continue
                # from a job perspective keep track of where my tasks are
                job.tasks_running_on[k] = node.node_id
                assigned_task[k] = v

        # at least we have some task in the node.
        if ps_count > 0 or worker_count > 0:
            node.try_reserve_and_placed_job(job, False)
            nodes_assigned[node.node_id] = node

        if len(nodes_assigned) >= least_num_full_nodes and num_full_tasks == assigned_task:
            util.print_fn("assigned number of nodes %d" %
                          (len(nodes_assigned)))
            break

    # if not enough, clear everything.
    # NOTE: all tasks need to be assigned!!!
    if len(assigned_task) < num_full_tasks or len(nodes_assigned) < least_num_full_nodes :
        for node in iter(nodes_assigned.values()):
            node.placed_jobs.pop(job.job_id)
            for t in iter(job.tasks.values()):
                pop_t = node.placed_tasks.pop(t.task_id, None)
                if pop_t is not None:
                    node.release_allocated_resources(pop_t)
        nodes_assigned.clear()
        return {}, False

    if len(nodes_assigned) >= least_num_full_nodes and len(assigned_task) == num_full_tasks:
        util.print_fn("placed job %s, on node %s" % (job.job_id, str(nodes_assigned.keys())))
        return nodes_assigned, True

    raise ArithmeticError()


def try_single_node_alloc_ms(infrastructure, job):
    """
    From Tiresias:
    try get gpus from a single node,
    if can't find a node, give up, and return False
    NOTE: single node, assume no network transfer costs
    omitting data transfer i.e. DFS data transfer.
    """
    allocated = False
    assigned_node = {}
    for node in infrastructure.get_free_nodes():
        if allocated:
            return assigned_node, allocated

        if ((node.gpu_free() >= job.gpus) and
                (node.cpu_free() >= job.total_cpus_required()) and
                (node.mem_free() >= job.total_mem_required())):
            if not node.try_alloc_job(job, True): continue
            # succeed !
            allocated = True
            assigned_node[node.node_id] = node
    return assigned_node, allocated
