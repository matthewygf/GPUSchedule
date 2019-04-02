import math
import copy
from core import util

def ms_yarn_placement(scheduler, next_job):
    gpu_demand = next_job.gpus
    success = False
    network_traffics = 0
    try_alloc_ms = try_cross_node_alloc_ms if gpu_demand > scheduler.infrastructure.num_gpu_p_node else try_single_node_alloc_ms
    node_ids, network_costs, success = try_alloc_ms(scheduler.infrastructure, next_job)
    network_traffics += network_costs
    return node_ids, network_costs, success

placement_algorithms = {
    'yarn': ms_yarn_placement 
}

def network_costs_update(network_costs, job):
    """NOTE: cross rack network_costs"""
    pass

def schedule_fifo(scheduler, delta):
    """NOTE: First in first out, does not preempt or migrate"""
    placement = placement_algorithms[scheduler.placement]
    # check if there is any node available
    num_free = scheduler.num_free_nodes()
    if num_free <= 0:
        #util.print_fn("Everything is full")
        return
    # F in F out, get the first job from the queue
    next_job = scheduler.jq_manager.pop()
    node_ids, network_costs, success = placement(scheduler, next_job)
    if success:
        # AT THIS POINT: THIS ACTUALLY RUN THE JOB.
        assert scheduler.add_to_running(node_ids, next_job.job_id), "Couldn't allocate the job"
    else:
        next_job.pending_time += delta
        scheduler.jq_manager.insert(next_job, job_in_queue=0)

scheduling_algorithms = {
    'fifo': schedule_fifo,
    #'sf': schedule_smallest_first
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
    num_full_nodes = math.ceil(job.gpus / infrastructure.num_gpu_p_node)
    extra_node_gpu = job.gpus % infrastructure.num_gpu_p_node

    nodes_assigned = {}
    to_be_assigned = copy.deepcopy(job.tasks)
    print(to_be_assigned.keys())
    num_full_tasks = len(job.tasks)
    assigned_task = 0

    for node in infrastructure.get_free_nodes():
        util.print_fn("to_be_assigned %d" % len(to_be_assigned))
        copy_to_be_assigned = to_be_assigned.copy()
        if len(copy_to_be_assigned) == 0: break
        
        # this is checking how many nodes can fit the job current remaining tasks.
        ps_tasks_can_fit, worker_tasks_can_fit = node.can_fit_num_task(to_be_assigned)

        # can fit at least some amount of ps and worker tasks
        ps_count = 0
        worker_count = 0
        for t in iter(copy_to_be_assigned.values()):
            if 'ps' in t.task_id and ps_count <= ps_tasks_can_fit:
                pop_t = to_be_assigned.pop(t.task_id, None)
                ps_count += 1
            elif 'worker' in t.task_id and worker_count <= worker_tasks_can_fit:
                pop_t = to_be_assigned.pop(t.task_id, None)
                worker_count += 1
            else:
                continue

            if pop_t is not None:
                result = node.try_reserve_and_placed_task(pop_t)
                if not result:
                    # re-insert
                    to_be_assigned[pop_t.task_id] = pop_t
                    continue
                # from a job perspective keep track of where my tasks are
                job.tasks_running_on[pop_t.task_id] = node.node_id
                assigned_task += 1
                util.print_fn("found placement for task %s in node %s" % (t.task_id, node.node_id))
        
        # at least we have some task in the node.
        if ps_count > 0 or worker_count > 0:        
            node.try_reserve_and_placed_job(job, False)
            nodes_assigned[node.node_id] = node

        if len(nodes_assigned) >= num_full_nodes and num_full_tasks == assigned_task:
            util.print_fn("assigned number of nodes %d" % 
                        (len(nodes_assigned)))
            break
    
    # if not enough, clear everything.
    if len(nodes_assigned) < num_full_nodes or assigned_task < num_full_tasks:
        util.print_fn("Needed %d: have nodes %d Not enough nodes to satisfied job %s, num of tasks remain to be assigned %d" % 
                        (num_full_nodes, len(nodes_assigned), job.job_id, len(to_be_assigned)))
        for node in iter(nodes_assigned.values()):
            node.placed_jobs.pop(job.job_id)
            for t in iter(job.tasks.values()):
                node.placed_tasks.pop(t.task_id, None)
        nodes_assigned.clear()
        return [], 0, False 
    
    # TODO: NETWORK COSTS
    if len(nodes_assigned) >= num_full_nodes and assigned_task == num_full_tasks:
        return list(nodes_assigned.keys()), 0, True
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
    node_id = None
    network_costs = 0 
    for node in infrastructure.get_free_nodes():
        if allocated:
            return [node_id], network_costs, allocated

        if ((node.gpu_free() >= job.gpus) and 
            (node.cpu_free() >= job.total_cpus_required()) and 
            (node.mem_free() >= job.total_mem_required())):
            if not node.try_alloc_job(job): continue
            # succeed !
            allocated = True
            node_id = node.node_id
    return [node_id], network_costs, allocated


    
