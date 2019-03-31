import math

def ms_yarn_placement(scheduler, next_job):
    gpu_demand = next_job.gpus
    success = False
    network_traffics = 0
    try_alloc = try_cross_node_alloc if gpu_demand > scheduler.infrastructure.num_gpu_p_node else try_single_node_alloc
    node_ids, network_costs, success = try_alloc(scheduler.infrastructure, next_job)
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
        return False
    # F in F out, get the first job from the queue
    next_job = scheduler.jq_manager.pop()
    node_id, network_costs, success = placement(scheduler, next_job)
    if success:
        # AT THIS POINT: THIS ACTUALLY RUN THE JOB.
        assert scheduler.add_to_running(node_id, next_job.job_id), "Couldn't allocate the job"
    else:
        next_job.pending_time += delta
        scheduler.jq_manager.insert(next_job, job_in_queue=0)

scheduling_algorithms = {
    'fifo': schedule_fifo
}

def try_cross_node_alloc(infrastructure, job):
    """
    From Tiresias:
    try get gpus from multiple nodes
        [ need gpus / gpu_p_node ] nodes, and one node with [need_gpu % gpu_p_node]
    if can't find, give up, and return False
    """
    # if someone decide to have 5 gpus but we have 4 per node,
    # we assigned 2 full node.
    num_full_nodes = math.ceil(job.gpu / infrastructure.num_gpu_p_node)
    extra_node_gpu = job.gpu % infrastructure.num_gpu_p_node

    nodes_assigned = []
    to_be_assigned = job.tasks
    for node in infrastructure.get_free_nodes():
        assigned = False

        if to_be_assigned == 0: break
        
        # this is checking how many nodes can fit the job current remaining tasks.
        ps_tasks_can_fit, worker_tasks_can_fit = node.can_fit_num_task(to_be_assigned)
        # NOTE: keep track of which task has been assigned.
        tasks_index_assigned = []

        if ps_tasks_can_fit > 0:
            for i, t in enumerate(to_be_assigned):
                if t.task_id.startswith('ps'):
                    node.tasks.append(t)
                    tasks_index_assigned.append(i)
                    assigned = True

        if worker_tasks_can_fit > 0:
            for i, t in enumerate(to_be_assigned):
                if t.task_id.startswith('worker'):
                    node.tasks.append(t)
                    tasks_index_assigned.append(i)
        
        # remove the ones already got assigned.
        for i in tasks_index_assigned:
            to_be_assigned.pop(i)

        if assigned:
            node.jobs.append(job)
            nodes_assigned.append(node)

        if len(nodes_assigned) == num_full_nodes:
            break
    
    # if not enough, clear everything.
    if len(nodes_assigned) < num_full_nodes:
        nodes_assigned.clear()
        node.jobs.clear()
        node.tasks.clear()
        return [], 0, False 
    
    # TODO: NETWORK COSTS
    if len(nodes_assigned) == num_full_nodes:
        return [n.node_id for n in nodes_assigned], 0, True

def try_single_node_alloc(infrastructure, job):
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
            return node_id, network_costs, allocated

        if ((node.gpu_free() >= job.gpus) and 
            (node.cpu_free() >= job.total_cpus_required()) and 
            (node.mem_free() >= job.total_mem_required())):
            if not node.try_alloc_job(job): continue
            # succeed !
            allocated = True
            node_id = node.node_id
    return node_id, network_costs, allocated


    
