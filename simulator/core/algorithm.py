import math

def ms_yarn_placement(scheduler, next_job):
    gpu_demand = next_job.gpu
    success = False
    if gpu_demand > scheduler.infrastructure.num_gpu_p_node:
        node_id, success = try_cross_node_alloc(
            scheduler.infrastructure, next_job)
    else:
        node_id, success = try_single_node_alloc(
            scheduler.infrastructure, next_job
        )
    return node_id, success

placement_algorithms = {
    'yarn': ms_yarn_placement 
}

def schedule_fifo(scheduler, delta):
    """NOTE: First in first out, does not preempt or migrate"""
    placement = placement_algorithms[scheduler.placement]
    # check if there is any node available
    num_free = scheduler.num_free_nodes()
    if num_free <= 0:
        return False
    # F in F out, get the first job from the queue
    next_job = scheduler.jq_manager.pop()
    node_id, success = placement(scheduler, next_job)

    # # @Note: This is hard-coded to use yarn for now. We can switch them later
    #     node_id, success = placement(scheduler, next_job)
    #     # TODO: below need to re do
    #     if success:
    #         next_job.failed = False
    #         assert scheduler.add_to_running_job(node_id, next_job.job_id), "Couldn't allocate the job"
    #     else:
    #         if next_job.failed:
    #             next_job.pending_time += delta
    #         next_job.failed = True
    #         scheduler.job_queue.insert(0, next_job)
    #         break

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
    # job.gpu / infrastructure.num_gpu_p_node
    # TODO: cross_node_alloc
    pass


def try_single_node_alloc(infrastructure, job):
    """
    From Tiresias:
    try get gpus from a single node,
    if can't find a node, give up, and return False
    """
    all_nodes = infrastructure.nodes
    allocated = False
    node_id = None
    for node in all_nodes:
        if allocated:
            return node_id, allocated

        if ((node.gpu_free() >= job.gpu) and 
            (node.cpu_free() >= job.cpu) and 
            (node.mem_free() >= job.mem)):
            if not node.try_alloc_job(job): continue
            # succeed !
            allocated = True
            node_id = node.node_id
    return node_id, allocated


    
