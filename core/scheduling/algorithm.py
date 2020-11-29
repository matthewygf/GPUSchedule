import logging
import math
from core import util
import numpy as np
from core.scheduling.horus import horus_score
from heapq import heappop, heappush


class NodeDeviceInfo(object):
    def __init__(self, node, device_scores, total_score, min_score):
        """
        a class to hold the relevant scoring information and allow the heap to compare
        """
        self.device_scores = device_scores
        self.node = node
        self.total_score = total_score
        self.min_score = min_score

    def __lt__(self, other):
        return self.min_score > other.min_score

def ms_yarn_placement(infrastructure, next_job):
    gpu_demand = next_job.gpus
    try_alloc_ms = try_cross_node_alloc_ms if gpu_demand > infrastructure.num_gpu_p_node else try_single_node_alloc_ms
    nodes, success = try_alloc_ms(infrastructure, next_job)
    return nodes, success

def horus_placement(infrastructure, next_job):
    ''' horus produce a scores for each node given a job.
        according to gpu utilizations and gpu memory capacity. 
        NOTE: 
        no bandwidth consideration at the moment.
        i.e. the following are not considered 
        1. prefer 1 node to hold all tasks.
        2. schedule the tasks as close as possible.
        3. PCIe bandwidth
    '''
    gpu_demand = next_job.gpus
    
    # maintain a max heap with min size of nodes to hold the tasks.
    nodes_stack = []
    score_cache = OrderedDict()
    for _, t in next_job.tasks.items():
        for node in infrastructure.get_free_nodes():
            # this is checking how many nodes can fit the job current remaining tasks.
            can_fit = node.can_fit(t)
            if not can_fit:
                continue
            
            # score the node as the sum of the gpu scores ?
            device_scores, node_sum, min_cost = horus_score(node, t)
            node_device_info = NodeDeviceInfo(node, device_scores=device_scores, total_score=node_sum, min_score=min_cost)
            heappush(nodes_stack, node_device_info)

            # assuming each node has 1 device avaliable, then at most,
            # we only need to maintain a heap of size =  each node with 1 gpu
            if len(nodes_stack) > gpu_demand:
                item = heappop(nodes_stack)
                score_cache[node.node_id] = item
                logging.info("removing: %s min_cost at: %d" % (str(item.node.node_id), item.min_score))
    
    # dp to find best distanced placement.
    # if possible.
    # if cross node add a cost, if cross rack add a bigger cost
    logging.info("ranking nodes : %s" % [n.node.node_id for n in nodes_stack])

    # now sort the stack into minimum first
    nodes_stack = sorted(nodes_stack, key=lambda x: x.min_score)
    best_assignment_for_node = []
    for _ in range(len(nodes_stack)):
        best_assignment_for_node.append([])


    logging.info(best_assignment_for_node)

    num_tasks = len(next_job.tasks)
    for i, info in enumerate(nodes_stack):
        
        # case 1: if one node can satisfy the job. do it
        worker_tasks_can_fit = info.node.can_fit_num_task(next_job.tasks)
        
        if worker_tasks_can_fit == len(next_job.tasks):
            # only one node needed.
            best_assignment_for_node[i].append(info.node)
            break
        
        # case 2: get some more node from the closest racks.
        racks_to_consider = infrastructure.get_racks_by_dist(info.node.rack_id)
        # starting from close rack if possible.
        logging.info("---------------------")
        for r_id, dist in racks_to_consider:
            # infrastructure.racks[r_id]
            # logging.info("rack %s - dist at %d" % (r_id, dist))

            while worker_tasks_can_fit < num_tasks:
                # TODO:
                worker_tasks_can_fit += n.can_fit_num_task(next_job)
                best_assignment_for_node[i].append(info.node)


    raise NotImplementedError()


placement_algorithms = {
    'yarn': ms_yarn_placement,
    'horus': horus_placement
}


def schedule_fifo(placement_algo, infrastructure, jobs_manager, delta, **kwargs):
    """NOTE: First in first out, does not preempt or migrate"""
    # F in F out, get the first job from the queue
    next_job = jobs_manager.get_next_job(delta)
    if next_job is None:
        util.print_fn("no job ready at time %d" % (delta))
        return None, None, None
    assert next_job.is_waiting()
    nodes, success = placement_algo(infrastructure, next_job)
    if success:
        _ = jobs_manager.pop(delta)
        return nodes, next_job, success
    next_job.pending_time += delta
    return nodes, next_job, success

def schedule_horus(placement_algo, infrastructure, jobs_manager, delta, k=5, **kwargs):
    """NOTE: schedule based on utilization and queue size."""
    #. 1. get min(k, queuing) jobs
    look_ahead = []
    min_k = max(0, min(k, jobs_manager.queuing_jobs(delta)))
    for _ in range(0, min_k):
        j = jobs_manager.pop(delta)
        assert j.is_waiting()
        look_ahead.append(j)
    
    current_len = len(look_ahead)
    assert current_len == min_k

    #. 2. for each job, score each node
    job_node_costs = {}
    current_min_cost = None
    look_ahead_pos = None
    for idx, j in enumerate(look_ahead):
        sorted_nodes_score = placement_algo(infrastructure, j)
        job_node_costs[j.job_id] = sorted_nodes_score
        if current_min_cost is None:
            current_min_cost = j
            look_ahead_pos = idx
        elif current_min_cost > sorted_nodes_score[0]:
            current_min_cost = sorted_nodes_score[0]
            look_ahead_pos = idx
        else:
            # skip
            continue
    
    #. 3. schedule the min_cost job
    #  a. put the rest of the job back into the corresponding queue.
    target_job = look_ahead.pop(look_ahead_pos)
    assert len(look_ahead) == current_len - 1
    jobs_manager.insert(look_ahead)
    return job_node_costs[target_job.job_id], target_job, True
    





scheduling_algorithms = {
    'fifo': schedule_fifo,
    'horus': schedule_horus,
    # 'sf': schedule_smallest_first
}


def try_cross_node_alloc_ms(infrastructure, job, sort_fn=None, filter_fn=None):
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
    all_nodes = infrastructure.nodes.values()
    
    
    if filter_fn:
        all_nodes = filter_fn(all_nodes)

    if sort_fn:
        all_nodes = sort_fn(all_nodes)
    
    
    for node in all_nodes:
        if not node.is_free(): continue

        if len(assigned_task) == num_full_tasks: break

        # this is checking how many nodes can fit the job current remaining tasks.
        worker_tasks_can_fit = node.can_fit_num_task(to_be_assigned)
        if worker_tasks_can_fit == 0:
            continue

        worker_count = 0
        pop_t = None
        check_next = False
        for k, v in iter(job.tasks.items()):
            if k in assigned_task:
                continue

            if 'worker' in k and worker_count <= worker_tasks_can_fit:
                pop_t = to_be_assigned.pop(k, None)
                worker_count += 1
            else:
                continue

            if pop_t is not None:
                result = node.try_reserve_and_placed_task(pop_t)
                if not result:
                    # we didn't actually placed anything if it was false.
                    # put it back.
                    to_be_assigned[k] = pop_t
                    worker_count -= 1
                    logging.info("unable to reserve job %s task %s on node %s, check next node..." % (job.job_id, k, node.node_id))
                    check_next = True
                    break
                # from a job perspective keep track of where my tasks are
                job.tasks_running_on[k] = node.node_id
                logging.info("Job %s - task %s placed on %s" % (job.job_id, k, node.node_id))
                assigned_task[k] = v

        # at least we have some task in the node.
        if  worker_count > 0:
            node.try_reserve_and_placed_job(job, False)
            nodes_assigned[node.node_id] = node
            logging.info("Job %s require %d - placed on nodes %s" % (job.job_id, 
                least_num_full_nodes, str(nodes_assigned.keys())))

        if check_next:
            continue

        if len(nodes_assigned) >= least_num_full_nodes and num_full_tasks == len(assigned_task):
            #util.print_fn("assigned number of nodes %d" %   (len(nodes_assigned)))
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
        logging.info("not enough ")
        return {}, False

    if len(nodes_assigned) >= least_num_full_nodes and len(assigned_task) == num_full_tasks:
        util.print_fn("placed job %s with task %d, on node - %s" % (job.job_id, job.worker_count, str(nodes_assigned.keys())))
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
