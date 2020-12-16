import logging
import math
from core import util
import numpy as np
from collections import OrderedDict
from core.scheduling.horus import horus_score, gandiva_score
from heapq import heappop, heappush

score_fn = {
    'horus': horus_score,
    'horus+': horus_score,
    'gandiva': gandiva_score
}

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

def ms_yarn_placement(infrastructure, next_job, scheme):
    gpu_demand = next_job.gpus
    try_alloc_ms = try_cross_node_alloc_ms if gpu_demand > infrastructure.num_gpu_p_node else try_single_node_alloc_ms
    nodes, success = try_alloc_ms(infrastructure, next_job)
    return nodes, success

def horus_placement(infrastructure, next_job, scheme):
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
            can_fit = node.can_fit(t, pack=True)
            if not can_fit:
                continue
            
            # score the node as the sum of the gpu scores ?
            device_scores, node_sum, min_cost = score_fn[scheme](node, t)
            node_device_info = NodeDeviceInfo(node, device_scores=device_scores, total_score=node_sum, min_score=min_cost)
            heappush(nodes_stack, node_device_info)

            # assuming each node has 1 device avaliable, then at most,
            # we only need to maintain a heap of size =  each node with 1 gpu
            if len(nodes_stack) > gpu_demand:
                item = heappop(nodes_stack)
                score_cache[node.node_id] = item
    
    # now sort the stack into minimum first
    nodes_stack = sorted(nodes_stack, key=lambda x: x.min_score)
    best_assignment_for_node = []
    for _ in range(len(nodes_stack)):
        # to store each node with its associate distributed training node
        best_assignment_for_node.append((set(), {}, False))

    num_tasks = len(next_job.tasks)
    at_least_one_succeeded = False
    for i, info in enumerate(nodes_stack):

        for t_id, t in next_job.tasks.items():
            ok = infrastructure.nodes[info.node.node_id].try_reserve_and_placed_task(t, pack=True)
            if ok:
                assert infrastructure.nodes[info.node.node_id].try_reserve_and_placed_job(next_job, False) == True
                current_set_of_nodes = best_assignment_for_node[i][0]
                current_set_of_nodes.add(info.node.node_id)

                current_set_of_task_nodes_mapping = best_assignment_for_node[i][1]
                if t_id in current_set_of_task_nodes_mapping:
                    raise ValueError()
                current_set_of_task_nodes_mapping[t_id] = info.node.node_id
                best_assignment_for_node[i] = (current_set_of_nodes, current_set_of_task_nodes_mapping, best_assignment_for_node[i][2])
        
        # else if one node cannot take it.
        # choose from closer racks
        racks_to_consider = infrastructure.get_racks_by_dist(info.node.rack_id)
        for r_id, _ in racks_to_consider:
            if len(best_assignment_for_node[i][1]) >= num_tasks:
                at_least_one_succeeded = True
                break
            
            the_rack = infrastructure.racks[r_id]
            
            for _, rack_n in the_rack.nodes.items():
                if len(best_assignment_for_node[i][1]) >= num_tasks:
                    break

                for t_id, t in next_job.tasks.items():
                    if t_id in best_assignment_for_node[i][1]:
                        continue
                    ok = infrastructure.nodes[rack_n.node_id].try_reserve_and_placed_task(t, pack=True)
                    if ok:
                        assert infrastructure.nodes[rack_n.node_id].try_reserve_and_placed_job(next_job, False) == True

                        current_set_of_nodes = best_assignment_for_node[i][0]
                        current_set_of_nodes.add(rack_n.node_id)

                        current_set_of_task_nodes_mapping = best_assignment_for_node[i][1]
                        if t_id in current_set_of_task_nodes_mapping:
                            raise ValueError()
                        current_set_of_task_nodes_mapping[t_id] = rack_n.node_id
                        best_assignment_for_node[i] = (current_set_of_nodes, current_set_of_task_nodes_mapping, best_assignment_for_node[i][2])

                    if len(best_assignment_for_node[i][1]) >= num_tasks:
                        break
    
        # clear previously reserved for backtracking
        poped_job = {}
        for task_id , assigned_node in best_assignment_for_node[i][1].items():
            if not next_job.job_id in poped_job:
                infrastructure.nodes[assigned_node].placed_jobs.pop(next_job.job_id)
                poped_job[next_job.job_id] = 1
            else:
                poped_job[next_job.job_id] += 1
            
            pop_t = infrastructure.nodes[assigned_node].placed_tasks.pop(task_id, None)
            if pop_t is not None:
                infrastructure.nodes[assigned_node].release_allocated_resources(pop_t, reserved=True)
        
        # have enough for current node
        if len(best_assignment_for_node[i][1]) >= num_tasks:
            # if task,node map satisfied the number of task, we should be okay.
            # start again for the next node,
            # as we are iterating to see which combinations of nodes are the best
            at_least_one_succeeded = True
            best_assignment_for_node[i] = (best_assignment_for_node[i][0], best_assignment_for_node[i][1], True)
            continue

    # no possible placements for the tasks.
    if not at_least_one_succeeded:
        return None, False
    
    # now we should have assignments ready for each node from the heap
    # and its associated jobs assignment across nodes and racks
    # sort by smallest amount of nodes needed 
    # TODO: whether to by nodes or by dist of racks
    # currently, the less nodes the better
    # only succeeded nodes
    best_assignment_for_node = [ x for x in best_assignment_for_node if x[2] ]
    logging.info("succeeded placements plans: %s", best_assignment_for_node)
    if len(best_assignment_for_node) == 0:
        return None, False

    best_assignment_for_node = sorted(best_assignment_for_node, key=lambda x: len(x[0]))
    task_node_map = best_assignment_for_node[0][1]
    cnt = 0
    results = {}
    for task_id, node_id in task_node_map.items():
        t = next_job.tasks[task_id]
        success = infrastructure.nodes[node_id].try_reserve_and_placed_task(t, pack=True)
        if success:
            logging.info("placed task %s at node %s", task_id, node_id)
            results[node_id] = infrastructure.nodes[node_id]
            # from a job perspective keep track of where my tasks are
            next_job.tasks_running_on[task_id] = infrastructure.nodes[node_id].node_id
            infrastructure.nodes[node_id].try_reserve_and_placed_job(next_job, False)
            cnt += 1

    assert cnt == len(next_job.tasks), ('has %d expected %d - assigned map %s ' % 
        (cnt, len(next_job.tasks), task_node_map))
    
    return results, True

placement_algorithms = {
    'yarn': ms_yarn_placement,
    'horus': horus_placement,
    'horus+': horus_placement,
    'gandiva': horus_placement
}

def schedule_fifo(scheme, placement_algo, infrastructure, jobs_manager, delta, **kwargs):
    """NOTE: First in first out, does not preempt or migrate"""
    # F in F out, get the first job from the queue
    next_job = jobs_manager.get_next_job(delta)
    if next_job is None:
        util.print_fn("no job ready at time %d" % (delta))
        return None, None, None
    assert next_job.is_waiting()
    nodes, success = placement_algo(infrastructure, next_job, scheme)
    if success:
        _ = jobs_manager.pop(delta)
        return nodes, next_job, success
    
    return nodes, next_job, success

def schedule_horus(scheme, placement_algo, infrastructure, jobs_manager, delta, k=5, **kwargs):
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
    look_ahead_pos = None
    nodes_to_schedule = None
    for idx, j in enumerate(look_ahead):
        nodes, success = placement_algo(infrastructure, j, scheme)
        
        if success:
            look_ahead_pos = idx
            nodes_to_schedule = nodes
            break
        
    #. 3. schedule the first job that is successful
    #  as it's already sorted by utilization
    #  then put the rest of the jobs back into the corresponding queue.
    if look_ahead_pos is None or nodes_to_schedule is None:
        jobs_manager.insert(look_ahead)
        return None, None, False
        
    target_job = look_ahead.pop(look_ahead_pos)
    assert len(look_ahead) == current_len - 1
    jobs_manager.insert(look_ahead)
    return nodes_to_schedule, target_job, True

def schedule_horus_plus(scheme, placement_algo, infrastructure, jobs_manager, delta, k=5, **kwargs):
    '''
    poped from the queue with the most credit.
    - like a leaky bucket.
    '''
    #. 1. get min(k, queuing) jobs
    look_ahead = []
    look_ahead_q = []
    total_jobs = jobs_manager.total_jobs(delta)
    min_k = max(0, min(k, jobs_manager.queuing_jobs(delta)))
    if min_k == 0:
        return None, None, False
    logging.info("min K: %d", min_k)
    for _ in range(0, min_k):
        jobs_manager.job_queue_manager.update_credits()
        q_idx = jobs_manager.job_queue_manager.queue_idx_by_credit()
        j = jobs_manager.pop(delta, queue_idx = q_idx)
        assert j.is_waiting()
        look_ahead.append(j)
        look_ahead_q.append(q_idx)
    
    current_len = len(look_ahead)
    assert current_len == min_k

    #. 2. for each job, score each node
    look_ahead_pos = None
    nodes_to_schedule = None
    for idx, j in enumerate(look_ahead):
        nodes, success = placement_algo(infrastructure, j)
        
        if success:
            look_ahead_pos = idx
            nodes_to_schedule = nodes
            break
    
    #. 3. schedule the first job that is successful
    #  as it's already sorted by utilization
    #  then put the rest of the jobs back into the corresponding queue.
    if look_ahead_pos is None or nodes_to_schedule is None:
        assert len(look_ahead) == current_len
        jobs_manager.insert(look_ahead, queue_insert_position=look_ahead_q)
        assert total_jobs == jobs_manager.total_jobs(delta)
        return None, None, False
    target_job = look_ahead.pop(look_ahead_pos)
    look_ahead_q.pop(look_ahead_pos)
    assert len(look_ahead) == current_len - 1
    jobs_manager.insert(look_ahead, queue_insert_position=look_ahead_q)
    assert total_jobs-1 == jobs_manager.total_jobs(delta)
    return nodes_to_schedule, target_job, True

scheduling_algorithms = {
    'fifo': schedule_fifo,
    'horus': schedule_horus,
    'horus+': schedule_horus_plus,
    'gandiva': schedule_fifo
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
                # logging.info("Job %s - task %s placed on %s" % (job.job_id, k, node.node_id))
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
                    node.release_allocated_resources(pop_t, reserved=True)
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

        if ((node.get_free_devices(False) >= job.gpus) and
                (node.cpu_free() >= job.total_cpus_required()) and
                (node.mem_free() >= job.total_mem_required())):
            if not node.try_alloc_job(job, True): continue
            # succeed !
            allocated = True
            assigned_node[node.node_id] = node
    return assigned_node, allocated


def time_slice_check(infrastructure, jobs_manager):
    '''Only need to time slice if there is queuing jobs'''
    queue_len = jobs_manager.queuing_jobs()
    if queue_len == 0:
        return
    
    # NOTE: 
    # assume time quanta is 100 for now as we have scaled our jobs size.
    quanta = 100

    # scan through all currently running tasks.
    todo = []
    for k, v in jobs_manager.running_jobs.items():
        processed = v.time_processed()
        if processed > 1 and processed % quanta == 0:
            todo.append(k)
            timeslice_num = processed // quanta
            logging.info(f"****time slice up for job {k}, time sliced number at: {timeslice_num}")
    
    for j in todo:
        jobs_manager.preempt(j, infrastructure)

plugin_algorithms = {
    'gandiva': time_slice_check
}
