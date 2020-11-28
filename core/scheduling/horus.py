UTILCOST = 1.6
MEMCOST = 1.3

def horus_score(node, task):
    node_device_score = {}
    node_sum_score = 0
    for d in node.device_cache:
        if not d.can_fit(task):
            continue
            
        # scoring.
        # worsefit_score
        mem_cost = (d.current_memory()+task.gpu_memory_max/d.memory)
        util_cost = (d.current_util() + task.gpu_utilization_max)
        util_cost = util_cost - 100.0
        if util_cost > 0:
            util_cost = abs(util_cost) * UTILCOST
        else:
            util_cost = abs(util_cost)
        
        cost = (mem_cost * MEMCOST) + (util_cost / 100) + len(d.running_tasks)
        node_device_score[d.device_id] = cost
        node_sum_score += cost
    return node_device_score, node_sum_score

