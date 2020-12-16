UTILCOST = 1.6
MEMCOST = 1.3

def gandiva_score(node, task):
    node_device_score = {}
    node_sum_score = 0
    min_cost = 999
    for _, d in node.device_cache.items():
        if not d.can_fit(task):
            continue

        # scoring.
        # worsefit_score
        mem_cost = (d.get_current_memory()+task.gpu_memory_max/d.memory)
        util_cost = d.get_current_utilization()
        
        cost = (mem_cost * MEMCOST) + (util_cost / 100) + len(d.running_tasks)
        node_device_score[d.device_id] = cost
        node_sum_score += cost
        # TODO: score cpu and memory too if needed
        if cost < min_cost:
            min_cost = cost
    return node_device_score, node_sum_score, min_cost

def horus_score(node, task):
    node_device_score = {}
    node_sum_score = 0
    min_cost = 999
    for _, d in node.device_cache.items():
        if not d.can_fit(task):
            continue
            
        # scoring.
        # worsefit_score
        mem_cost = (d.get_current_memory()+task.gpu_memory_max/d.memory)
        util_cost = (d.get_current_utilization() + task.gpu_utilization_max)
        util_cost = util_cost - 100.0
        if util_cost > 0:
            util_cost = abs(util_cost) * UTILCOST
        else:
            util_cost = abs(util_cost)
        
        cost = (mem_cost * MEMCOST) + (util_cost / 100) + len(d.running_tasks)
        node_device_score[d.device_id] = cost
        node_sum_score += cost
        # TODO: score cpu and memory too if needed
        if cost < min_cost:
            min_cost = cost
    return node_device_score, node_sum_score, min_cost

