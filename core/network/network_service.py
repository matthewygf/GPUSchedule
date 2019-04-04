def calculate_network_costs(infrastructure, job):
    """
        NOTE:
            calculate the slow down given nodes are assigned,
            very basic network cost model,
            2 is round trip.
            basic = (datasize/bandwidth * job_iteration * 2) + crossracks_deviation
    """
    # let's check where the PS is, if there is a PS.

    if not job.is_distributed():
        return 0

    ps_nodes = set()
    wk_nodes = set()
    for k, v in job.tasks_running_on.items():
        if 'ps' in k:
            ps_nodes.add(v)
        else:
            wk_nodes.add(v)

    diff = ps_nodes.difference(wk_nodes)
    cross_many = len(diff)
    # assume PS has sharded parameters,
    # so the more difference we have,
    # the more communication we need to do.
    # per second **Some Heuristics**
    extra_seconds = ((job.model_size / infrastructure.bandwidth) + (cross_many*0.25)) * job.iterations * 2
    return extra_seconds
