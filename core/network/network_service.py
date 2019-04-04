def calculate_network_costs(infrastructure, nodes_assigned, job):
    """
        NOTE:
            calculate the slow down given nodes are assigned,
            very basic network cost model,
            2 is round trip.
            basic = (datasize/bandwidth * job_iteration * 2) + crossracks_deviation
    """
    # okay let's check where the PS is, if there is a PS.
    # first
    """
    job
    for n in nodes_assigned:
        jb = n.placed_jobs[job_id]
        (jb.model_size / infrastructure.bandwidth * jb.iteration * 2)
    print(nodes_assigned)
    print(job_id)
    """
    return 0
