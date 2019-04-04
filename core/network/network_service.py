from core import util
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

    diff = ps_nodes.symmetric_difference(wk_nodes)
    cross_many = len(diff)
    if cross_many == 0:
        # cross node will induced latency, if all resides on the same node, 
        # assume there is nothing even if there is PS-workers
        return 0

    # assume PS has sharded parameters,
    # so the more difference we have,
    # the more communication we need to do.
    # per second **Some Heuristics**
    model_per_sec = (job.model_size / infrastructure.bandwidth)
    nodes_induced_sec = (cross_many*0.025)
    iteration_round_trip = job.iterations * 2.0
    extra_seconds = ( model_per_sec + nodes_induced_sec ) * iteration_round_trip
    util.print_fn("Cross %s need to added extra %f for job %s" % (str(diff), extra_seconds, job.job_id))
    return extra_seconds
