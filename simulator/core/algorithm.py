def schedule_fifo(scheduler, delta):
    # @Note: This is hard-coded to use yarn for now. We can switch them later
    placement = placement_algorithms['yarn']
    while len(scheduler.job_queue) > 0:
        next_job = scheduler.job_queue.pop(0)
        node, success = placement(scheduler, next_job)

        if success:
            next_job.failed = False
            assert node.add_job(next_job), "Couldn't allocate the job"
        else:
            if next_job.failed:
                next_job.pending_time += delta
            next_job.failed = True
            scheduler.job_queue.insert(0, next_job)
            break



scheduling_algorithms = {
#    'sjf': shortest_first_sim_jobs,
}

