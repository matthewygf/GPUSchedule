import util
import lp
import flags
FLAGS = flags.FLAGS

def fit_first_sim_jobs(job_queue, cluster, logger):
    '''
    new jobs are added to the end of the ending queue
    but any fit job should be executed in fifo order
    '''
    while (len(job_queue.job_events) + len(job_queue.pending_jobs))> 0:
        if len(job_queue.job_events) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        event = job_queue.job_events[0]
        event_time = event['time']
        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            job_queue.remove_migratable(e_job)

            #job completes
            cluster.release_job_res(e_job)
            logger.job_complete(e_job, event_time)


        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            #add into pending list
            job_queue.move_to_pending(s_job)

        new_start_list = list()
        for p_job in job_queue.pending_jobs:
            # ret = CLUSTER.alloc_gpus(p_job)
            if cluster.check_free_gpu() <= 0:
                break
            ret = try_get_job_res(cluster, p_job)
            if ret == True:
                ''' if remove_from_pending, then will miss the next p_job in the list '''
                new_start_list.append(p_job)
                # JOBS.remove_from_pending(p_job, event_time)
                # JOBS.add_job_end_event(p_job)
                # util.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
            else:
                continue

        for ns_job in new_start_list:
            job_queue.remove_from_pending(ns_job, event_time)
            job_queue.add_job_end_event(ns_job)
            util.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])

        #sort pending jobs based on the num_gpu
        #JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        job_queue.job_events.pop(0)
        job_queue.job_events.sort(key = lambda e:e.__getitem__('time'))

        logger.checkpoint(event_time)


'''
Allocate job resource
'''
def try_get_job_res(cluster, job):
    '''
    select placement scheme
    '''
    if FLAGS.scheme == 'yarn':
        ret = cluster.ms_yarn_placement(job)
    elif FLAGS.scheme == 'balance':
        ret = lp.placement(job)
    elif FLAGS.scheme == 'random':
        ret = cluster.random_placement(job)
    elif FLAGS.scheme == 'crandom':
        ret = cluster.consolidate_random_placement(job)
    elif FLAGS.scheme == 'greedy':
        ret = cluster.greedy_placement(job)
    elif FLAGS.scheme == 'gandiva':
        ret = cluster.gandiva_placement(job)
    elif FLAGS.scheme == 'count':
        ret = cluster.none_placement(job)
    else:
        ret = cluster.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    return ret
