from core import util
from core import lp
from core import flags
from core import algorithm
import time

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
            ret = try_get_job_res(cluster, job_queue, p_job)
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
        job_queue.job_events.sort(key=lambda e:e.__getitem__('time'))

        logger.checkpoint(job_queue, event_time)


'''
Allocate job resource
'''
def try_get_job_res(cluster, job_queue, job):
    '''
    select placement scheme
    '''
    if FLAGS.scheme == 'yarn':
        ret = cluster.ms_yarn_placement(job_queue, job)
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

class Scheduler(object):
    """
    Scheduler object to manage jobs and schedule placements
    """
    def __init__(self, infrastructure, jobqueuemanager):
        self.infrastructure = infrastructure
        self.jq_manager = jobqueuemanager
        self.placement = infrastructure.flags.scheme
        self.schedule = infrastructure.flags.schedule
        # Scheduler should keep tracks of running jobs
        # and which nodes are busy
        self.jobs_finished = {}
        self.busy_nodes = []

        # TODO: RL agent
        self.agent = None

    def add_rack(self, rack):
        self.infrastructure.racks.append(rack)

    def collate_all_nodes(self):
        result = self.infrastructure.nodes
        return result

    def num_free_nodes(self):
        all_nodes = self.collate_all_nodes()
        return sum([n.is_free() for n in iter(all_nodes.values())])

    def _schedule(self, delta):
        return algorithm.scheduling_algorithms[self.schedule](self, delta)

    def unfinished_node_count(self):
        nodes = self.collate_all_nodes()
        count = sum([not node.is_finished for node in iter(nodes.values())])
        return count

    def release_finished_jobs(self, current_time):
        for node_id in self.busy_nodes:
            busy_node = self.infrastructure.nodes[node_id]
            finished_jobs_in_nodes = busy_node.try_finished_jobs(current_time)
            for fj in finished_jobs_in_nodes:
                self.jobs_finished[fj] = True
                busy_node.running_jobs.pop(fj)

    def add_to_running(self, node_ids, job_id):
        result = False
        self.jobs_finished[job_id] = False
        nodes = self.collate_all_nodes()
        for idx in node_ids:
            self.busy_nodes.append(idx)
            result = nodes[idx].execute_job(job_id)
            if not result:
                return result
        return result

    def start(self):
        start_time = time.time()
        delta_time = 0
        current_remaining = self.jq_manager.total_jobs()
        running_jobs = len([v for k, v in self.jobs_finished.items() if not v])
        while current_remaining + running_jobs > 56:
            # NOTE: Make decision on whether to:
            # 1. schedule new jobs
            # 2. preempt running jobs
            # 3. migrate running jobs
            self._schedule(delta_time)
            new_current_remaining = self.jq_manager.total_jobs() 
            #util.print_fn("Current remaining was %d, now is %d" % (current_remaining, new_current_remaining))
            end_time = time.time()
            self.release_finished_jobs(end_time)
            delta_time = end_time - start_time
            start_time = end_time
            current_remaining = new_current_remaining
            running_jobs = len([v for k, v in self.jobs_finished.items() if not v])

    def sort_job_trace(self):
        for i, q in enumerate(self.jq_manager.queues):
            self.jq_manager.queues[i] = sorted(q, key=lambda x: int(x.submit_time))
