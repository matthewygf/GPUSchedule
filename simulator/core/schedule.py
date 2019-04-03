from core import util
from core import lp
from core import flags
from core import algorithm
import time

FLAGS = flags.FLAGS

class Scheduler(object):
    """
    Scheduler object to manage jobs and schedule placements
    """
    def __init__(self, infrastructure, jobs_manager):
        self.infrastructure = infrastructure
        # jobs manager maintains running jobs / finished jobs
        # NOTE: the queues are in jobs_manager
        self.jobs_manager = jobs_manager
        self.placement = infrastructure.flags.scheme
        self.schedule = infrastructure.flags.schedule
        self.pending_time = 0.0
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

    def _schedule(self, *args):
        return algorithm.scheduling_algorithms[self.schedule](self, *args)
    
    def _gen_jobs(self, delta_time):
        self.jobs_manager.gen_jobs(delta_time)

    def unfinished_node_count(self):
        nodes = self.collate_all_nodes()
        count = sum([not node.is_finished for node in iter(nodes.values())])
        return count

    def release_finished_jobs(self, current_time):
        self.jobs_manager.release_finished_jobs(self.infrastructure, current_time)

    def add_to_running(self, nodes, job_id):
        for k, v in iter(nodes.items()):
            self.jobs_manager.start_job(v, job_id)
            assert(k in self.jobs_manager.busy_nodes)

    def start(self):
        start_time = time.time()
        delta_time = 0
        current_remaining = self.jobs_manager.total_jobs(delta_time)
        running_jobs = len(self.jobs_manager.running_jobs)
        steps = 0
        while current_remaining + running_jobs > 0:
            # NOTE: Make decision on whether to:
            # 1. Done: schedule new jobs 
            # 2. TODO: preempt running jobs 
            # 3. TODO: migrate running jobs
            # 4. TODO: stochastic job arrival process
            self._gen_jobs(delta_time)
            self._schedule(delta_time)
            new_current_remaining = self.jobs_manager.total_jobs(delta_time) 
            end_time = time.time()
            self.release_finished_jobs(end_time)
            delta_time = end_time - start_time
            current_remaining = new_current_remaining
            running_jobs = len(self.jobs_manager.running_jobs)
            self.pending_time = self.jobs_manager.average_pending_time()
            steps += 1
            if (steps>=2500) and ((steps % 2500) == 0):
                util.print_fn("Remaining jobs: %d, Running Jobs: %d Finished Jobs %d" %
                            (new_current_remaining, running_jobs, len(self.jobs_manager.finished_jobs)))
                util.print_fn(self.jobs_manager.running_jobs.keys())
        finished_time = time.time()
        Total_time_taken = finished_time - start_time
        util.print_fn("Total Time Taken in seconds: %d" % total_time_taken)

    def sort_job_trace(self):
        self.jobs_manager.sort_job_trace()
        