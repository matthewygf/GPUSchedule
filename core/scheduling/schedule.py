import time
from core.scheduling import algorithm
from core import flags
from core import util
from core.network import network_service

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

    def _schedule(self, delta):
        if self.num_free_nodes() < 1:
            return
        jobs_all = self.jobs_manager.total_jobs(delta)
        scheduling_algo = algorithm.scheduling_algorithms[self.schedule]
        placement_algo = algorithm.placement_algorithms[self.placement]
        nodes, job, success = scheduling_algo(placement_algo, self.infrastructure, self.jobs_manager,delta)
        if success:
            extras = network_service.calculate_network_costs(self.infrastructure, job)
            job.add_network_costs(extras)
            self.add_to_running(nodes, job.job_id)
        else:
            assert (jobs_all == self.jobs_manager.total_jobs(delta))

    def _gen_jobs(self, delta_time):
        self.jobs_manager.gen_jobs(delta_time)

    def unfinished_node_count(self):
        nodes = self.collate_all_nodes()
        count = sum([not node.is_finished for node in iter(nodes.values())])
        return count

    def release_finished_jobs(self, current_time):
        jobs_to_finish = self.jobs_manager.prepare_finish_tasks(current_time)
        for jtf in jobs_to_finish:
            success = False
            for task_id, node_id in iter(jtf.tasks_running_on.items()):
                running_task = self.infrastructure.nodes[node_id].running_tasks.pop(task_id)
                assert not running_task.finished
                jtf.task_finished(task_id)
                self.infrastructure.nodes[node_id].release_allocated_resources(running_task)
                success = jtf.try_finished()
                if success:
                    self.jobs_manager.running_jobs.pop(jtf.job_id)
                    if jtf.job_id not in self.jobs_manager.finished_jobs:
                        self.jobs_manager.finished_jobs[jtf.job_id] = jtf
                if len(self.infrastructure.nodes[node_id].running_tasks) == 0:
                    self.jobs_manager.busy_nodes.remove(node_id)
            assert success

    def add_to_running(self, nodes, job_id):
        for k, v in iter(nodes.items()):
            self.jobs_manager.start_job(v, job_id)
            assert (k in self.jobs_manager.busy_nodes)

    def _clear_nodes(self):
        for k, v in iter(self.infrastructure.nodes.items()):
            if len(v.running_tasks) == 0:
                v.reset_resource()
            else:
                sum_workers = sum([1 for w in iter(v.running_tasks.values()) if not w.is_ps])
                if sum_workers != v.gpu_used:
                    v.reset_resource(sum_workers)

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
            time.sleep(1)
            # HACK
            self._clear_nodes()
            if current_remaining > 0:
                # TODO: this will likely to be changed
                self._schedule(delta_time)
            new_current_remaining = self.jobs_manager.total_jobs(delta_time)
            time.sleep(1)
            end_time = time.time()
            self.release_finished_jobs(end_time)
            delta_time = end_time - start_time
            current_remaining = new_current_remaining
            running_jobs = len(self.jobs_manager.running_jobs)
            self.pending_time = self.jobs_manager.average_pending_time()
            steps += 1
            util.print_fn("Remaining jobs: %d, Running Jobs: %d Finished Jobs %d" %
                          (new_current_remaining, running_jobs, len(self.jobs_manager.finished_jobs)))
            util.print_fn(self.jobs_manager.running_jobs.keys())
            for k, v in iter(self.infrastructure.nodes.items()):
                util.print_fn("Node %s is %s, GPU used %d, each node has tasks %s" %
                              (k,
                               'busy' if len(v.running_tasks) > 0 else 'free',
                               v.gpu_used,
                               str(v.running_tasks.keys())))

        finished_time = time.time()
        total_time_taken = finished_time - start_time
        util.print_fn("Total Time Taken in seconds: %d" % total_time_taken)

    def sort_job_trace(self):
        self.jobs_manager.sort_job_trace()
