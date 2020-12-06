from log_manager import LogInfo
import logging
from math import inf
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

    def __init__(self, infrastructure, jobs_manager, log_manager, enable_migration=False):
        self.infrastructure = infrastructure
        # jobs manager maintains running jobs / finished jobs
        # NOTE: the queues are in jobs_manager
        self.jobs_manager = jobs_manager
        self.log_manager = log_manager
        self.placement = infrastructure.flags.scheme
        self.schedule = infrastructure.flags.schedule
        # TODO: RL agent
        self.enable_migration = enable_migration
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
        util.print_fn("ALL jobs: %d" % (jobs_all))
        scheduling_algo = algorithm.scheduling_algorithms[self.schedule]
        placement_algo = algorithm.placement_algorithms[self.placement]
        nodes, job, success = scheduling_algo(placement_algo, self.infrastructure, self.jobs_manager, delta)
        if success:
            if self.infrastructure.enable_network_costs:
                extras = network_service.calculate_network_costs(self.infrastructure, job)
                orginal_duration = job.duration
                job.add_network_costs(extras)
                util.print_fn("Job %s : Original duration %f , New duration %f" %
                            (job.job_id, orginal_duration, job.duration))
            self.add_to_running(nodes, job.job_id, delta)
            assert jobs_all - 1 == self.jobs_manager.total_jobs(delta), (
                'Expected %d - Got %d' % (jobs_all-1, self.jobs_manager.total_jobs(delta))
            )
        else:
            assert (jobs_all == self.jobs_manager.total_jobs(delta))

    def _scan_for_migrate(self):
        # if there is idle device
        # if there is overloaded device
        # we can identify whether idle devices satisfy the overloaded device's job.
        # if it does, we preempt that job, and put it back to the job queue
        idle_devices = []
        most_overload_device = None
        most_overload_count = 0
        for n in self.infrastructure.nodes.values():
            if n.is_idle():
                for d in n.device_cache.values():
                    if d.is_idle():
                        idle_devices.append((n.node_id, d.device_id))
                    else:
                        load = len(d.running_tasks)
                        if load > most_overload_count:
                            most_overload_device = d
    
        if len(idle_devices) == 0:
            return

        target_job = None
        target_util = 0
        if most_overload_device is not None and most_overload_count > 2:
            for t in most_overload_device.running_tasks.values():
                if t.gpu_utilization_avg > target_util:
                    target_job = t.job_id
                    target_util = t.gpu_utilization_avg
        
        if target_job is not None:
            logging.info("*******preempting: %s on device %s  " % (target_job, most_overload_device.device_id))
            self.jobs_manager.preempt(target_job)
    
    def _construct_info(self):
        idle_nodes = 0
        busy_nodes = 0
        idle_gpus = 0
        busy_gpus = 0
        avg_gpu_utilization = 0
        avg_gpu_memory_allocated = 0
        sum_gpu_memory_cap = 0
        for n in self.infrastructure.nodes.values():
            if n.is_idle():
                idle_nodes += 1
            else:
                busy_nodes += 1

            for d in n.device_cache.values():
                if d.is_idle():
                    idle_gpus += 1
                    avg_gpu_utilization += 0
                    avg_gpu_memory_allocated += 0
                else:
                    busy_gpus += 1
                    avg_gpu_utilization += d.get_current_utilization()
                    avg_gpu_memory_allocated += d.get_current_memory()
                sum_gpu_memory_cap += d.memory
        
        avg_gpu_utilization /= (idle_gpus + busy_gpus)
        avg_gpu_memory_allocated /= sum_gpu_memory_cap

        queuing_jobs = self.jobs_manager.queuing_jobs()
        running_jobs = len(self.jobs_manager.running_jobs)
        finished_jobs = len(self.jobs_manager.finished_jobs)

        # for all queuing jobs avg the pending time
        avg_pending_time = self.jobs_manager.avg_pending_time()

        return LogInfo(idle_nodes, busy_nodes, busy_gpus, idle_gpus,
                     avg_gpu_utilization, avg_gpu_memory_allocated, avg_pending_time,
                     running_jobs, queuing_jobs, finished_jobs)


    def unfinished_node_count(self):
        nodes = self.collate_all_nodes()
        count = sum([not node.is_finished for node in iter(nodes.values())])
        return count

    def release_finished_jobs(self, current_time):
        jobs_to_finish = self.jobs_manager.prepare_finish_tasks(current_time)
        logging.info("jobs to finish: %s" % (jobs_to_finish))
        # TODO: show jobs to finish, might have bugs.
        for jtf in jobs_to_finish:
            success = False
            for task_id, node_id in iter(jtf.tasks_running_on.items()):
                running_task = self.infrastructure.nodes[node_id].running_tasks.pop(task_id)
                assert not running_task.finished
                jtf.task_finished(task_id)
                reduce_interference_set = self.infrastructure.nodes[node_id].release_allocated_resources(running_task)
                if len(reduce_interference_set) > 0:
                    self.jobs_manager.reset_interference(reduce_interference_set)
                success = jtf.try_finished()
                if success:
                    logging.info("job %s finish" % (jtf.job_id))
                    self.jobs_manager.running_jobs.pop(jtf.job_id)
                    if jtf.job_id not in self.jobs_manager.finished_jobs:
                        self.jobs_manager.finished_jobs[jtf.job_id] = jtf
                if len(self.infrastructure.nodes[node_id].running_tasks) == 0:
                    self.jobs_manager.busy_nodes.remove(node_id)
            assert success

    def add_to_running(self, nodes, job_id, delta_time):
        for k, v in iter(nodes.items()):
            self.jobs_manager.start_job(v, job_id, delta_time)
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
        current_remaining = self.jobs_manager.remaining_jobs(delta_time)
        queuing_jobs = self.jobs_manager.queuing_jobs(delta_time)
        running_jobs = len(self.jobs_manager.running_jobs)
        steps = 0
        while current_remaining + running_jobs > 0:
            time.sleep(1)
            # scale factor - scale the running minutes
            _ = self.jobs_manager.gen_jobs(delta_time, scale_factor=0.5)
            if self.jobs_manager.queuing_jobs(delta_time) > 0:
                self._schedule(delta_time)
            current_remaining = self.jobs_manager.remaining_jobs(delta_time)
            queuing_jobs = self.jobs_manager.queuing_jobs(delta_time)
            delta_time += 1
            time.sleep(1)
            self.jobs_manager.step()
            self.release_finished_jobs(delta_time)
            running_jobs = len(self.jobs_manager.running_jobs)
            if self.enable_migration and delta_time > 100:
                self._scan_for_migrate()
            steps += 1
            loginfo = self._construct_info()
            self.log_manager.step_cluster(loginfo, delta_time)
            util.print_fn("Remaining jobs: %d, Queuing Jobs: %d Running Jobs: %d Finished Jobs %d" %
                           (current_remaining, queuing_jobs, running_jobs, len(self.jobs_manager.finished_jobs)))
            util.print_fn("Total Jobs in Memory: %d" % (current_remaining+queuing_jobs+running_jobs+len(self.jobs_manager.finished_jobs)))
            util.print_fn("running jobs: %s " % (self.jobs_manager.running_jobs.keys()))

        finished_time = time.time()
        total_time_taken = finished_time - start_time
        util.print_fn("Total Time Taken in seconds: %d" % total_time_taken)

