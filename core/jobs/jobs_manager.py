from core.jobs.job import Job
from typing import List
from core import util
from core.jobs import job_generator
import logging
from core.jobs.utils import clusterize
import numpy as np
import infra.interference as interference 
class JobsManager(object):
    """
    This acts like the Application/Framework master
    """
    def __init__(self, flags, job_queue_manager):
        self.job_queue_manager = job_queue_manager
        self.flags = flags
        if flags.trace_file:
            self.job_generator = job_generator.JobTraceReader(flags.trace_file)
            self.job_generator.prepare_jobs()
            self.replay_trace = True
        else:
            self.replay_trace = False
            self.job_generator = job_generator.JobGenerator()
        self.running_jobs = {}
        self.finished_jobs = {}
        self.busy_nodes = []
        self.interference_factor = interference.FACTOR
    
    def sort_job_trace(self):
        for i, q in enumerate(self.job_queue_manager.queues):
            self.job_queue_manager.queues[i] = sorted(q, key=lambda x: int(x.submit_time))

    def get_next_job(self, delta, queue_idx=0, job_in_queue=0):
        j = self.job_queue_manager.get_next_job(queue_idx, job_in_queue)
        if j.submit_time <= delta:
            return j
        logging.info("job %s should not be scheduled yet... submitted time %.2f, current time %.2f" % (j.job_id, delta, j.submit_time))
        return None

    def pop(self, delta, queue_idx=0, job_in_queue=0):
        j = self.job_queue_manager.pop(queue_idx, job_in_queue)
        if j.submit_time <= delta:
            return j
        self.job_queue_manager.insert(queue_idx, job_in_queue)
        return None
    
    def get_insert_position(self, jobs):
        ''' based on flags, identify the jobs position to insert'''
        insert_pos = []
        idx = 0
        if self.flags.schedule == "fifo":
            for _ in jobs:
                insert_pos.append(idx)
                idx += 1
        elif self.flags.schedule.startswith("horus") or self.flags.schedule == "gandiva":
            # doesn't matter, let the queue do its priority heapsort
            for _ in jobs:
                insert_pos.append(idx)
                idx += 1
        else:
            raise NotImplementedError()
        return insert_pos

    def add_pending_time(self):
        num_queues = len(self.job_queue_manager.queues)
        for i in range(0, num_queues):
            num_jobs = len(self.job_queue_manager.queues[i])
            for j in range(0, num_jobs):
                self.job_queue_manager.queues[i][j].step_pending()

    def pending_time_infos(self):
        num_queues = len(self.job_queue_manager.queues)
        pending_time = 0
        pending_cnt = 0
        max_pending = 0
        all_pend = []
        for i in range(0, num_queues):
            num_jobs = len(self.job_queue_manager.queues[i])
            for j in range(0, num_jobs):
                j_pend = self.job_queue_manager.queues[i][j].pending_time
                max_pending = max(j_pend, max_pending)
                all_pend.append(j_pend)
                pending_time += j_pend
            pending_cnt += num_jobs

        return pending_time / (pending_cnt + 1e-9) , float(np.median(all_pend)), max_pending

    def get_queue_position(self, jobs):
        ''' based on flags, identify the queues position to insert'''
        queue_pos = []
        idx = 0
        if self.flags.schedule == "fifo" or self.flags.schedule == "gandiva":
            for _ in jobs:
                queue_pos.append(idx)
            return queue_pos
        elif self.flags.schedule == "horus":
            #one queue also in horus normal ver.
            for _ in jobs:
                queue_pos.append(idx)
            return queue_pos
        elif self.flags.schedule == "horus+":
            if len(jobs) == 0:
                return queue_pos
            # kmeans and see.
            # cluster all jobs again.
            centroids, queue_pos, loss = clusterize(jobs, k=len(self.job_queue_manager.queues))
            logging.info("k-means: %s queues pos: %s - loss at %.2f" % (str(centroids), str(queue_pos), loss))
            return queue_pos
        else:
            raise NotImplementedError()
        return queue_pos


    def insert(self, jobs, queue_insert_position=None):
        if self.flags.schedule == "horus+" and queue_insert_position is None:
            q_len = self.job_queue_manager.total_jobs()
            o_len = len(jobs)
            queued_jobs = self.job_queue_manager.pop_all_queuing_jobs()
            jobs = queued_jobs + jobs
            assert len(jobs) == q_len + o_len
            # logging.info("jobs1 : %d" % len(jobs))
        jobs_insert_position = self.get_insert_position(jobs)
        assert len(jobs) == len(jobs_insert_position)
        # logging.info("jobs2 : %d" % len(jobs))

        if queue_insert_position is None:
            queue_insert_position = self.get_queue_position(jobs)
        assert len(jobs) == len(queue_insert_position), (
            "Expected %d - Got %d" % (len(jobs), len(queue_insert_position))
        )
        # logging.info("jobs3 : %d" % len(jobs))

        for job, q_index, j_index in zip(jobs, queue_insert_position, jobs_insert_position):
            self.job_queue_manager.insert(job, q_index, j_index)
            # logging.info("%d %d %d" ,self.job_queue_manager.total_jobs(), q_index, j_index)
        if self.flags.schedule == "horus+" and queue_insert_position is None:
            assert len(jobs) == self.job_queue_manager.total_jobs(), (
                "Expected %d Got %d" % (len(jobs), self.job_queue_manager.total_jobs())
            )
            

    def step(self):
        self.add_pending_time()
        for j in self.running_jobs.values():
            j.step()
        if self.flags.schedule == "horus+":
            self.job_queue_manager.update_credits()

    def preempt(self, job_id, infrastructure):
        poped_job = self.running_jobs.pop(job_id)
        poped_already = set()
        for t, n in poped_job.tasks_running_on.items():
            if n not in poped_already:
                infrastructure.nodes[n].placed_jobs.pop(job_id)
                poped_already.add(n)

            pop_t = infrastructure.nodes[n].running_tasks.pop(t, None)
            if pop_t is not None:
                reduce_interference_set = infrastructure.nodes[n].release_allocated_resources(pop_t, reserved=True)
                if len(reduce_interference_set) > 0:
                    # reduce interference for poped task
                    if t in reduce_interference_set and pop_t.interfered:
                        pop_t.interfered = False
                        pop_t.duration = pop_t.duration * (1-self.interference_factor)
                        poped_job.tasks[t] = pop_t
                        reduce_interference_set.pop(t)

                    # reduce interference for the rest
                    self.reset_interference(reduce_interference_set)

        poped_job.preempted()
        self.insert([poped_job])

    def reset_interference(self, set_of_jobs):
        for t, j in set_of_jobs.items():
            if j in self.running_jobs:
                running_j = self.running_jobs[j]
                running_t = running_j.tasks[t]
                if running_t.interfered:
                    running_t.interfered = False
                    running_t.duration = running_t.duration * (1-self.interference_factor)
                    running_j.tasks[t] = running_t
                self.running_jobs[j] = running_j

    def start_job(self, node, job_id, delta_time):
        executed_job, started_task_count = node.execute_job(job_id, delta_time)
        assert started_task_count > 0

        if started_task_count > 0:
            # this node has started some tasks, so it is busy
            if node.node_id not in self.busy_nodes:
                self.busy_nodes.append(node.node_id)
        
        # This case is when some tasks are ready, but not all.
        if executed_job is None:
            return False

        # for jobs that are scattered across, we only need one
        # as each job have a dict to see where it's tasks are.
        if executed_job.job_id not in self.running_jobs:
            self.running_jobs[executed_job.job_id] = executed_job
        
        return True

    def remaining_jobs(self, delta_time=None):
        if self.replay_trace:
            return self.job_generator.remaining_jobs()
        raise NotImplementedError()

    def queuing_jobs(self, delta_time=None):
        return self.job_queue_manager.total_jobs(delta_time)

    def total_jobs(self, delta_time):
        ''''''
        if self.replay_trace:
            # not yet generated
            return self.remaining_jobs() + self.job_queue_manager.total_jobs(delta_time)

        return self.job_queue_manager.total_jobs(delta_time)
    
    def total_finished_jobs(self):
        return len(self.finished_jobs)

    def gen_jobs(self, delta_time, scale_factor=1):
        samples = self.job_generator.generate_jobs(delta_time)
        # put into the queue or queues.
        logging.info("generated: %d" % len(samples))
        converted_jobs = []
        for idx, row in samples.iterrows():
            j = Job(idx, row.minutes * scale_factor, row.normalized_time, row.gpu_per_container,
                    gpu_utilization_avg=row.gpu_utilization_avg, gpu_utilization_max=row.gpu_utilization_max,
                    gpu_memory_max=util.convert_bytes(row.memory_max, unit="MiB"), 
                    gpu_memory_avg=util.convert_bytes(row.memory_avg, unit="MiB"),
                    total_gpus=row.used_gpus)
            converted_jobs.append(j)
        self.insert(converted_jobs)
        return len(samples)

    def prepare_finish_tasks(self, current_time):
        jobs_to_finish = []
        for k, v, in iter(self.running_jobs.items()):
            if v.time_processed() < (v.duration): 
                logging.info("job %s should finish at %d , now at %d" % (k, v.duration, v.time_processed()))
                continue
            jobs_to_finish.append(v)
        return jobs_to_finish

            
            
    

