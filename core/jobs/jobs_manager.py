from core.jobs.job import Job
from typing import List
from core import util
from core.jobs import job_generator
import logging

class JobsManager(object):
    """
    This acts like the Application/Framework master
    NOTE: Assume jobs use all reduce instead of ps / workers.
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
    
    def sort_job_trace(self):
        for i, q in enumerate(self.job_queue_manager.queues):
            self.job_queue_manager.queues[i] = sorted(q, key=lambda x: int(x.submit_time))

    def pop(self, delta, queue_idx=0, job_in_queue=0):
        j = self.job_queue_manager.pop(queue_idx, job_in_queue)
        # if j.submit_time >= delta:
        #     return j
        return j
    
    def get_insert_position(self, jobs):
        ''' based on flags, identify the jobs position to insert'''
        insert_pos = []
        idx = 0
        if self.flags.schedule == "fifo":
            for _ in jobs:
                insert_pos.append(idx)
                idx += 1
        else:
            raise NotImplementedError()
    
    def get_queue_position(self, jobs):
        ''' based on flags, identify the queues position to insert'''
        queue_pos = []
        idx = 0
        if self.flags.schedule == "fifo":
            for _ in jobs:
                queue_pos.append(idx)
        else:
            raise NotImplementedError()

    def _insert(self, jobs):
        jobs_insert_position = self.get_insert_position(jobs)
        queue_insert_position = self.get_queue_position(jobs)
        for job, q_index, j_index in zip(jobs, queue_insert_position, jobs_insert_position):
            self.job_queue_manager.insert(job, q_index, j_index)

    def start_job(self, node, job_id):
        executed_job, started_task_count = node.execute_job(job_id)
        assert started_task_count > 0

        if started_task_count > 0:
            # this node has started some tasks, so it is busy
            if node.node_id not in self.busy_nodes:
                self.busy_nodes.append(node.node_id)
                # because we have started tasks,
                # CUDA would have been allocated,
                # we estimate how much a task would use up the gpu utilz
                # node.estimate_gpu_utilization()
        
        # This case is when some tasks are ready, but not all.
        if executed_job is None:
            return False

        # for jobs that are scattered across, we only need one
        # as each job have a dict to see where it's tasks are.
        if executed_job.job_id not in self.running_jobs:
            self.running_jobs[executed_job.job_id] = executed_job
        
        return True

    def remaining_jobs(self, delta_time):
        if self.replay_trace:
            return self.job_generator.remaining_jobs()
        raise NotImplementedError()

    def total_jobs(self, delta_time):
        ''''''
        if self.replay_trace:
            # not yet generated
            return self.remaining_jobs() + self.job_queue_manager.total_jobs(delta_time)

        return self.job_queue_manager.total_jobs(delta_time)
    
    def total_finished_jobs(self):
        return len(self.finished_jobs)

    def gen_jobs(self, delta_time):
        samples = self.job_generator.generate_jobs(delta_time)
        # put into the queue or queues.
        logging.info("generated: %d" % len(samples))
        converted_jobs = []
        for row in samples.iterrows():
            j = Job(row.name)
            converted_jobs.append(j)
        self._insert(converted_jobs)
        return len(samples)

    def average_pending_time(self):
        # TODO:
        pass

    def prepare_finish_tasks(self, current_time):
        jobs_to_finish = []
        for k, v, in iter(self.running_jobs.items()):
            duration = current_time - v.start_time
            # replay faster.
            if duration < (v.duration / 5): continue
            jobs_to_finish.append(v)
        return jobs_to_finish

            
            
    

