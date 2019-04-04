from core import util
class JobsManager(object):
    """
    This acts like the Application/Framework master
    NOTE: All jobs right now follows parameter server frameworks.
    TODO: Allreduce NCCL jobs
    """
    def __init__(self, job_queue_manager):
        self.job_queue_manager = job_queue_manager
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
    
    def insert(self, job, queue_idx=0, job_in_queue=0):
        return self.job_queue_manager.insert(job, queue_idx, job_in_queue)

    def start_job(self, node, job_id):
        executed_job, started_task_count = node.execute_job(job_id)
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

    def total_jobs(self, delta_time):
        return self.job_queue_manager.total_jobs(delta_time)
    
    def total_finished_jobs(self):
        return len(self.finished_jobs)

    def gen_jobs(self, delta_time):
        pass

    def average_pending_time(self):
        # TODO:
        pass

    def prepare_finish_tasks(self, current_time):
        jobs_to_finish = []
        for k, v, in iter(self.running_jobs.items()):
            duration = current_time - v.start_time
            if duration < (v.duration / 10): continue
            jobs_to_finish.append(v)
        return jobs_to_finish

            
            
    

