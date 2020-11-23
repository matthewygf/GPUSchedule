class CompareAbleByUtilization(object):
    
    def __lt__(self, other):
        '''max heap by utilization'''
        if self.gpu_utilization_avg:
            return self.gpu_utilization_avg > other.gpu_utilization_avg

        return False

class Task(CompareAbleByUtilization):
    """NOTE: 
    each job can have multiple tasks, 
    each task can be identified from job_id
    assume each task from the same job has the same duration time.
    """
    def __init__(self, 
                 job_id,
                 task_id,
                 duration,
                 gpu_utilization_avg=0,
                 gpu_utilization_max=0,
                 gpu_mem_avg=0,
                 gpu_mem_max=0,
                 cpu=0,
                 mem=0,
                 gpu=0):
        self.job_id = str(job_id)
        self.task_id = str(task_id)
        self.cpu = cpu
        self.mem = mem
        self.gpu = gpu
        self.start_time = 0
        self.gpu_utilization_avg = gpu_utilization_avg
        self.gpu_utilization_max = gpu_utilization_max
        self.gpu_mem_avg=gpu_mem_avg
        self.gpu_mem_max=gpu_mem_max
        self.duration = duration
        self.started = False
        self.running = False
        self.finished = False
        self.migration_count = 0

    def execute(self, delta_time):
        self.start_time = delta_time
        self.migration_count += 1
        self.started = True
        self.running = True


class Job(CompareAbleByUtilization):
    """
    NOTE:
    Assumption:
    1. each GPU is a worker, in reality, this could be different.
    2. all job is an all-reduce approach.
    3. if number of gpu required by a job is less than 1,
        assume only 1 gpu, no worker , no ps.
    4. if number of gpu required by a job is greater than 1,
        assumed ps is the mod of num_gpu_p_node,
        if less than 4, then it is between model replica, no need ps.
    5. assume each task (e.g. ps, workers) have same amount of cpu.
    6. assume each task (e.g. ps, workers) have same amount of mem.
    7. assume Synchronize SGD.
    """
    def __init__(self, 
                 job_id, 
                 duration, 
                 submit_time,
                 gpu_p_worker=0,
                 gpu_utilization_avg=0,
                 gpu_utilization_max=0,
                 gpu_memory_max=0,
                 gpu_memory_avg=0,
                 total_gpus=0):
        self.job_id = str(job_id)
        self.started = False
        self.running = False
        self.finished = False
        self.failed_schedule = 0
        self.start_time = 0
        self.duration = duration
        self.submit_time = int(submit_time)
        self.pending_time = 0.0
        self.migration_count = 0
        self.worker_count = total_gpus // gpu_p_worker 
        self.gpu_per_worker = gpu_p_worker
        self.gpus = total_gpus
        self.task_count = int(self.worker_count)
        self.task_id = ['worker' + str(task) for task in range(0, self.task_count) ]
        self.gpu_mem_avg = gpu_memory_avg
        self.gpu_mem_max = gpu_memory_max
        self.gpu_utilization_avg = gpu_utilization_avg
        self.gpu_utilization_max = gpu_utilization_max
        self.cpus_per_task = 12 # avg num cpu core requested 
        self.memory_per_task = 60 # avg mem requested
        self.tasks_running_on = {}
        self.tasks_finished = 0
        self.tasks = self.setup_tasks()
    
    def total_cpus_required(self):
        return self.cpus_per_task * self.task_count

    def total_mem_required(self):
        return self.memory_per_task * self.task_count

    def setup_tasks(self):
        result = {}
        for taskidx in self.task_id:
            t = Task(self.job_id, self.job_id+"_"+taskidx,
                     self.duration, gpu_utilization_avg=self.gpu_utilization_avg,
                     gpu_utilization_max=self.gpu_utilization_max, gpu_mem_avg=self.gpu_mem_avg,
                     gpu_mem_max=self.gpu_mem_max, cpu=self.cpus_per_task, mem=self.memory_per_task,
                     gpu=self.gpu_per_worker)
            result[t.task_id] = t 
        return result

    def is_waiting(self):
        result = not self.started and not self.running
        return result
    
    def is_preempted(self):
        result = self.started and not self.running
        return result

    def task_finished(self, t_id):
        if t_id in self.tasks and not self.tasks[t_id].finished:
            self.tasks[t_id].finished = True
            self.tasks_finished += 1

    def try_finished(self):
        if self.finished:
            return True
        # job wasn't deep copied to multiple nodes
        if self.tasks_finished == self.task_count:
            self.running = False
            self.finished = True
            return True
        return False

    def try_execute(self, delta_time):
        """
        only execute when all tasks are executing
        """
        executed_tasks_count = 0
        for k, v in iter(self.tasks.items()):
            if v.running and not v.finished:
                executed_tasks_count += 1

        if executed_tasks_count == self.task_count:
            self.start_time = delta_time
            self.migration_count += 1
            self.started = True
            self.running = True
            return True, executed_tasks_count
        return False, executed_tasks_count

    def add_network_costs(self, extra_s):
        self.duration += extra_s

    def is_distributed(self):
        return self.ps_count > 1

    def restart(self):
        self.running = True
        self.migration_count += 1
