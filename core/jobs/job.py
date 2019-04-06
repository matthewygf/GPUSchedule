import math
from core import util
from core import models
from model import model_factory
import csv
import time
import sys

class Task(object):
    """NOTE: 
    each job can have multiple tasks, 
    each task can be identified from job_id
    assume each task from the same job has the same duration time.
    """
    def __init__(self, 
                 job_id,
                 task_id,
                 duration,
                 is_ps=False,
                 model_size=0,
                 cpu=0,
                 mem=0,
                 gpu=0):
        self.job_id = str(job_id)
        self.task_id = str(task_id)
        self.is_ps = is_ps
        self.cpu = cpu
        self.mem = mem
        self.gpu = gpu
        self.model_size = model_size
        self.start_time = 0
        self.duration = duration
        self.started = False
        self.running = False
        self.finished = False
        self.migration_count = 0

    def execute(self):
        self.start_time = time.time()
        self.migration_count += 1
        self.started = True
        self.running = True
    
class Job(object):
    """
    NOTE:
    Assumption:
    1. each GPU is a worker, in reality, this could be different.
    2. all job is a parameter server approach.
    3. if number of gpu required by a job is less than 1,
        assume only 1 gpu, no worker , no ps.
    4. if number of gpu required by a job is greater than 1,
        assumed ps is the mod of num_gpu_p_node,
        if less than 4, then it is between model replica, no need ps.
    5. assume each task (ps, workers) have same amount of cpu.
    6. assume each task (ps, workers) have same amount of mem.
    7. assume Synchronize SGD.
    TODO:
    #http://arxiv.org/abs/1807.11205
    1. All reduce jobs

    #http://arxiv.org/abs/1712.01887
    2. Maybe Deep Gradient Compression (DGC)
    """
    def __init__(self, 
                 job_id, 
                 model, 
                 duration, 
                 iterations, 
                 interval, 
                 submit_time,
                 gpu=0):
        self.job_id = str(job_id)
        self.started = False
        self.running = False
        self.finished = False
        self.failed_schedule = 0
        self.start_time = 0
        self.interations = iterations
        self.duration = int(duration)
        self.submit_time = int(submit_time)
        self.pending_time = 0.0
        self.model = model
        self.model_size = model_factory.model_sizes[model]
        self.is_cnn = self.model in model_factory.cnn_models
        self.migration_count = 0
        self.ps_count = gpu // 4 if gpu > 1 else 0
        self.worker_count = gpu 
        self.gpus = gpu
        self.task_count = self.ps_count + self.worker_count
        self.task_id = ['worker' + str(task) if task <= self.worker_count else 'ps' + str(task - self.worker_count) for task in range(1, self.task_count+1) ]
        self.cpus_per_task = 4 # heuristic
        self.memory_per_task = 6 # heuristic
        self.iterations = iterations
        self.interval = interval
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
            is_ps = 'ps' in taskidx
            needgpu = 1 if not is_ps else 0
            t = Task(self.job_id, self.job_id+"_"+taskidx,
                     self.duration, is_ps,
                     self.model_size, self.cpus_per_task,
                     self.memory_per_task, needgpu)
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

    def try_execute(self):
        """
        only execute when all tasks are executing
        """
        executed_tasks_count = 0
        for k, v in iter(self.tasks.items()):
            if v.running and not v.finished:
                executed_tasks_count += 1

        if executed_tasks_count == self.task_count:
            self.start_time = time.time()
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
