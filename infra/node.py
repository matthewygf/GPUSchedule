import copy
from core import util
import time


class Node(object):
    def __init__(self, rack_id, node_id, cpus=0, gpus=0, memory=0):
        self.node_id = str(node_id)
        self.cpu_count = cpus
        self.gpu_count = gpus
        self.mem_size = memory
        self.rack_id = rack_id
        self.network_usage = 0
        self.cpu_used = 0
        self.gpu_used = 0
        self.mem_used = 0

        # NOTE: all tasks are deep copied so can be safely deleted upon finished
        # we just wanted to keep track of the running tasks, 
        # and the placed_jobs was for tracking
        self.running_tasks = {}
        self.placed_tasks = {}
        self.placed_jobs = {}
        self.finished_tasks = []

    def get_network_usage(self):
        # assumed the jjob can 
        return self.network_usage

    def check_util(self):
        result = (float(self.cpu_used) / float(self.cpu_count),
                  float(self.gpu_used) / float(self.gpu_count),
                  float(self.mem_used) / float(self.mem_size))

        return result

    def resize_node(self, cpus, gpus, memory):
        self.cpu_count = cpus
        self.gpu_count = gpus
        self.mem_size = memory

    def cpu_free(self):
        result = (self.cpu_count - self.cpu_used)
        return result

    def gpu_free(self):
        result = (self.gpu_count - self.gpu_used)
        return result

    def mem_free(self):
        result = (self.mem_size - self.mem_used)
        return result

    def is_free(self):
        return self.gpu_free() > 0 or self.cpu_free() > 0 or self.mem_free() > 0

    def reset_resource(self, num=0, gpu=1, cpu=4, mem=6):
        assert len(self.running_tasks) >= num
        self.gpu_used = num * gpu
        self.cpu_used = num * cpu
        self.mem_used = num * mem

    def release_allocated_resources(self, task):
        """NOTE: release"""
        # clear tasks
        self.cpu_used -= task.cpu
        if not task.is_ps:
            assert task.gpu == 1
            self.gpu_used -= task.gpu
            assert self.gpu_used < 4, task.task_id
        assert self.gpu_used >= 0
        self.mem_used -= task.mem
        self.finished_tasks.append(task.task_id)

    def can_fit_num_task(self, tasks):
        """
        NOTE: CRITICAL STUFF
        return integer, -1 if can fit all tasks.
        """
        worker_count = sum([1 for t in tasks.values() if not t.is_ps])
        first_task = next(iter(tasks.values()))
        cpu_per_task = first_task.cpu
        mem_per_task = first_task.mem

        gpus_offset = self.gpu_free() - worker_count
        cpus_task_offset = (self.cpu_free() // cpu_per_task) - len(tasks)
        mems_task_offset = (self.mem_free() // mem_per_task) - len(tasks)
        num_tasks_gpus_can_fit = worker_count if gpus_offset >= 0 else worker_count + gpus_offset
        num_tasks_cpus_can_fit = len(tasks) if cpus_task_offset >= 0 else len(tasks) + cpus_task_offset
        num_tasks_mems_can_fit = len(tasks) if mems_task_offset >= 0 else len(tasks) + mems_task_offset
        num_ps_tasks_can_fit = min(num_tasks_cpus_can_fit, num_tasks_mems_can_fit) - num_tasks_gpus_can_fit
        return num_ps_tasks_can_fit, num_tasks_gpus_can_fit

    def execute_job(self, job_id):
        # check placed tasks in current node that is correspond to same job_id
        started_task_count = 0
        job_to_execute = self.placed_jobs.pop(job_id, None)
        if job_to_execute is None:
            raise ValueError()

        for k, v in iter(job_to_execute.tasks_running_on.items()):
            if v == self.node_id:
                jt = self.placed_tasks.pop(k)
                jt.execute()
                self.running_tasks[k] = jt
        result, started_task_count = job_to_execute.try_execute()
        if result:
            return job_to_execute, started_task_count
        return None, started_task_count

    def calculate_utilization(self):
        """
        TODO: 
        Each model has a particular utilization we sampled from a range.
        Only used for Gandiva, Our Scheduler
        return:
            a tuple, each entry represent a gpu utilization.
        """
        return (0.0, 0.0, 0.0, 0.0)

    def try_reserve_and_placed_task(self, task):
        result = False
        time.sleep(1)
        cpu_offset = self.cpu_free() - task.cpu
        mem_offset = self.mem_free() - task.mem
        gpu_offset = self.gpu_free() - task.gpu
        result = (cpu_offset >= 0) and (mem_offset >= 0) and (gpu_offset >= 0)
        if not result:
            return result
        assert self.gpu_used + task.gpu <= self.gpu_count
        self.cpu_used += task.cpu
        self.mem_used += task.mem
        if not task.is_ps:
            assert task.gpu == 1
            self.gpu_used += task.gpu
        self.placed_tasks[task.task_id] = task
        return result

    def try_reserve_and_placed_job(self, job, count_task_in_current_node=False):
        """
        NOTE: 
            param:
            count_task_in_current_node: if count_task_in_current_node is True, 
            will count whether all the task in the job is in current node placed_tasks
        """
        found = False
        for t in iter(job.tasks.keys()):
            if t not in self.placed_tasks:
                if not count_task_in_current_node:
                    continue
                else:
                    return False
            else:
                # util.print_fn("Found at least 1 task %s, in node %s" % (t, self.node_id))
                found = True
                break

        self.placed_jobs[job.job_id] = job
        return found

    def try_alloc_job(self, job, is_single=False):
        """
        NOTE: right now this assume all tasks can fit then we placed the tasks and corresponding job.
        """
        result = False
        ps_tasks, worker_tasks = self.can_fit_num_task(job.tasks)

        if ps_tasks + worker_tasks >= job.task_count:
            copy_j = job.tasks.copy()
            placed = 0
            for t in iter(copy_j.values()):
                result = self.try_reserve_and_placed_task(t)
                if result:
                    job.tasks_running_on[t.task_id] = self.node_id
                    placed += 1
            if placed > 0:
                result = self.try_reserve_and_placed_job(job, is_single)
                if not result:
                    # Not executed yet
                    for jt in job.tasks.items():
                        job.tasks_running_on.pop(jt.task_id, None)
                        self.placed_tasks.pop(jt.task_id, None)
                        self.release_allocated_resources(jt)
                    self.placed_jobs.pop(job.job_id)
                    util.print_fn("RELEASED: Job does not fit on node", util.LOG_LEVEL_WARNING)
                    return result
                util.print_fn(
                    "placed SINGLE NODE job %s, num tasks %d on node %s" % (job.job_id, len(job.tasks), self.node_id))
        else:
            util.print_fn("Job does not fit on node", util.LOG_LEVEL_WARNING)
        return result
