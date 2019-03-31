from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core import util

'''
TODO: add cpu and network load support in class _Node
'''
class _Node(object):
    def __init__(self, id, num_gpu=0, num_cpu=0, mem=0):
        self.id = id
        self.num_cpu = num_cpu
        self.free_cpus = num_cpu
        self.num_gpu = num_gpu       
        self.free_gpus = num_gpu
        #network load: can be bw, or the amount of traffic
        # in and out should be the same
        self.network_in = 0
        self.network_out = 0

        self.mem = mem
        self.free_mem = mem

        #node class for gandiva
        self.job_gpu = 0
        self.num_jobs = 0

        util.print_fn('    Node[%d] has %d gpus, %d cpus, %d G memory' % (id, num_gpu, num_cpu, mem))
    
    def init_node(self, num_gpu=0, num_cpu=0, mem=0):
        if num_gpu != 0:
            self.num_gpu = num_gpu
            self.free_gpus = num_gpu
        if num_cpu != 0:
            self.num_cpu = num_cpu
            self.free_cpus = num_cpu
        if mem != 0:
            self.mem = mem
            self.free_mem = mem 

        self.add_gpus(self.num_gpu)        
        self.add_cpus(self.num_gpu)        


    ''' GPU  '''
    def add_gpus(self, num_gpu=0):
        pass

    def check_free_gpus(self):
        return self.free_gpus


    def alloc_gpus(self, num_gpu=0):
        '''
        If enough free gpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_gpu > self.free_gpus:
            return False
        else:
            self.free_gpus -= num_gpu
            return True

    def release_gpus(self, num_gpu=0):
        '''
        release using gpus back to free list
        '''
        if self.free_gpus + num_gpu > self.num_gpu:
            self.free_gpus = self.num_gpu
            return False
        else:
            self.free_gpus += num_gpu
            return True


    ''' CPU '''

    def add_cpus(self, num_cpu=0):
        pass

    def check_free_cpus(self):
        return self.free_cpus

    def alloc_cpus(self, num_cpu=0):
        '''
        If enough free cpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            return True

    def release_cpus(self, num_cpu=0):
        '''
        release using cpus back to free list
        '''
        if self.free_cpus + num_cpu > self.num_cpu:
            self.free_cpus = self.num_cpu
            return False
        else:
            self.free_cpus += num_cpu
            return True 


    '''network'''

    def add_network_load(self, in_load=0, out_load=0):
        self.network_in += in_load
        self.network_out += out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def release_network_load(self, in_load=0, out_load=0):
        self.network_in -= in_load
        self.network_out -= out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)

    def set_network_load(self, in_load=0, out_load=0):
        self.network_in = in_load
        self.network_out = out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def alloc_job_res(self, num_gpu=0, num_cpu=0):
        '''
        alloc job resource
        '''
        gpu = self.alloc_gpus(num_gpu)
        cpu = self.alloc_cpus(num_cpu)

        if cpu == False or gpu == False:
            self.release_gpus(num_gpu)
            self.release_cpus(num_cpu)
            return False

        return True 

    def release_job_res(self, node_dict):
        '''
        input is node_dict from placement
        {'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w2, ps2]}
        '''
        self.release_network_load(node_dict['network'], node_dict['network'])
        cpu = self.release_cpus(node_dict['num_cpu'])
        gpu = self.release_gpus(node_dict['num_gpu'])

        self.free_mem = self.free_mem + node_dict['mem']

        return (cpu and gpu)

    def release_job_gpu_cpu(self, num_gpu, num_cpu):
        '''
        input is gpu and cpu
        '''
        cpu = self.release_cpus(num_cpu)
        gpu = self.release_gpus(num_gpu)

        return (cpu and gpu)


class Node(object):
    def __init__(self, node_id, cpus=0, gpus=0, memory=0):
        self.node_id = node_id
        self.cpu_count = cpus
        self.gpu_count = gpus
        self.mem_size = memory

        self.network_usage = 0
        self.cpu_used = 0
        self.gpu_used = 0
        self.mem_used = 0

        self.running_jobs = []
        self.placed_jobs = []
        self.running_tasks = []
        self.placed_tasks = []

    def __eq__(self, other):
        result = self.node_id == other.node_id
        return result

    def get_network_usage(self):
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
        return self.gpu_free() > 0 and self.cpu_free() > 0 and self.mem_free() > 0

    def release_resources(self, job):
        self.cpu_used -= job.cpu
        self.gpu_used -= job.gpu
        self.mem_used -= job.mem
        assert self.jobs.__contains__(job), "Node did not contain the job specified"
        self.jobs.remove(job)

    def can_fit_num_task(self, tasks):
        """
        NOTE: CRITICAL STUFF
        return integer, -1 if can fit all tasks.
        """
        worker_count = sum([1 for t in tasks if not t.is_ps])
        cpu_per_task = tasks[0].cpu
        mem_per_task = tasks[0].mem

        gpus_offset = self.gpu_free() - worker_count
        cpus_task_offset = (self.cpu_free() // cpu_per_task) - len(tasks)
        mems_task_offset = (self.mem_free() // mem_per_task) - len(tasks)
        num_tasks_gpus_can_fit = worker_count if gpus_offset >= 0 else worker_count + gpus_offset
        num_tasks_cpus_can_fit = len(tasks) if cpus_task_offset >= 0 else len(tasks) + cpus_task_offset
        num_tasks_mems_can_fit = len(tasks) if mems_task_offset  >= 0 else len(tasks) + mems_task_offset
        num_ps_tasks_can_fit = min(num_tasks_cpus_can_fit, num_tasks_mems_can_fit) - num_tasks_gpus_can_fit
        return num_ps_tasks_can_fit, num_tasks_gpus_can_fit

    def execute_job(self, job_id):
        # check placed tasks in current node that is correspond to same job_id
        result = False
        started_jobs = []
        started_tasks = []
        for i, job in enumerate(self.placed_jobs):
            if job.job_id == job_id:
                for j, t in enumerate(self.placed_tasks):
                    if t.job_id == job_id:
                        t.execute()
                        started_tasks.append(j)
                        self.running_tasks.append(t)    
                job.execute()
                self.running_jobs.append(job)
                started_jobs.append(i)
                break

        for idx in started_tasks:
            self.placed_tasks.pop(idx)

        for idx in started_jobs:
            self.placed_jobs.pop(idx)
            result = True

        util.print_fn("node %d, total len of placed tasks: %d, total len of placed jobs %d" % 
                        (self.node_id, len(self.placed_tasks), len(self.placed_jobs)))
        util.print_fn("node %d, total len of running tasks: %d, total len of running jobs %d" % 
                        (self.node_id, len(self.running_tasks), len(self.running_jobs)))
        return result

    def try_alloc_job(self, job):
        result = False
        ps_tasks, worker_tasks = self.can_fit_num_task(job.tasks)
        if ps_tasks + worker_tasks >= job.task_count:
            self.placed_jobs.append(job)
            self.placed_tasks = self.placed_tasks + job.tasks
            util.print_fn("placed job %d, tasks %d at node %d" % (job.job_id, len(job.tasks), self.node_id))
            result = True
        else:
            util.print_fn("Job does not fit on node", util.LOG_LEVEL_WARNING)
        return result