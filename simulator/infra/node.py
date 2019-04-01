import copy
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
        self.node_id = str(node_id)
        self.cpu_count = cpus
        self.gpu_count = gpus
        self.mem_size = memory

        self.network_usage = 0
        self.cpu_used = 0
        self.gpu_used = 0
        self.mem_used = 0

        self.running_jobs = {}
        self.placed_jobs = {}
        self.running_tasks = {}
        self.placed_tasks = {}

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

    def release_allocated_resources(self, job, placed_only=False):
        """NOTE: release"""
        # clear tasks
        for t in iter(job.tasks.values()):
            pop_t = self.placed_tasks.pop(t.task_id, None)
            if pop_t is not None:
                self.cpu_used -= pop_t.cpu
                self.gpu_used -= pop_t.gpu
                self.mem_used -= pop_t.mem
        # assume task in the node is the main consumer
        if placed_only:
            self.placed_jobs.pop(job.job_id)
            return True
        else:
            self.placed_jobs.pop(job.job_id, None)
            #TODO:
            # running jobs finished etc...
        return False

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
        num_tasks_mems_can_fit = len(tasks) if mems_task_offset  >= 0 else len(tasks) + mems_task_offset
        num_ps_tasks_can_fit = min(num_tasks_cpus_can_fit, num_tasks_mems_can_fit) - num_tasks_gpus_can_fit
        return num_ps_tasks_can_fit, num_tasks_gpus_can_fit

    def execute_job(self, job_id):
        # check placed tasks in current node that is correspond to same job_id
        result = False
        started_jobs = []
        started_tasks = []
        job_to_execute = self.placed_jobs.pop(job_id, None)
        if job_to_execute is None:
            raise ValueError()
        for jt_idx in iter(job_to_execute.tasks.keys()):
            jt = self.placed_tasks.pop(jt_idx, None)
            if jt is not None:
                jt.execute()
                self.running_tasks[jt_idx] = jt
        job_to_execute.execute()
        self.running_jobs[job_id] = job_to_execute 
        # NOTE: Assume if job in running jobs, then every tasks above is added.
        result = (job_id in self.running_jobs)
        util.print_fn("node %s, total len of placed tasks: %d, total len of placed jobs %d" % 
                        (self.node_id, len(self.placed_tasks), len(self.placed_jobs)))
        util.print_fn("node %s, total len of running tasks: %d, total len of running jobs %d" % 
                        (self.node_id, len(self.running_tasks), len(self.running_jobs)))
        return result

    def try_reserve_and_placed_task(self, task):
        result = False
        cpu_offset = self.cpu_free() - task.cpu
        mem_offset = self.mem_free() - task.mem
        gpu_offset = self.gpu_free() - task.gpu
        result = (cpu_offset >= 0) and (mem_offset >= 0) and (gpu_offset >= 0)
        if not result:
            return result
        self.cpu_used += task.cpu
        self.mem_used += task.mem
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
                #util.print_fn("Found at least 1 task %s, in node %s" % (t, self.node_id))
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
            util.print_fn("node fit current job %s Trying to allocate job" % job.job_id)

            copy_j = job.tasks.copy()
            for t in iter(copy_j.values()):
                result = self.try_reserve_and_placed_task(t)
            result = self.try_reserve_and_placed_job(job, is_single)
            if not result:
                # Not executed yet
                self.release_allocated_resources(job, placed_only=True)
                util.print_fn("Job does not fit on node", util.LOG_LEVEL_WARNING)
                return result
            util.print_fn("placed job %s, tasks %d at node %s" % (job.job_id, len(job.tasks), self.node_id))
        else:
            util.print_fn("Job does not fit on node", util.LOG_LEVEL_WARNING)
        return result