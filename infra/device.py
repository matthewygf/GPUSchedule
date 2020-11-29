from collections import OrderedDict
import numpy as np

class Device(object):
    def __init__(self, device_id, node_id, memory_cap, enable_pack):
        """TODO: compute capability, bandwidth etc..."""
        self.device_id = device_id
        self.node_id = node_id
        self.memory = memory_cap
        self.memory_used = 0
        self.enable_pack = enable_pack
        self.running_tasks = OrderedDict()

    def add_task(self, task):
        result = False
        current_mem = self.get_current_memory()
        if (current_mem + task.gpu_memory_avg) < self.memory + 50:
            self.running_tasks[task.task_id] = task
            result = True
        return result
    
    def get_current_utilization(self):
        # go through each task and randomly sample from normal dist.
        util = 0
        for _, t in self.running_tasks.items():
            util += min(100, np.random.normal(loc=t.gpu_utilization_avg, scale=(t.gpu_utilization_max - t.gpu_utilization_avg)/2, size=1))
            util = min(util, 100)
        return util

    def get_current_memory(self):
        # go through each task and randomly sample from normal dist.
        mem = 0
        for _, t in self.running_tasks.items():
            mem += min(self.memory, np.random.normal(loc=t.gpu_memory_avg, scale=(t.gpu_memory_max - t.gpu_memory_avg)/2, size=1))
            mem = min(self.memory, mem)
        return mem

    def pop_task(self, task_id):
        return self.running_tasks.pop(task_id, None)
    
    def can_fit(self, task):
        # theoretically, as long as we don't exceed the memory we should be good to go
        if self.get_current_memory() + task.gpu_memory_avg < self.memory + 50:
            return True
        return False