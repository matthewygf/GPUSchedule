from collections import OrderedDict
import numpy as np

class Device(object):
    def __init__(self, device_id, node_id, memory_cap, enable_pack):
        """TODO: compute capability, bandwidth etc..."""
        self.device_id = device_id
        self.node_id = node_id
        self.memory = memory_cap
        self.enable_pack = enable_pack
        self.running_tasks = OrderedDict()

    def is_idle(self):
        return len(self.running_tasks) == 0

    def add_task(self, task, pack=False):
        result = False
        if self.can_fit(task):
            if not pack and len(self.running_tasks) > 0:
                # not placing
                return False
            self.running_tasks[task.task_id] = task
            result = True
        return result

    def reset(self):
        self.running_tasks = OrderedDict()
    
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
            mem += min(self.memory, t.gpu_memory_max)
            mem = min(self.memory, mem)
        return mem

    def pop_task(self, task_id):
        return self.running_tasks.pop(task_id, None)
    
    def can_fit(self, task):
        # theoretically, as long as we don't exceed the memory we should be good to go
        current_mem = self.get_current_memory()

        # one can think of this as the PCIe bandwidth setting.
        if len(self.running_tasks) >= 3:
            return False

        if self.memory - (current_mem + task.gpu_memory_max) > 500:
            return True
        return False