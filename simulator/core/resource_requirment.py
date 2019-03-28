class ResourceRequirements(object):
    def __init__(self, cpu=0, gpu=0, mem=0):
        self.cpu_needed = cpu
        self.gpu_needed = gpu
        self.mem_needed = mem
