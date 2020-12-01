import csv
import os

class LogInfo(object):
    def __init__(self, 
                 num_idle_nodes,
                 num_busy_nodes,
                 num_busy_gpus,
                 num_idle_gpus,
                 avg_gpu_utilization,
                 avg_gpu_memory_allocated,
                 avg_pending_time,
                 num_running_jobs,
                 num_queuing_jobs,
                 num_finish_jobs) -> None:
        self.idle_ns = num_idle_nodes
        self.busy_ns = num_busy_nodes
        self.busy_gs = num_busy_gpus
        self.idle_gs = num_idle_gpus
        self.avg_g_utils = avg_gpu_utilization
        self.avg_g_mem = avg_gpu_memory_allocated
        self.avg_pending = avg_pending_time
        self.num_running_jobs = num_running_jobs
        self.num_queuing_jobs = num_queuing_jobs
        self.num_finish_jobs = num_finish_jobs

class LogManager(object):
    def __init__(self, log_path, flags):
        self.log_path = log_path
        self.flags = flags
        self.is_count = self.flags.scheme == 'count'
    
    def init(self, infrastructure):
        self.log_cluster = os.path.join(self.log_path, 'cluster.csv')
        self.log_job = os.path.join(self.log_path, 'job.csv')
        if not self.is_count:
            self.log_cpu = os.path.join(self.log_path, 'cpu.csv')
            self.log_gpu = os.path.join(self.log_path, 'gpu.csv')
            self.log_network = os.path.join(self.log_path, 'network.csv')
            self.log_mem = os.path.join(self.log_path, 'memory.csv')
        
        self._init_all_csv(infrastructure)

    def _init_all_csv(self, infrastructure):
        # init cluster log
        cluster_log = open(self.log_cluster, 'w+')
        writer = csv.writer(cluster_log)
        writer.writerow([
            'delta', 'num_idle_nodes', 'num_busy_nodes',
            'num_busy_gpus', 'num_idle_gpus', 'avg_gpu_utilization',
            'avg_gpu_memory_allocated', 'avg_pending_time',
            'num_running_jobs', 'num_queuing_jobs',
            'num_finish_jobs'])
        cluster_log.close()

        # init cpu, gpu, mem, network logs
        if not self.is_count:
            # 1. cpu
            cpu_log = open(self.log_cpu, 'w+')
            writer = csv.writer(cpu_log)
            writer.writerow(['time']+['cpu' + str(i) for i in range(len(infrastructure.nodes))])
            cpu_log.close()
            del cpu_log

            # 2. gpu
            gpu_log = open(self.log_gpu, 'w+')
            writer = csv.writer(gpu_log)
            writer.writerow(['time']+['gpu' + str(i) for i in range(infrastructure.get_total_gpus())])
            gpu_log.close()
            del gpu_log

            # 3. mem
            mem_log = open(self.log_mem, 'w+')
            writer = csv.writer(mem_log)
            writer.writerow(['time', 'max', '99th', '95th', 'med'])
            mem_log.close()
            del mem_log

            # 4. network
            network_log = open(self.log_network, 'w+')
            writer = csv.writer(network_log)
            titles = []
            titles.append('time')
            for i in range(len(infrastructure.nodes)):
                titles.append('in'+str(i))
                titles.append('out'+str(i))
            writer.writerow(titles)
            network_log.close()
            del network_log
            del titles
        
        # init jobs log
        job_log = open(self.log_job, 'w+')
        writer = csv.writer(job_log)
        if self.flags.schedule == 'gpu-demands':
            writer.writerow(['time', '1-GPU', '2-GPU', '4-GPU', '8-GPU', '12-GPU', '16-GPU', '24-GPU', '32-GPU'])
        else:
            if self.flags.scheme == 'count':
                writer.writerow(['time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed_time', 'JCT', 'duration', 'pending_time', 'preempt', 'resume', 'promote'])
            else:
                writer.writerow(['time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed_time', 'JCT', 'duration', 'pending_time', 'preempt', 'promote'])
        job_log.close()

        assert os.path.exists(self.log_cluster)
    
    def step_cluster(self, loginfo, delta):
        with open(self.log_cluster, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow({
                'delta': delta,
                'num_idle_nodes': loginfo.idle_ns,
                'num_busy_nodes': loginfo.busy_ns,
                'num_busy_gpus': loginfo.busy_gs,
                'num_idle_gpus': loginfo.idle_gs,
                'avg_gpu_utilization': loginfo.avg_g_utils,
                'avg_gpu_memory_allocated': loginfo.avg_g_mem,
                'avg_pending_time': loginfo.avg_pending,
                'num_running_jobs': loginfo.num_running_jobs,
                'num_queuing_jobs': loginfo.num_queuing_jobs,
                'num_finish_jobs': loginfo.num_finish_jobs
            })
