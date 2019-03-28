import csv

class LogManager(object):
    def __init__(self, log_path, flags):
        self.log_path = log_path
        self.flags = flags
        self.is_count = self.flags.scheme == 'count'
    
    def init(self):
        self.log_cluster = os.path.join(self.log_path, 'cluster.csv')
        self.log_job = os.path.join(self.log_path, 'job.csv')
        if not self.is_count:
            self.log_cpu = os.path.join(self.log_path, 'cpu.csv')
            self.log_gpu = os.path.join(self.log_path, 'gpu.csv')
            self.log_network = os.path.join(self.log_path, 'network.csv')
            self.log_mem = os.path.join(self.log_path, 'memory.csv')
        
        self._init_all_csv()

    def _init_all_csv(self, infrastructure):
        # init cluster log
        cluster_log = open(self.log_cluster, 'w+')
        writer = csv.writer(cluster_log)
        if self.flags.scheme == 'gandiva':
            writer.writerow(['time', 'idle_node', 'busy_node', 'full_node', 'fra_gpu', 'busy_gpu', 'pending_job', 'running_job', 'completed_job', 'len_g1', 'len_g2', 'len_g4', 'len_g8', 'len_g16', 'len_g32', 'len_g64'])
        else:
            writer.writerow(['time', 'idle_node', 'busy_node', 'full_node', 'idle_gpu', 'busy_gpu', 'pending_job', 'running_job', 'completed_job'])
        cluster_log.close()
        del cluster_log

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
            writer.writerow(['time']+['gpu'] + str(i) for i in range(infrastructure.get_total_gpus()))
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