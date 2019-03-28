import os
import csv
from core import util
from core import job

class JobQueueManager(object):
    """A job queue object 
    that host all the jobs instead of wacky list, or dictionaries"""
    def __init__(self, flags, file_path=None, num_queue=1): 
        self.flags = flags
        self.file_path = file_path
     
        self.num_queue = num_queue
        self.queues = []
        self.queues = [list() for i in range(self.num_queue)]
        self.queue_limit = [3250, 7200, 18000]
        self._setup()
        
        #TODO: whats to do with the workers and gittins
        # mem info in GB
        # self.worker_mem = 5
        # self.ps_mem = 6
        # self.p_w_mem = 0.1
        # self.gittins_delta = 3250
        # self.mean_duration = 800

    def parse_job_file(self):
        """from a csv convert to jobs"""
        print("going to start parsing jobs csv")
        if not os.path.exists(self.file_path):
            raise ValueError()

        fd = open(self.file_path, 'r')
        deli = ','
        if self.file_path.find('.csv') == (len(self.file_path) - 4):
            deli = ','
        elif self.file_path.find('.txt') == (len(self.file_path) - 4):
            deli = ' '

        reader = csv.DictReader(fd, delimiter=deli)
        ''' Add job from job trace file'''
        keys = reader.fieldnames
        util.print_fn(
            '--------------------------------- Read TF jobs from: %s ---------------------------------' 
            % os.path.basename(self.file_path))
        util.print_fn('    we get the following fields:\n        %s' % keys)
        job_idx = 0
        for row in reader:
            # Add job into JOBS
            self._add_to_job_queue(self.parse_job(row, job_idx))
            job_idx += 1

        assert job_idx == self._total_jobs()
        # print(lp.prepare_job_info(JOBS.job_list[0]))
        util.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
        fd.close()

    def parse_job(self, job_dict, idx):
        duration = job_dict['duration']
        model = job_dict['model_name']
        interval = job_dict['interval']
        num_gpus = job_dict['num_gpu']
        submit_time = job_dict['submit_time']
        iterations = job_dict['iterations']
        return job.Job(idx, model, duration, iterations, interval, gpu=num_gpus)

    def _can_add(self, queue_idx):
            return len(self.queues[queue_idx]) < self.queue_limit[queue_idx]

    def _total_jobs(self):
        num = 0
        for q in self.queues:
            num += len(q)
        return num

    def _add(self, queue_idx, job):
        if self._can_add(queue_idx):
            self.queues[queue_idx].append(job)
        else:
            raise ArithmeticError()

    def _add_to_job_queue(self, job, queue_idx=None):
        """Args:
            queue_idx: if specified, added to specific queue"""
        if queue_idx is not None:
            self._add(queue_idx, job)
        else:
            self._add(0, job)

    def _setup(self):
        self.parse_job_file()
