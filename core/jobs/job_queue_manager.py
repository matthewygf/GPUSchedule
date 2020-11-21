import os
import csv
from core import util
from core.jobs import job

class JobQueueManager(object):
    """
    A job queue object
    that host all the jobs instead of wacky lists, or dictionaries"""
    def __init__(self, flags, file_path=None):
        self.flags = flags
        self.file_path = file_path
        self.num_queue = flags.num_queue
        self.queues = [list() for i in range(self.num_queue)]
        self.queue_limit = [9999 for i in range(self.num_queue)]
        #TODO: whats to do with the workers and gittins
        # mem info in GB
        # self.worker_mem = 5
        # self.ps_mem = 6
        # self.p_w_mem = 0.1
        # self.gittins_delta = 3250
        # self.mean_duration = 800

    def parse_job_file(self):
        """from a csv convert to jobs"""
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
        for row in reader:
            self._add_to_job_queue(self.parse_job(row))

        util.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % self.total_jobs())
        fd.close()

    def parse_job(self, job_dict):
        idx = job_dict['job_id']
        duration = job_dict['duration']
        model = job_dict['model_name']
        interval = job_dict['interval']
        num_gpus = int(job_dict['num_gpu'])
        submit_time = job_dict['submit_time']
        iterations = float(job_dict['iterations'])
        return job.Job(idx, model, duration, iterations, interval, submit_time, gpu=num_gpus)


    def _can_add(self, queue_idx):
            return len(self.queues[queue_idx]) < self.queue_limit[queue_idx]

    def total_jobs(self, delta_time=-1):
        num = 0
        for q in self.queues:
            num += len(q)
        # for q in self.queues:
        #     if delta_time > 0:
        #         for j in q:
        #             if j.submit_time >= delta_time:
        #                 num += 1
        #     else:
        #         num += len(q)
        return num

    def _add(self, queue_idx, new_job):
        if self._can_add(queue_idx):
            self.queues[queue_idx].append(new_job)
        else:
            raise ArithmeticError()

    def _add_to_job_queue(self, new_job, queue_idx=None):
        """Args:
            queue_idx: if specified, added to specific queue"""
        if queue_idx is not None:
            self._add(queue_idx, new_job)
        else:
            self._add(0, new_job)


    def pop(self, queue_idx=0, job_in_queue=0):
        return self.queues[queue_idx].pop(job_in_queue)

    def insert(self, job, queue_idx=0, job_in_queue=0):
        return self.queues[queue_idx].insert(job_in_queue, job)
