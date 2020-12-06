import logging
import os
import csv
from core import util
from core.jobs import job
from heapq import *
import numpy as np

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
        self.queue_credits = [0 for i in range(self.num_queue)]
        self.queues_is_pq = [True if self.flags.schedule.startswith("horus") else False for _ in range(self.num_queue) ]
        # TODO: whats to do with the workers and gittins
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
            if self.queues_is_pq[queue_idx]:
                heappush(self.queues[queue_idx], new_job)
            else:
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

    def get_next_job(self, queue_idx=0, job_in_queue=0):
        return self.queues[queue_idx][job_in_queue]

    def queue_idx_by_credit(self):
        # a [1,2,3,4]
        # b [1,2,3,4,5,6,7,8]
        # c [1,2,3,4,5,6,7,8,9,10]

        # timestep 1 - 2 : choose c
        # timestep 3: break ties with b,c
        # timestep 4: choose the opposite of timestep 3
        # timestep 10: break ties with a,b,c
        logging.info("queue credits: %s", str(self.queue_credits))
        q_idx = np.argmax(self.queue_credits)
        return q_idx
    
    def update_credits(self):
        for q in range(0, self.num_queue):
            logging.info(q)
            if len(self.queues[q]) > 0:
                self.queue_credits[q] = sum([j.pending_time for j in self.queues[q]]) + len(self.queues[q])
            else:
                self.queue_credits[q] = 0

    def pop(self, queue_idx=0, job_in_queue=0):
        if self.queues_is_pq[queue_idx]:
            if len(self.queues[queue_idx]) > 0:
                return heappop(self.queues[queue_idx])
            raise ValueError("Length of queue %d is %d" % (queue_idx, len(self.queues[queue_idx])))

        return self.queues[queue_idx].pop(job_in_queue)
    
    def pop_all_queuing_jobs(self):
        """NOTE: this fn will remove all the jobs from the queues."""
        jobs = []
        for q_idx in range(self.num_queue):
            num = len(self.queues[q_idx])
            for _ in range(num):
                poped = self.pop(q_idx)
                jobs.append(poped)
        return jobs

    def insert(self, job, queue_idx=0, job_in_queue=0):
        self.queue_credits[queue_idx] = self.queue_credits[queue_idx] + 1

        if self.queues_is_pq[queue_idx]:
            return heappush(self.queues[queue_idx], job)
        

        return self.queues[queue_idx].insert(job_in_queue, job)
