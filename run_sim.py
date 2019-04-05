import csv
import re
import sys
import types
import time
import math
import argparse
import copy
import os

from core import util
from core import flags
from core.jobs import job_queue_manager as jq
from core.jobs import jobs_manager as am
import log_manager as lm
from core.scheduling import schedule as sche
from infra import infrastructure as cluster

import logging
import datetime

# parse input arguments
flags.DEFINE_string('trace_file', 'tf_job.csv',
                    '''Provide TF job trace file (*.csv, *.txt).
                    *.csv file, use \',\' as delimiter; *.txt file, user \' \' as deliminter.
                    Default file is tf_job.csv ''')
flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
                    '''Simulation output folder, including cluster/node/gpu usage trace, pending job_queue info.
                    Default folder is result-[time]''')
flags.DEFINE_string('scheme', 'yarn',
                    '''
                    Job placement scheme:
                    0.count, just resource counting, without assignment (which gpu, which cpu)
                    1.yarn, ms yarn
                    2.random
                    3.crandom (consolidate + random)
                    4.greedy
                    5.balance
                    6.cbalance (consolidate + balance)
                    Default is yarn''')
flags.DEFINE_string('schedule', 'fifo',
                    '''
                    Job schedule scheme:
                    1.fifo
                    2.fjf, fit job first( in fifo order)
                    3.sjf, smallest job first
                    4.lpjf, longest pending job first
                    5.shortest, shortest-remaining-time job first
                    6.shortest-gpu, shortest-remaining-gputime job first
                    7.dlas, discretized las
                    8.dlas-gpu, dlas using gpu time
                    Default is fifo''')
flags.DEFINE_integer('num_switch', 1, '''Part of cluster spec: the number of switches in this cluster, default is 1''')

flags.DEFINE_integer('num_node_p_switch', 32,
                     '''Part of cluster spec: the number of nodes under a single switch, default is 32''')

flags.DEFINE_integer('bandwidth', 1250,
                     '''
                     Bandwidth per rack in MBps, default 10Gbps (GigaBit)
                     http://arxiv.org/abs/1805.07891 
                     modern is about 10Gbps 
                     ''')
flags.DEFINE_float('internode_latency', 0.015,
                     '''
                     internode, within the same rack, should there be latency
                     in seconds
                     ''')

flags.DEFINE_integer('num_gpu_p_node', 8,
                '''Part of cluster spec: the number of gpus on each node, default is 8''')
flags.DEFINE_integer('num_cpu_p_node', 64,
                '''Part of cluster spec: the number of cpus on each node, default is 64''')
flags.DEFINE_integer('mem_p_node', 256,
                '''Part of cluster spec: memory capacity on each node, default is 128''')
flags.DEFINE_string('cluster_spec', None,
                '''Part of cluster spec: cluster infra spec file,
                this file will overwrite the specs from num_switch, num_node_p_switch, and num_gpu_p_node
                Spec format:
                    num_switch,num_node_p_switch,num_gpu_p_node
                    int,int,int''')

flags.DEFINE_boolean('print', False,
                '''Enable print out information, default is False''')
flags.DEFINE_boolean('flush_stdout', True,
                '''Flush stdout, default is True''')
flags.DEFINE_version('0.1')


FLAGS = flags.FLAGS

'''
Gandiva scheduler assumption
'''
def gandiva_sim_jobs(gputime=False, solve_starvation=0):
    '''
    new jobs are added to the end of the ending queue
    but any fit job should be executed in fifo order
    '''
    cur_time = JOBS.job_events[0]['time']
    node_release = False
    time_diff = 0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs))> 0:
        # if len(JOBS.job_events) == 0:
        #     break
        CLUSTER.gandiva_node_set_adjust(cur_time, JOBS, LOG)
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        #update job progress for end_jobs
        node_release = CLUSTER.time_slicing_execute(cur_time, JOBS, LOG, time_diff)

        #get new start job
        event = util.search_dict_list(JOBS.job_events, 'time', cur_time)
        if event != None:
            #for new-start jobs, try to start
            for s_job in event['start_jobs']:
                ret = scheduler.try_get_job_res(CLUSTER, JOBS, s_job)
                if ret == False:
                    JOBS.move_to_pending(s_job)
                else:
                    s_job['start_time'] = cur_time
                    JOBS.running_jobs.append(s_job)
                    util.print_fn('----job[%d] starts' % s_job['job_idx'])

            #remove time_event
            JOBS.job_events.remove(event)

        if node_release:
            for p_job in JOBS.pending_jobs:
                ret = scheduler.try_get_job_res(CLUSTER, JOBS, p_job)
                if ret == True:
                    JOBS.remove_from_pending(p_job, cur_time)
                    p_job['start_time'] = cur_time
                    JOBS.running_jobs.append(p_job)
                    util.print_fn('----job[%d] starts from pending' % p_job['job_idx'])

        node_release = False

        if len(JOBS.job_events) <= 0:
            cur_time = cur_time + 10
            time_diff = 10
        else:
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']

            if (cur_time + 10) > next_e_time:
                time_diff = int(next_e_time - cur_time)
                cur_time = next_e_time
            else:
                cur_time = cur_time + 10
                time_diff = 10

        # cur_time = cur_time + 1


def smallest_first_sim_jobs(gputime=False):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served, until no resource
    '''
    end_events = list()
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        if end_time < start_time:
            event_time = end_time
            event = end_events[0]
        elif end_time > start_time:
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            event = JOBS.job_events.pop(0)
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            event = JOBS.job_events.pop(0)
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                #job completes
                CLUSTER.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)


        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                JOBS.move_to_runnable(s_job)
                s_job['remaining_time'] = s_job['duration']
                s_job['remaining_gputime'] = int(s_job['remaining_time'] * s_job['num_gpu'])
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = int(rjob['duration'] - rjob['total_executed_time'])
                if gputime:
                    rjob['remaining_gputime'] = int(rjob['remaining_time'] * rjob['num_gpu'])
            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['pending_time'] = int(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = event_time
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                pass
        #sort jobs with shortest first
        # if gputime:
        #     JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
        # else:
        #     JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))
        run_jobs = list()
        preempt_jobs = list()
        #scan / execute jobs one by one
        CLUSTER.empty_infra()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if 'placements' in rjob:
                    del rjob['placements'][:]
            ret = scheduler.try_get_job_res(CLUSTER, JOBS, rjob)
            if True == ret:
                # rjob['status'] = 'RUNNING'
                # if 0 == rjob['start_time'] and 0 != rjob['submit_time']:
                #     rjob['start_time'] = event_time
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = event_time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)

            else:
                # rjob['status'] = 'PENDING'
                if rjob['status'] == 'RUNNING':
                    preempt_jobs.append(rjob)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)

        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                end_time = int(event_time + rjob['remaining_time'])
                tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                if tmp_dict == None:
                    #not found, add the time into to job_events
                    tmp_dict = dict()
                    tmp_dict['time'] = end_time
                    tmp_dict['end_jobs'] = list()
                    tmp_dict['end_jobs'].append(rjob)
                    end_events.append(tmp_dict)
                else:
                    tmp_dict['end_jobs'].append(rjob)
        end_events.sort(key = lambda e:e.__getitem__('time'))


        LOG.checkpoint(JOBS, event_time)

def cal_shortest_expected_remaining(job_data, a):
    data = job_data['data']
    idx = next(x[0] for x in enumerate(data) if x[1] > a)

    if idx == (job_data['num'] - 1):
        return data[idx]

    num = job_data['num'] - 1 - idx
    return round(sum(data[idx: (job_data['num'] - 1)]) * 1.0 / num, 2)

def shortest_first_sim_jobs(gputime=False):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served, until no resource
    '''
    end_events = list()
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs)) > 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        # decide which is the next event: start or end  ?
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        if end_time < start_time:
            event_time = end_time
            event = end_events[0]
        elif end_time > start_time:
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            event = JOBS.job_events.pop(0)
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            event = JOBS.job_events.pop(0)
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        # for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                # job completes
                CLUSTER.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)


        # for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                # add into runnable list with pending status
                JOBS.move_to_runnable(s_job)
                if FLAGS.schedule == 'shortest-expected':
                    s_job['remaining_expected'] = cal_shortest_expected_remaining(JOBS.job_dist_data, 0)

                s_job['remaining_time'] = s_job['duration']
                s_job['remaining_gputime'] = int(s_job['remaining_time'] * s_job['num_gpu'])
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = int(rjob['duration'] - rjob['total_executed_time'])
                if FLAGS.schedule == 'shortest-expected':
                    s_job['remaining_expected'] = cal_shortest_expected_remaining(JOBS.job_dist_data, rjob['total_executed_time'])
                if gputime:
                    rjob['remaining_gputime'] = int(rjob['remaining_time'] * rjob['num_gpu'])
            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['pending_time'] = int(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = event_time
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                pass
        #sort jobs with shortest first
        if FLAGS.schedule == 'shortest-expected':
            JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_expected'))
        else:
            if gputime:
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
            else:
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

        run_jobs = list()
        preempt_jobs = list()
        #scan / execute jobs one by one
        CLUSTER.empty_infra()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if 'placements' in rjob:
                    del rjob['placements'][:]
            ret = scheduler.try_get_job_res(CLUSTER, JOBS, rjob)
            if True == ret:
                # rjob['status'] = 'RUNNING'
                # if 0 == rjob['start_time'] and 0 != rjob['submit_time']:
                #     rjob['start_time'] = event_time
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = event_time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)

            else:
                # rjob['status'] = 'PENDING'
                if rjob['status'] == 'RUNNING':
                    preempt_jobs.append(rjob)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)

        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                end_time = int(event_time + rjob['remaining_time'])
                tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                if tmp_dict == None:
                    #not found, add the time into to job_events
                    tmp_dict = dict()
                    tmp_dict['time'] = end_time
                    tmp_dict['end_jobs'] = list()
                    tmp_dict['end_jobs'].append(rjob)
                    end_events.append(tmp_dict)
                else:
                    tmp_dict['end_jobs'].append(rjob)
        end_events.sort(key = lambda e:e.__getitem__('time'))


        LOG.checkpoint(JOBS, event_time)

def multi_dlas_sim_jobs(gputime=False, solve_starvation=0):
    '''
    1. diffrent #GPU-job(s) should have different DLAS threshold
    2. the resource under each job category has its pattern, and should be determined dynamically

    TODO: each job category has separate dlas-scheduler queues;
    '''
    end_events = list()
    next_job_jump = sys.maxsize
    #interval for update cluster resource allocation
    reserve_interval = 600
    next_reserve = reserve_interval
    # while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
    while (len(JOBS.job_events) + int(sum(len(k.runnable_jobs) for k in JOBS.gpu_job.values())))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_event = None
        end_time = sys.maxszie
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        if event_time > next_job_jump:
            event_time = next_job_jump
            event = dict()


        if event_time > next_reserve:
            event_time = copy.copy(next_reserve)
            event = dict()
            next_reserve += reserve_interval
            JOBS.reserve_gpus(CLUSTER.num_gpu)

        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                '''
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)
                '''
                JOBS.end_job(e_job)
                LOG.job_complete(e_job, event_time)
                util.print_fn('---- job[%d] %dG is completed' % (e_job['job_idx'], e_job['num_gpu']))


        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job) #gpu_job operation in it
                s_job['q_id'] = 0 #any new start job should be in Q0
                JOBS.gpu_job[s_job['num_gpu']].queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)


        #update end events and sort, and get the most recent one
        del end_events[:]
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        next_job_jump = sys.maxsize

        for num_gpu, gjob in JOBS.gpu_job.items():
            if len(gjob.runnable_jobs) <= 0:
                continue
            print('~~~~~ working on %d-GPU %djobs~~~~~' % (num_gpu, len(gjob.runnable_jobs)), end=' ')

            for rjob in gjob.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    tmp = int(event_time - rjob['last_check_time'])
                    rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                    rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                    rjob['last_check_time'] = event_time

                    #check demotion
                    j_gt = 0
                    if gputime:
                        j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                    else:
                        j_gt = int(rjob['executed_time'])
                    cur_qid = rjob['q_id']
                    if cur_qid < int(gjob.num_queue - 1): #not for the last queue
                        if j_gt >= gjob.queue_limit[cur_qid]:
                            rjob['q_id'] = int(cur_qid + 1)
                            gjob.queues[rjob['q_id']].append(rjob)
                            gjob.queues[cur_qid].remove(rjob)
                            print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                elif 'PENDING' == rjob['status']:
                    tmp = int(event_time - rjob['last_check_time'])
                    rjob['last_check_time'] = event_time
                    rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                    if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                        rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                    #Q0 job no need to push_back, and must be a runned
                    if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                        if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                            rjob['executed_time'] = 0
                            rjob['last_pending_time'] = 0
                            gjob.queues[0].append(rjob)
                            gjob.queues[rjob['q_id']].remove(rjob)
                            rjob['q_id'] = 0
                            rjob['promote'] = int(rjob['promote'] + 1)

                elif 'END' == rjob['status']: # won't happen
                    gjob.runnable_jobs.remove(rjob)
                    # util.print_fn('---- job[%d] completed' % rjob['job_idx'])
                    pass



            ''' schedule jobs in each queue '''
            #empty gpu resource
            gjob.empty_gpu_alloc()
            # for "count" placement
            run_jobs = list()
            preempt_jobs = list()
            for queue in gjob.queues:
                for job in queue:
                    if gjob.alloc_free_gpus(job['num_gpu']):
                    # if gjob.free_gpu >= job['num_gpu']:
                        #should run
                        if job['status'] == 'PENDING':
                            #not running
                            run_jobs.append(job)
                        # CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu']) #done in alloc_free_gpus
                    else:
                        #should NOT run
                        if job['status'] == 'RUNNING':
                            #running
                            preempt_jobs.append(job)
                        continue

            util.print_fn('%d F' % gjob.free_gpu)

            for job in preempt_jobs:
                job['status'] = 'PENDING'
                # if job['q_id'] == 0:
                #     job['preempt'] = int(job['preempt'] + 1)
                job['preempt'] = int(job['preempt'] + 1)
            for job in run_jobs:
                job['status'] = 'RUNNING'
                job['resume'] = int(job['resume'] + 1)
                if job['start_time'] == sys.maxsize:
                    job['start_time'] = event_time


            #sort based on the job start time
            for queue in gjob.queues:
                pending_job = list()
                for job in queue:
                    # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
                    if job['status'] == 'PENDING':
                        pending_job.append(job)
                        # print(job['job_idx'])
                for job in pending_job:
                    queue.remove(job)
                queue.extend(pending_job)

            #update end events and sort, and get the most recent one
            for rjob in gjob.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    remaining_time = rjob['duration'] - rjob['total_executed_time']
                    end_time = int(event_time + remaining_time)
                    if end_time < min_end_time:
                        tmp_end_event['time'] = end_time
                        tmp_end_event['end_jobs'] = list()
                        tmp_end_event['end_jobs'].append(rjob)
                        min_end_time = end_time
                    elif min_end_time == end_time:
                        tmp_end_event['end_jobs'].append(rjob)

            # what's the closest queue_jump (demotion, and promotion) among all the jobs
            for rjob in gjob.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    qid = rjob['q_id']
                    if qid < int(gjob.num_queue - 1):
                        if gputime:
                            jump_time = int(math.ceil((gjob.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
                        else:
                            jump_time = int(gjob.queue_limit[qid] - rjob['executed_time'] + event_time)
                        if jump_time < next_job_jump:
                            next_job_jump = jump_time

                elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
                    if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
                        diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
                        if diff_time > 0:
                            jump_time = int(diff_time + event_time)
                            if jump_time < next_job_jump:
                                next_job_jump = jump_time

        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        LOG.checkpoint_multi_dlas_gpu(event_time)
        # when to update the reservation
        # JOBS.reserve_gpus(CLUSTER.num_gpu)

    JOBS.completion_check()

def dlas_sim_jobs(gputime=False, solve_starvation=0):
    '''
    Job's executed time -- priority queue
    Q0:[0, 30min)
    Q1:[30min,1h)
    Q2:[1h, 2h)
    Q3:[2h, 00)

    in each queue, jobs are scheduled in fit-first with FIFO
    how to avoid starvation?

    TODO:  2. add move_back for avoiding starvation
    '''
    end_events = list()
    next_job_jump = sys.maxsize
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        if event_time > next_job_jump:
            event_time = next_job_jump
            event = dict()

        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, event_time)
                # util.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                JOBS.queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                rjob['last_check_time'] = event_time

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                else:
                    j_gt = int(rjob['executed_time'])
                cur_qid = rjob['q_id']
                if cur_qid < int(JOBS.num_queue - 1): #not for the last queue
                    if j_gt >= JOBS.queue_limit[cur_qid]:
                        rjob['q_id'] = int(cur_qid + 1)
                        JOBS.queues[rjob['q_id']].append(rjob)
                        JOBS.queues[cur_qid].remove(rjob)
                        print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                if FLAGS.schedule == 'dlas-gpu-gittins':
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

                if FLAGS.schedule == 'dlas-gpu-gittins':
                    if gputime:
                        j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                    else:
                        j_gt = int(rjob['executed_time'])
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # util.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        CLUSTER.empty_infra()
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()

        # if FLAGS.schedule == 'dlas-gpu-gittins':
        #     q = JOBS.queues[0]
        #     q.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)

        for queue in JOBS.queues:
            if FLAGS.schedule == 'dlas-gpu-gittins':
                queue.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)
            for job in queue:
                if CLUSTER.free_gpu >= job['num_gpu']:
                    #should run
                    if job['status'] == 'PENDING':
                        #not running
                        run_jobs.append(job)
                    CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
                else:
                    #should NOT run
                    if job['status'] == 'RUNNING':
                        #running
                        preempt_jobs.append(job)
                    continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time


        #sort based on the job start time
        for queue in JOBS.queues:
            #job there are many students
            pending_job = list()
            for job in queue:
                # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
                if job['status'] == 'PENDING':
                    pending_job.append(job)
                    # print(job['job_idx'])
            for job in pending_job:
                queue.remove(job)
            queue.extend(pending_job)


        #for fit-job-first, move small running jobs in front of large pending jobs in each queue
        # for queue in JOBS.queues:
        #     num_j = len(queue)
        #     #find the first pending job
        #     for i in range(num_j):
        #         if queue[i]['status'] == 'PENDING':
        #             break
        #     first_pending_idx = i

        #     #picking running_after_pending_jobs
        #     run_after_pending = list()
        #     for j in range(first_pending_idx, num_j):
        #         if queue[j]['status'] == 'RUNNING':
        #             run_after_pending.append(queue[j])

        #     #reinsert all those jobs
        #     for job in run_after_pending:
        #         queue.remove(job)
        #     for job in run_after_pending:
        #         queue.insert(i, job)
        #         i = int(i + 1)


        # for queue in JOBS.queues:
        #     for job in queue:
        #         if 'RUNNING' == job['status']:
        #             if 'placements' in job:
        #                 del job['placements'][:]
        #             job['status'] = 'PENDING'
        #         ret = try_get_job_res(job)
        #         if True == ret:
        #             job['status'] = 'RUNNING'
        #             if 0 == job['start_time'] and 0 != job['submit_time']:
        #                 job['start_time'] = event_time
        #         else:
        #             job['status'] = 'PENDING'
        #             continue



        #update end events and sort, and get the most recent one
        del end_events[:]
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         remaining_time = rjob['duration'] - rjob['total_executed_time']
        #         end_time = int(event_time + remaining_time)
        #         tmp_dict = util.search_dict_list(end_events, 'time', end_time)
        #         if tmp_dict == None:
        #             #not found, add the time into to job_events
        #             tmp_dict = dict()
        #             tmp_dict['time'] = end_time
        #             tmp_dict['end_jobs'] = list()
        #             tmp_dict['end_jobs'].append(rjob)
        #             end_events.append(tmp_dict)
        #         else:
        #             tmp_dict['end_jobs'].append(rjob)
        # end_events.sort(key = lambda e:e.__getitem__('time'))
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['duration'] - rjob['total_executed_time']
                end_time = int(event_time + remaining_time)
                if end_time < min_end_time:
                    tmp_end_event['time'] = end_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_time
                elif min_end_time == end_time:
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        next_job_jump = sys.maxsize
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                qid = rjob['q_id']
                if qid < int(JOBS.num_queue - 1):
                    if gputime:
                        jump_time = int(math.ceil((JOBS.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
                    else:
                        jump_time = int(JOBS.queue_limit[qid] - rjob['executed_time'] + event_time)
                    if jump_time < next_job_jump:
                        next_job_jump = jump_time

            elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
                    diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
                    if diff_time > 0:
                        jump_time = int(diff_time + event_time)
                        if jump_time < next_job_jump:
                            next_job_jump = jump_time



        LOG.checkpoint(JOBS, event_time)

def get_gittins_index(a):
    job_info = JOBS.job_dist_data
    if a > job_info['data'][-2]:
        return 0
    idx = next(x[0] for x in enumerate(job_info['data']) if x[1] > a)
    return job_info['gittins'][idx]

def gittins_sim_jobs(job_dist_data, gputime=False, static_delta=True):
    '''
    gittins index
    '''
    solve_starvation = 0
    end_events = list()
    next_job_jump = sys.maxsize
    next_gittins_unit = copy.copy(JOBS.gittins_delta)

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        # if event_time > next_job_jump:
        #     event_time = next_job_jump
        #     event = dict()

        #check the next gittins_unit
        if event_time > next_gittins_unit:
            event_time = next_gittins_unit
            event = dict()

        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, event_time)
                # util.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                # JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                # JOBS.queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                rjob['last_check_time'] = event_time

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                else:
                    j_gt = int(rjob['executed_time'])
                # cur_qid = rjob['q_id']
                # if cur_qid < int(JOBS.num_queue - 1): #not for the last queue
                #     if j_gt >= JOBS.queue_limit[cur_qid]:
                #         rjob['q_id'] = int(cur_qid + 1)
                #         JOBS.queues[rjob['q_id']].append(rjob)
                #         JOBS.queues[cur_qid].remove(rjob)
                #         print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                # rjob['rank'] = cal_r_gittins_index(job_dist_data, j_gt)
                rjob['rank'] = get_gittins_index(j_gt)

            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

                j_gt = rjob['executed_time']
                # rjob['rank'] = cal_r_gittins_index(job_dist_data, j_gt)
                rjob['rank'] = get_gittins_index(j_gt)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # util.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('rank'))

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        CLUSTER.empty_infra()
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()
        # for queue in JOBS.queues:
        #     queue.sort(key = lambda e:e.__getitem__('rank'))
        #     for job in queue:
        #         if CLUSTER.free_gpu >= job['num_gpu']:
        #             #should run
        #             if job['status'] == 'PENDING':
        #                 #not running
        #                 run_jobs.append(job)
        #             CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
        #         else:
        #             #should NOT run
        #             if job['status'] == 'RUNNING':
        #                 #running
        #                 preempt_jobs.append(job)
        #             continue

        for job in JOBS.runnable_jobs:
            if CLUSTER.free_gpu >= job['num_gpu']:
                #should run
                if job['status'] == 'PENDING':
                    #not running
                    run_jobs.append(job)
                CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
            else:
                #should NOT run
                if job['status'] == 'RUNNING':
                    #running
                    preempt_jobs.append(job)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time


        # #sort based on the job start time
        # for queue in JOBS.queues:
        #     #job there are many students
        #     pending_job = list()
        #     for job in queue:
        #         # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
        #         if job['status'] == 'PENDING':
        #             pending_job.append(job)
        #             # print(job['job_idx'])
        #     for job in pending_job:
        #         queue.remove(job)
        #     queue.extend(pending_job)


        #update end events and sort, and get the most recent one
        del end_events[:]
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         remaining_time = rjob['duration'] - rjob['total_executed_time']
        #         end_time = int(event_time + remaining_time)
        #         tmp_dict = util.search_dict_list(end_events, 'time', end_time)
        #         if tmp_dict == None:
        #             #not found, add the time into to job_events
        #             tmp_dict = dict()
        #             tmp_dict['time'] = end_time
        #             tmp_dict['end_jobs'] = list()
        #             tmp_dict['end_jobs'].append(rjob)
        #             end_events.append(tmp_dict)
        #         else:
        #             tmp_dict['end_jobs'].append(rjob)
        # end_events.sort(key = lambda e:e.__getitem__('time'))
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['duration'] - rjob['total_executed_time']
                end_time = int(event_time + remaining_time)
                if end_time < min_end_time:
                    tmp_end_event['time'] = end_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_time
                elif min_end_time == end_time:
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        # next_job_jump = sys.maxsize
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         qid = rjob['q_id']
        #         if qid < int(JOBS.num_queue - 1):
        #             if gputime:
        #                 jump_time = int(math.ceil((JOBS.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
        #             else:
        #                 jump_time = int(JOBS.queue_limit[qid] - rjob['executed_time'] + event_time)
        #             if jump_time < next_job_jump:
        #                 next_job_jump = jump_time

        #     elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
        #         if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
        #             diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
        #             if diff_time > 0:
        #                 jump_time = int(diff_time + event_time)
        #                 if jump_time < next_job_jump:
        #                     next_job_jump = jump_time


        next_gittins_unit += event_time
        LOG.checkpoint(event_time)

def dlas_pack_sim_jobs(gputime=False, solve_starvation=0):
    '''
    Job's executed time -- priority queue
    Q0:[0, 30min)
    Q1:[30min,1h)
    Q2:[1h, 2h)
    Q3:[2h, 00)

    in each queue, jobs are scheduled in fit-first with FIFO
    how to avoid starvation?

    TODO:  2. add move_back for avoiding starvation
    '''
    end_events = list()
    next_job_jump = sys.maxsize
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        if event_time > next_job_jump:
            event_time = next_job_jump
            event = dict()

        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, event_time)
                # util.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                JOBS.queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                rjob['last_check_time'] = event_time

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                else:
                    j_gt = int(rjob['executed_time'])
                cur_qid = rjob['q_id']
                if cur_qid < int(JOBS.num_queue - 1): #not for the last queue
                    if j_gt >= JOBS.queue_limit[cur_qid]:
                        rjob['q_id'] = int(cur_qid + 1)
                        JOBS.queues[rjob['q_id']].append(rjob)
                        JOBS.queues[cur_qid].remove(rjob)
                        print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))


            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time'])
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # util.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        CLUSTER.empty_infra()
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()
        for queue in JOBS.queues:
            for job in queue:
                # if CLUSTER.free_gpu >= job['num_gpu']:
                if CLUSTER.free_gpu_util(job):
                    #should run
                    if job['status'] == 'PENDING':
                        #not running
                        run_jobs.append(job)

                    CLUSTER.dlas_pack_get_gpu_util(job)
                    # CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
                else:
                    #should NOT run
                    if job['status'] == 'RUNNING':
                        #running
                        preempt_jobs.append(job)
                    continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time


        #sort based on the job start time
        for queue in JOBS.queues:
            #job there are many students
            pending_job = list()
            for job in queue:
                # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
                if job['status'] == 'PENDING':
                    pending_job.append(job)
                    # print(job['job_idx'])
            for job in pending_job:
                queue.remove(job)
            queue.extend(pending_job)


        #for fit-job-first, move small running jobs in front of large pending jobs in each queue
        # for queue in JOBS.queues:
        #     num_j = len(queue)
        #     #find the first pending job
        #     for i in range(num_j):
        #         if queue[i]['status'] == 'PENDING':
        #             break
        #     first_pending_idx = i

        #     #picking running_after_pending_jobs
        #     run_after_pending = list()
        #     for j in range(first_pending_idx, num_j):
        #         if queue[j]['status'] == 'RUNNING':
        #             run_after_pending.append(queue[j])

        #     #reinsert all those jobs
        #     for job in run_after_pending:
        #         queue.remove(job)
        #     for job in run_after_pending:
        #         queue.insert(i, job)
        #         i = int(i + 1)


        # for queue in JOBS.queues:
        #     for job in queue:
        #         if 'RUNNING' == job['status']:
        #             if 'placements' in job:
        #                 del job['placements'][:]
        #             job['status'] = 'PENDING'
        #         ret = try_get_job_res(job)
        #         if True == ret:
        #             job['status'] = 'RUNNING'
        #             if 0 == job['start_time'] and 0 != job['submit_time']:
        #                 job['start_time'] = event_time
        #         else:
        #             job['status'] = 'PENDING'
        #             continue



        #update end events and sort, and get the most recent one
        del end_events[:]
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         remaining_time = rjob['duration'] - rjob['total_executed_time']
        #         end_time = int(event_time + remaining_time)
        #         tmp_dict = util.search_dict_list(end_events, 'time', end_time)
        #         if tmp_dict == None:
        #             #not found, add the time into to job_events
        #             tmp_dict = dict()
        #             tmp_dict['time'] = end_time
        #             tmp_dict['end_jobs'] = list()
        #             tmp_dict['end_jobs'].append(rjob)
        #             end_events.append(tmp_dict)
        #         else:
        #             tmp_dict['end_jobs'].append(rjob)
        # end_events.sort(key = lambda e:e.__getitem__('time'))
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['duration'] - rjob['total_executed_time']
                end_time = int(event_time + remaining_time)
                if end_time < min_end_time:
                    tmp_end_event['time'] = end_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_time
                elif min_end_time == end_time:
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        next_job_jump = sys.maxsize
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                qid = rjob['q_id']
                if qid < int(JOBS.num_queue - 1):
                    if gputime:
                        jump_time = int(math.ceil((JOBS.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
                    else:
                        jump_time = int(JOBS.queue_limit[qid] - rjob['executed_time'] + event_time)
                    if jump_time < next_job_jump:
                        next_job_jump = jump_time

            elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
                    diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
                    if diff_time > 0:
                        jump_time = int(diff_time + event_time)
                        if jump_time < next_job_jump:
                            next_job_jump = jump_time



        LOG.checkpoint(event_time)

def longest_pending_first_sim_jobs():
    '''
    !!!!!!  NOT USEFUL
    new job has the highest priority then pending jobs
    jobs are scheduled based on their pending time: longest first
    '''
    pass
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        event = JOBS.job_events[0]
        event_time = event['time']
        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            #job completes
            CLUSTER.release_job_res(e_job)
            # CLUSTER.release_gpus(e_job)
            LOG.job_complete(e_job, event_time)


        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            #add into pending list
            JOBS.move_to_pending(s_job)

        #sort pending jobs based on the num_gpu
        JOBS.update_pending_time(event_time)
        JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('pending_time'), reverse=True)
        # JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #for pending jobs, try to start
        for p_job in JOBS.pending_jobs:
            # ret = CLUSTER.alloc_gpus(p_job)
            ret = scheduler.try_get_job_res(CLUSTER, JOBS, p_job)
            if ret == True:
                #if job is migratable, add into migratable job list
                # JOBS.add_migratable(p_job)
                JOBS.remove_from_pending(p_job, event_time)
                JOBS.add_job_end_event(p_job)
                util.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                # JOBS.read_job_info(p_job['job_idx'])
            else:
                break



        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        LOG.checkpoint(event_time)

def sim_job_events():
    '''
    Simulate job start/end, and gpu allocation
    pick one event from sorted job_event list
    1.ending jobs
    2.check the pending job list, for potential job placements
    3.start jobs
    4.logging
    '''
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break
        event = JOBS.job_events[0]
        event_time = event['time']
        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            #job completes
            CLUSTER.release_job_res(e_job)
            # CLUSTER.release_gpus(e_job)
            LOG.job_complete(e_job, event_time)

        #for pending jobs, try to start
        for p_job in JOBS.pending_jobs:
            # ret = CLUSTER.alloc_gpus(p_job)
            ret = scheduler.try_get_job_res(CLUSTER, JOBS, p_job)
            if ret == True:
                #if job is migratable, add into migratable job list
                # JOBS.add_migratable(p_job)
                JOBS.remove_from_pending(p_job, event_time)
                JOBS.add_job_end_event(p_job)
                util.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                # JOBS.read_job_info(p_job['job_idx'])
            else:
                # pending_jobs are sorted, if one is not able to be placement, then the rest are not necessary to consider
                break

        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            ret = scheduler.try_get_job_res(CLUSTER, JOBS, s_job)
            # ret = CLUSTER.alloc_gpus(s_job)
            if ret == False:
                #allocation failed, add into pending jobs
                JOBS.move_to_pending(s_job)
                util.print_fn('----job[%d] move to pending' % s_job['job_idx'])
            else:
                #if job is migratable, add into migratable job list
                # JOBS.add_migratable(s_job)
                JOBS.add_job_end_event(s_job)
                util.print_fn('----job[%d] starts' % s_job['job_idx'])
                # JOBS.read_job_info(s_job['job_idx'])

        #sort pending jobs based on the num_gpu
        JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        LOG.checkpoint(event_time)

    pass

def sim_gpu_demands():
    '''
    Simulate job start/end, and gpu demands
    pick one event from sorted job_event list
    1.ending jobs
    2.check the pending job list, for potential job placements
    3.start jobs
    4.logging
    '''
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break
        event = JOBS.job_events[0]
        event_time = event['time']
        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            # CLUSTER.release_job_res(e_job)
            # LOG.job_complete(e_job, event_time)
            JOBS.delete_gpu_job(e_job)

        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            #if job is migratable, add into migratable job list
            # JOBS.add_migratable(s_job)
            s_job['end_time'] = s_job['submit_time'] + s_job['duration']
            JOBS.add_job_end_event(s_job)
            util.print_fn('----job[%d] starts' % s_job['job_idx'])
            # JOBS.read_job_info(s_job['job_idx'])
            JOBS.add_gpu_job(s_job)



        #sort pending jobs based on the num_gpu
        # JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        # LOG.checkpoint(event_time)
        LOG.checkpoint_gpu_demands(event_time)

def cal_r_gittins_index(job_data, a):
    '''
    a means attained-service to that job
    gittins_index = P/E
    r_gi = E/P
    '''
    ut_delta = JOBS.gittins_delta

    data = job_data['data']
    if a > (job_data['data'][-1] - 1):
        return 0.0
    else:
        idx = next(x[0] for x in enumerate(data) if x[1] > a)

    next_a = a + ut_delta
    if next_a > (job_data['data'][-1] - 1):
        idx_delta = job_data['num'] - 1
    else:
        idx_delta = next(x[0] for x in enumerate(data) if x[1] > next_a)
    # print(idx, idx_delta)

    p = round(((idx_delta - idx) * 1.0) / (job_data['num'] - idx), 5)

    e_sum = sum(data[idx : idx_delta]) + (ut_delta * (job_data['num'] - idx_delta))
    e = round(e_sum / (job_data['num'] - idx), 5)

    # rank of gittins index = 1/gi
    # r_gi = round(e / p, 4)
    r_gi = round(p * 1000000 / e, 4)

    # print(idx, idx_delta, p, e_sum, e, r_gi)
    return r_gi

def parse_job_dist():
    job_dist_file = os.path.join(os.getcwd(), 'yarn-gput1000.csv')
    fd = open(job_dist_file, 'r')
    reader = csv.DictReader(fd, delimiter = ',')
    durations = list()
    for row in reader:
        durations.append(int(row['duration']))
    fd.close()
    total_len = len(durations)
    durations.sort()
    print("  %s samples are learned" % total_len)

    job_dict = dict()
    job_dict['num'] = total_len
    job_dict['data'] = durations

    gi = list()
    for v in job_dict['data']:
        gi.append(cal_r_gittins_index(job_dict, int(v-1)))

    # print(gi)
    job_dict['data'].append(sys.maxsize)
    gi.append(0.0)
    job_dict['gittins'] = gi

    return job_dict

def main(log_manager):

    if FLAGS.schedule == 'multi-dlas-gpu':
        assert FLAGS.scheme != 'count', 'multi-dlas-gpu need to be without count'

    #init infrastructure
    infrastructure = cluster.Infrastructure(FLAGS)

    # init Logging
    log_manager.init(infrastructure)

    # NOTE: init scheduler and jobs
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    trace_path = os.path.join(project_dir, FLAGS.trace_file)
    jq_manager = jq.JobQueueManager(FLAGS, trace_path)

    jobs_manager = am.JobsManager(jq_manager)
    scheduler = sche.Scheduler(infrastructure, jobs_manager)
    scheduler.sort_job_trace()
    
    # NOTE: start simulation
    scheduler.start()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)  # pylint: disable=line-too-long
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_dir = os.path.join('log', FLAGS.log_path)
    output_dir = os.path.join(output_dir, execution_id)
    util.make_dir_if_not_exist(output_dir)
    log_file = os.path.join(output_dir, 'output.log')
    filehandler = logging.FileHandler(
        filename=log_file, mode='w')
    filehandler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(filehandler)
    log_manager = lm.LogManager(output_dir, FLAGS)
    main(log_manager)

    logging.getLogger().removeHandler(filehandler)
