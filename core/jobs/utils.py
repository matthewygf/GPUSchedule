import numpy as np
import logging

def job_dist(x, y):
  score = abs(len(x.tasks) - len(y.tasks))
  score += abs(x.gpu_utilization_avg - y.gpu_utilization_avg)
  score += (x.gpu_per_worker - y.gpu_per_worker)**2
  score += (x.gpus - y.gpus)**2
  score += abs(x.gpu_utilization_max - y.gpu_utilization_max)
  score += abs(x.gpu_mem_avg - y.gpu_mem_avg)
  score += abs(x.gpu_mem_max - y.gpu_mem_max)
  return score

def transform_to_dist(job):
  score = len(job.tasks)
  score += job.gpu_utilization_avg
  score += job.gpu_per_worker
  score += job.gpus
  score += job.gpu_utilization_max
  score += job.gpu_mem_avg
  score += job.gpu_mem_max
  return score

def get_closest(jobs, score):
  c = None
  c_score = 999
  for j in jobs:
    dist = transform_to_dist(j)
    temp = abs(dist - score)
    if temp < c_score:
      c = j
      c_score = temp
  assert c is not None
  return c

def clusterize(jobs, k, dist_fn=job_dist, max_iter=1000):
    """k-means"""
    # randomly init centroid, selected k jobs to be the centroid
    centroids = [ jobs[i] for i in np.random.randint(len(jobs), size=k)]
    new_assignment = [None] * len(jobs)
    old_assignment = [None] * len(jobs)

    current_iter = 0
    # converge or max iterations
    while current_iter < max_iter and (new_assignment != old_assignment or current_iter == 0):
      old_assignment = list(new_assignment)
      current_iter += 1
      logging.info("k-means: current iter %d , current centroids are jobs %s" % (current_iter, centroids))
      for j_idx in range(len(jobs)):
        distances_between_j_and_centroid = [dist_fn(jobs[j_idx], centroids[c_idx]) for c_idx in range(len(centroids))]
        # each job now has a new assignment to the centroid
        new_assignment[j_idx] = np.argmin(distances_between_j_and_centroid)
      
      for c_idx in range(len(centroids)):
        members_of_c = [ jobs[j_idx] for j_idx in range(len(jobs)) if new_assignment[j_idx] == c_idx]
        members_of_c_dist = [transform_to_dist(m) for m in members_of_c]
        if len(members_of_c) > 0:
          # what is the mean of the our jobs ?
          mean_score = np.mean(members_of_c_dist, axis=0).astype(int)
          centroids[c_idx] = get_closest(members_of_c, mean_score)
        else:
          centroids[c_idx] = jobs[np.random.choice(len(jobs))]

    loss = 0
    for j_idx in range(len(jobs)):
      loss += dist_fn(jobs[j_idx], centroids[new_assignment[j_idx]])
    
    return centroids, new_assignment, loss