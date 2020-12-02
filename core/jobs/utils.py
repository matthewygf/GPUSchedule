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

def clusterize(jobs, k, dist_fn=job_dist, max_iter=1000):
    """k-means"""
    # randomly init centroid, selected k jobs to be the centroid
    centroids = [ jobs[i] for i in np.random.randint(len(jobs), size=k)]
    new_assignment = [0] * len(jobs)
    old_assignment = [-1] * len(jobs)

    current_iter = 0
    # converge or max iterations
    while current_iter < max_iter and new_assignment != old_assignment:
      old_assignment = list(new_assignment)
      current_iter += 1
      logging.info("k-means: current iter %d , current centroids are jobs %s" % (current_iter, centroids))
      for j_idx in range(len(jobs)):
        distances_between_j_and_centroid = [dist_fn(jobs[j_idx], centroids[c_idx]) for c_idx in range(len(centroids))]
        # each job now has a new assignment to the centroid
        new_assignment[j_idx] = np.argmin(distances_between_j_and_centroid)
      
      for c_idx in range(len(centroids)):
        members_of_c = [ jobs[j_idx] for j_idx in range(len(jobs)) if new_assignment[j_idx] == c_idx]
        if len(members_of_c) > 0:
          centroids[c_idx] = np.mean(members_of_c, axis=0).astype(int)
        else:
          centroids[c_idx] = jobs[np.random.choice(len(jobs))]

    loss = 0
    for j_idx in range(len(jobs)):
      loss += dist_fn(jobs[j_idx], new_assignment[j_idx])
    
    return centroids, new_assignment, loss