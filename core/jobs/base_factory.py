
class CompareAbleByUtilization(object):
  '''max heap by utilization - pop the small one keep max'''

  def __init__(self) -> None:
    print("comparing by utilization - Min Heap")

  def __lt__(self, other):
      if self.gpu_utilization_avg:
          return self.gpu_utilization_avg < other.gpu_utilization_avg

      return False

class CompareAbleByPendingTime(object):
  '''min heap by pending time - pop the large one keep min'''

  def __init__(self) -> None:
    print("comparing by pending time - Min Heap")

  def __lt__(self, other):
      return self.pending_time > other.pending_time

class BaseJobFactory(object):
  def __init__(self, flags):
    if flags.schedule.startswith("horus"):
      self.base = CompareAbleByUtilization
    else:
      self.base = CompareAbleByPendingTime
  
BASE_OBJ = None

