GPU cluster simulator for distributed deep learning training
===
**NOTE**: Currently there are a couple of assumptions:
1. Homogenous cluster set up
2. model gradients transfer is the same as the model size saved in ckpts (model_factory)
3. Parameter Server / Worker frameworks (All-reduce not yet implemented)
4. **Synchronize SGD**

**Execution**
Before the exection, what's needed?
    <!-- 1. Infrastructure details
    Define the hierarchy and resource capacity of the infrastructure in ``cluster_spec.csv``. For example, we have a cluster with 4 racks (switches). Under each rack (switch), there are 32 nodes. And each node has 128 CPU cores, 256 GB memory, and 8 GPUs. Then ``cluster_spec.csv`` will look like this:
        ```csv
        num_switch,num_node_p_switch,num_gpu_p_node,num_cpu_p_node,mem_p_node
        4,32,8,128,256
        ``` -->
1. Job trace
The job trace to simulate. For each job, the simulator needs the following information:
    * ``job_id``: for tracking
    * ``num_gpu``: gpu requirement
    * ``submit_time``: when the job is submitted. The simulator is event-based and discrete-time. Therefore, the time value starts from ``0``, and in second-scale.
    * ``iterations``: the number of iterations to training. Used by Network costs calculation when in data parallel jobs.
    * ``model_name``: what's the model in that job. This is used to estimate GPU memory usage, and network costs.
    * ``duration``: how long this job will run. This information is used to generate job completion event by the simulator.
    * ``interval``: job submission interval from this job to the next job
    

2. How to run the simulator?
    A simple example of the execution commend should be:
    ```
    python execute.py
    ```
    Inside the execute file The following options are necessary:
    * ``--cluster_spec``: infrastructure spec file
    * ``--trace_file``: job trace
    * ``--scheme``: **placement scheme**
    * ``--schedule``: **scheduler**

    Optional inputs:
    * ``--print``: print debug information
    * ``--log_path``: the output path of the log (cluster, job). The default will be ``time-stamp`` folder under current path

3. What are the placement and scheduling algorithms provided?
    *Placement*: 
    * ``yarn``: get GPUs from the same server nodes under the same switch

    *Scheduling*
    * ``fifo``
    * ``sjf``: Smallest-job-first, in terms of GPU requirement
    * **TODO BELOW**
    * ``lpjf``: longest pending job first
    * ``shorest``: shorestest remaining time job first
    * ``shorest-gpu``: shortest-remaining-gputime job first
    * ``dlas``: discretized LAS (just time-based)
        In ``jobs.py``,  you need to specify ``num_queue`` and ``queue_limit`` for ``MLFQ`` (also for ``dlas-gpu``, and ``gittins``)
        ```python
        # Example1: there are two queues, and the threshold for Q1 is 3600 seconds
        self.queue_limit = [3600]

        # Example2: there are four queues, and the threshold for queues is 3600, 7200, 18000 seconds
        self.queue_limit = [3600, 7200, 18000]
        ```
    * ``dlas-gpu``: discretized LAS (gpu-time-based)
    * ``gittins``: discretized Gittins Index (gpu-time-based)


4. What's the output?
    Based on the ``--log_path``, all the output files are in that folder (e.g., ``result-20190210-12-20-37`` including:
    1. ``cluster.csv``: cluster-level resource utilization info at each event point
    2. ``jobs.csv``: the job execution information
    <!-- 3. ``cpu.csv``, ``gpu.csv``, ``memory.csv``, ``network.csv``: those are the utilization details of each resource unit at event points. However, those logs are not accurate under some combinations of placement and scheduler. When ``count`` is chosen, those files are not generated. -->

    The output logs are defined in ``log.py``; You can modify that file to adjust the output information.


Others
--------------
g.yeung1@lancaster.ac.uk