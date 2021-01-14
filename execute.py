from subprocess import Popen
import os
import time

def do_once(scheme, schedule, num_queue):
    # cluster_spec = 'n4g4'
    # trace_file = '60'

    # scheme = 'yarn'
    num_nodes_p_switch = 32
    num_switch = 4
    # schedule = 'fifo'
    migrate = True
    trace_file = 'month'
    log_sub_dir = "nodes_p_s"+str(num_nodes_p_switch) + "_job_" + trace_file
    log_path = os.path.join(log_sub_dir, f"{scheme}_{schedule}")
    python_ex = 'python.exe' if os.name == 'nt' else 'python3'
    cmd = [
        python_ex, 'run_sim.py',
        '--num_node_p_switch', str(num_nodes_p_switch),
        '--num_switch', str(num_switch),
        # '--cluster_spec', cluster_spec+'.csv',
        '--scheme', scheme,
        '--trace_file', os.path.join("data",trace_file+'.csv'),
        '--num_queue', str(num_queue),
        '--schedule', schedule,
        '--enable_network_costs', 'False',
        '--enable_migration', str(migrate),
        '--log_path', log_path
    ]
    p = Popen(cmd)
    poll = None    
    pid = p.pid
    print("process pid %d: " % pid)
    try:
        while poll is None:
            time.sleep(5)
            poll = p.poll()
    except KeyboardInterrupt:
        p.kill()
        
    return

def main():
    # schemes = ['horus+','horus+','horus','gandiva', 'yarn']
    # queues = [4,5,1,1,1]
    # schedules = ['horus+','horus+','horus','gandiva','fifo']
    schemes = ['horus+', 'horus+', 'horus+']
    queues = [3, 4, 5]
    schedules = ['horus+', 'horus+', 'horus+']
    for scheme, schedule, num_q in zip(schemes, schedules, queues):
        for _ in range(0, 3):
            do_once(scheme, schedule, num_q)


if __name__ == "__main__":
    main()
