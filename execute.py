from subprocess import Popen
import os
import time

def main():
    # cluster_spec = 'n4g4'
    # trace_file = '60'

    scheme = 'yarn'
    num_nodes_p_switch = 32
    num_switch = 8
    schedule = 'fifo'
    trace_file = 'cleaned_samples_month'
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
        '--schedule', schedule,
        '--enable_network_costs', 'False',
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
    exit()

if __name__ == "__main__":
    main()