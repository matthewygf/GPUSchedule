from subprocess import Popen
import os
import time

def main():
    cluster_spec = 'n4g4'
    trace_file = '60'
    scheme = 'yarn'
    schedule = 'fifo'
    log_path = cluster_spec + "_job_" + trace_file + "/" + scheme + "_" + schedule
    python_ex = 'python.exe' if os.name == 'nt' else 'python3'
    cmd = [
        python_ex, 'run_sim.py',
        '--cluster_spec', cluster_spec+'.csv',
        '--scheme', scheme,
        '--trace_file', trace_file+'_job.csv',
        '--schedule', schedule,
        '--enable_network_costs', 'True',
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