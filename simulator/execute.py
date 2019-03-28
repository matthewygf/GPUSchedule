import subprocess
from subprocess import Popen
import os

# # Just a small script to run the bash script from PyCharm
# placement=("yarn") 
# #schedule=("fifo" "fjf" "sjf" "shortest" "shortest-gpu" "dlas" "dlas-gpu")
# #schedule=("dlas" "dlas-gpu" "dlas-gpu-100" "dlas-gpu-8" "dlas-gpu-4" "dlas-gpu-2" "dlas-gpu-1" "dlas-gpu-05")
# # schedule=("dlas-gpu")
# schedule=("gandiva")
# #schedule=("shortest-gpu")
# #schedule=("dlas" "dlas-gpu")
# # schedule=("dlas-gpu-05")
# # schedule=("dlas-gpu-1" "dlas-gpu-2" "dlas-gpu-4" "dlas-gpu-8" "dlas-gpu-10" "dlas-gpu-100" "dlas-gpu-1000")
# #schedule=("fifo")
# jobs=("60")
# setups=("n4g4")

def main():
    cluster_spec = 'n4g4'
    trace_file = '60'
    scheme = 'yarn'
    schedule = 'fifo'
    log_path = cluster_spec + "_job_" + trace_file + "/" +scheme + "_" + schedule

    cmd = [
        'python.exe', 'run_sim.py',
        '--cluster_spec', cluster_spec+'.csv',
        '--scheme', scheme,
        '--trace_file', trace_file+'_job.csv',
        '--schedule', schedule,
        '--log_path', log_path
    ]
    p = Popen(cmd, cwd=os.getcwd(), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))

if __name__ == "__main__":
    main()