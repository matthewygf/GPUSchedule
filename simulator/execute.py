import subprocess
from subprocess import Popen
import os

# Just a small script to run the bash script from PyCharm

p = Popen("./sim_runs.sh", cwd=os.getcwd(), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
stdout, stderr = p.communicate()

print("[Output]")
print(stdout.decode('utf-8'))
print(stderr.decode('utf-8'))

