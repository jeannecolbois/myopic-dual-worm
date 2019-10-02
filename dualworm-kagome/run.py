from glob import glob
import os
listruns = glob('*.run')

for run in listruns:
    os.system('sbatch ' + run)
