import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d_%Hh')

contents = '''#!/bin/bash
#SBATCH --output="nsm{0}_r{2}_J1L12.out"
#SBATCH --error="nsm{0}_r{2}_J1L12.err"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 14
#SBATCH --account=ctmc

#SBATCH --mail-type ALL
#SBATCH --mail-user jeanne.colbois@epfl.ch
#SBATCH --time 6-12:00:00

##working directory
SCRGD="/scratch/$USER/Runs_{1}_nsm{0}_r{2}_J1L12"
SCR="/scratch/$USER/Runs_{1}_nsm{0}_r{2}_J1L12/nsm{0}_r{2}_J1L12"

##home directory
HD="/home/$USER/2017-DipolarIsing/Correlations/Runs_04-02-19_GSCorrJ1"
HDR="/home/$USER/2017-DipolarIsing/Correlations/Runs_04-02-19_GSCorrJ1/ResJ1_L12_nsm{0}"

##create the scratch directories to work
mkdir $SCRGD
mkdir $SCR
##copy data from HD to SCR
cp $HD/*.py $SCR

##go to scratch and run code
cd $SCR

module load intel
module load python/3.6.5

time python3 RunBasis_GSCorrelations_3dot1dot6_C.py --L 12 --J1 1 --J2 0 --J3 0 --J4 0 --nst 10000 --nsm {0} --nips 20 --nb 500 --t_list 0.01 0.1 1 10 --nt_list 28 29 29 --log_tlist --nmaxiter 10 --stat_temps_lims 0.01 0.02 --energy --magnetisation --correlations --nthreads 14 --output J1_L12_nsm{0}_r{2}

##create the results directory in home
mkdir $HDR
##copy the results to the results directory
cp $SCR/*.pkl $HDR
##remove the files
cd $SCRGD
rm -r $SCR
'''

iteration = [0, 1]
nsm = 3000000
for r in iteration:
    with open("J1_L12_r{0}.run".format(r), 'w') as f:
         f.write(contents.format(nsm, now, r))
