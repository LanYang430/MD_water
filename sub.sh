#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=pimd_295K
#SBATCH -N 1 -n 5 -t 550:00:00 -c 1
#SBATCH --gres=gpu:4 -p gpu -w gpu002
#SBATCH --mem=64000mb -o out -e err

export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1

scrdir=/tmp

# clean folder
rm $scrdir/ipi_unix_dmff_*
rm $scrdir/ipi_unix_eann_*
echo "***** start time *****"
date

cd  $SLURM_SUBMIT_DIR
# run server
bash run_server.sh &
sleep 30

# check socket
ls -l $scrdir

# run client
iclient=1
while [ $iclient -le 4 ];do
    bash run_EANN.sh &
    export CUDA_VISIBLE_DEVICES=$((iclient+1))
    bash run_client_dmff.sh &
    iclient=$((iclient+1))
    sleep 1s
done

wait

echo "***** finish time *****"
date

sleep 1
 
