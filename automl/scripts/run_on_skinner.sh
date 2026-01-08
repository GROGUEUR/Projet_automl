#!/bin/bash
#SBATCH --job-name=AutoML_Opti
#SBATCH --output=logs/opti_%j.out
#SBATCH --error=logs/opti_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50GB
#SBATCH --time=12:00:00


# Création des dossiers
mkdir -p logs results


# === Lancement du Job ===
echo "Début du job sur le noeud : $(hostname)"
echo "Date : $(date)"

python optimize_hyperparameters.py \
    --data-path /info/corpus/ChallengeMachineLearning/data_C \
    --search-method random \
    --n-iter 50 \
    --cv 5 \
    --output results/opti_results_${SLURM_JOB_ID}.pkl

echo "Fin du job"
