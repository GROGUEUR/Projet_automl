#!/bin/bash
#SBATCH --job-name=AutoML_Opti
#SBATCH --output=logs/opti_%j.out
#SBATCH --error=logs/opti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --partition=std   # Vérifie le nom (souvent std, compute, ou cpu_prod)

# === Configuration de l'environnement ===
module purge
module load python/3.9  # Ou anaconda3 selon dispo sur Skinner

# Activer venv (adapte le chemin si besoin)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "~/automl_env" ]; then
    source ~/automl_env/bin/activate
else
    echo "❌ Pas d'environnement virtuel trouvé !"
    exit 1
fi

# Création des dossiers
mkdir -p logs results

# === Lancement du Job ===
echo "Début du job sur le noeud : $(hostname)"
echo "Date : $(date)"

python scripts/optimize_hyperparameters.py \
    --data-path /info/corpus/ChallengeMachineLearning \
    --search-method random \
    --n-iter 50 \
    --cv 5 \
    --output results/opti_results_${SLURM_JOB_ID}.pkl

echo "✅ Fin du job"