#!/bin/bash
#SBATCH --job-name=train_stobert_nli_chaosnli
#SBATCH --account=Project_2001194
#SBATCH -o stobert_chaosnli_res.txt
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elaine.zosa@helsinki.fi

source /home/hande/Work/bert_jax/venv/bin/activate

BASE=/home/hande/Work/

python -m pdb $BASE/bert_jax/train_jax_stobert.py \
	--dataset mnli-m-chaosnli \
	--data_path /home/hande/Work/DATA/HND_NLI/ \
	--model_name_or_path bert-base-uncased \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--num_train_epochs 10 \
	--learning_rate 2e-5 \
	--output_dir $BASE/outputs/bert_jax \
    --seed 1506

