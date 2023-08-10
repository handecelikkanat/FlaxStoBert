#!/bin/bash
#SBATCH --job-name=flaxstobert_nli_chaosnli
#SBATCH --account=Project_2007780
#SBATCH -o logs/flaxstobert_%j.out
#SBATCH -e logs/flaxstobert_%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:4
#SBATCH --mail-type=ALL

BASE=/scratch/project_2007780/hande/
CODEDIR=$BASE/node-BNNs/bert_jax
DATADIR=$BASE/DATA/HND_NLI

source $CODEDIR/venv/bin/activate

#-m pdb
python $CODEDIR/train_jax_stobert.py \
	--dataset mnli-m-tiny-chaosnli \
	--data_path $DATADIR \
	--model_name_or_path bert-base-uncased \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--num_train_epochs 10 \
	--learning_rate 2e-5 \
	--output_dir $CODEDIR/outputs/bert_jax \
    --seed 7567 \
    --gpu_devices 0 1 \

