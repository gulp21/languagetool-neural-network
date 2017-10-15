#!/bin/bash

#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=06:00:00
#PBS -A "stupsprojmabre"

lang=eng
tokensFile=res_training/$lang/${lang}_news_2015_3M-sentences-raw.txt_small_tokens

export LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID".embedding.log"

module load TensorFlow/1.1.0
module load Python/3.4.5
module load CUDA/7.5.18

cd ~/projektarbeit/lt/grammarchecker
echo `date` >> $LOGFILE
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
cd src/main/python/embedding
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=1
cd ~/projektarbeit/lt/grammarchecker
python3 src/main/python/embedding/word2vec.py --train_data $tokensFile --eval_data res_training/$lang/question-words.txt --save_path res_training/$lang/embedding --epochs_to_train 10  >> $LOGFILE
