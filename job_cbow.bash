#!/bin/bash

#PBS -l select=1:ncpus=8:mem=50gb
#PBS -l walltime=24:00:00
#PBS -A "stupsprojmabre"

lang=deu
langcode="de-DE"
corpus=res_training/$lang/news_tatoeba_training.txt-tokens
outdir=res_training/$lang/cbow

mkdir $outdir

export LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID".cbow.log"

module load TensorFlow/1.1.0
module load Python/3.4.5
module load CUDA/7.5.18

cd ~/projektarbeit/languagetool-neural-network
echo `date` >> $LOGFILE
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
./gradlew pythonGateway 2>&1 >> $LOGFILE &
sleep 10
cd src/main/python
PYTHONPATH=$PYTHONPATH:. 
export PYTHONPATH
echo $PYTHONPATH
python3 embedding/cbow.py ../../../$corpus $langcode 50000 20000 ../../../$outdir 2>&1 >> $LOGFILE
killall java 2>&1 >> $LOGFILE
