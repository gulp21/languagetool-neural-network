#!/bin/bash

#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=06:00:00
#PBS -A "stupsprojmabre"

lang=eng
subject1=to
subject2=too

module load TensorFlow/1.1.0
module load Python/3.4.5
module load CUDA/7.5.18

cd ~/projektarbeit/lt/grammarchecker

training_file="/tmp/${subject1}_${subject2}_training.py"
validate_file="/tmp/${subject1}_${subject2}_validate.py"

output_path="res_training/$lang/${subject1}_${subject2}"
mkdir $output_path

export LOGFILE=$output_path/$PBS_JOBNAME"."$PBS_JOBID".log"

echo `date` create_training_files >> $LOGFILE
./gradlew createNGramDatabase -PlanguageCode="en-US" -PcorpusFile="res_training/$lang/${lang}_news_2015_3M-sentences-raw.txt" -Ptokens="$subject1 $subject2" >> $LOGFILE

echo `date` create_classifier >> $LOGFILE
python3 src/main/python/nn_words.py res_training/$lang/embedding/dictionary.txt res_training/$lang/embedding/final_embeddings.txt $training_file $validate_file $output_path >> $LOGFILE

echo `date` finished, $output_path >> $LOGFILE
