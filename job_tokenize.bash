#!/bin/bash

#PBS -l select=1:ncpus=8:mem=50gb
#PBS -l walltime=24:00:00
#PBS -A "stupsprojmabre"

lang=deu
langcode="de-DE"
corpus=res_training/$lang/news_tatoeba_training.txt

module load Java/1.8.0_151

cd ~/projektarbeit/languagetool-neural-network
./gradlew -debug tokenizeFile -PlanguageCode="$langcode" -PsentencesFile="$corpus"
