#!/bin/bash

if [ ! $# -eq 3 ]; then
    echo Usage example:
    echo $0 "training-corpus.txt" to too
    exit -1
fi

file_with_sentences=$1
subject1=$2
subject2=$3

validation_percent=20
file_shuffled="/tmp/`basename $file_with_sentences`_shuf"
tmp_file_subject1="/tmp/$subject1"
tmp_file_subject2="/tmp/$subject2"
tmp_file_dataset="/tmp/${subject1}_${subject2}"
training_file="/tmp/${subject1}_${subject2}_training"
validate_file="/tmp/${subject1}_${subject2}_validate"
tokens_file=${tmp_file_dataset}_tokens
subject1_regexp="[^a-z]$subject1[^a-z]"
subject2_regexp="[^a-z]$subject2[^a-z]"

function print_dataset_file_information {
    # $1: filename
    echo $1
    echo `wc -l < $1` lines
    echo `grep -c $subject1_regexp $1` $subject1
    echo `grep -c $subject2_regexp $1` $subject2
}

shuf $file_with_sentences > $file_shuffled

grep $subject1_regexp $file_shuffled > $tmp_file_subject1
grep $subject2_regexp $file_shuffled > $tmp_file_subject2
cat $tmp_file_subject1 $tmp_file_subject2 | sort -u | shuf > $tmp_file_dataset
dataset_lines=`wc -l < $tmp_file_dataset`
validation_set_lines=$((dataset_lines*validation_percent/100))
training_set_lines=$((dataset_lines-validation_set_lines))
head -n $training_set_lines $tmp_file_dataset > $training_file
tail -n $validation_set_lines $tmp_file_dataset > $validate_file

print_dataset_file_information $tmp_file_dataset
print_dataset_file_information $training_file
print_dataset_file_information $validate_file

echo generate $training_file.py, $validate_file.py
./gradlew createNGramDatabase -PlanguageCode="pt-PT" -PtrainingFile="$training_file" -PvalidationFile="$validate_file" -Ptoken1="$subject1" -Ptoken2="$subject2"
