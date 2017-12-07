#!/bin/bash

if [ ! $# -eq 5 ]; then
    echo Usage:
    echo $0 dictionary_path embedding_path training_data_file.py test_data_file.py output_path
    exit -1
fi

python3 src/main/python/nn_words.py "$1" "$2" "$3" "$4" "$5"
