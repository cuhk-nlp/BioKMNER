#!/bin/sh

mkdir logs
mkdir results

# train; use POS
python -W ignore run_BioKMNER_pos.py --data_dir=./data/sample_data/ --bert_model=/path/to/biobert_dir/ --max_seq_length=150 --max_word_size=150 --do_train --train_batch_size=2  --num_train_epochs 3 --do_eval --warmup_proportion=0.1 --patient=10 --feature_flag=pos --learning_rate=2e-5

# train; use chunk
python -W ignore run_BioKMNER_chunk.py --data_dir=./data/sample_data --bert_model=/path/to/biobert_dir/ --max_seq_length=150 --max_word_size=150 --do_train --train_batch_size=2  --num_train_epochs 3 --do_eval --warmup_proportion=0.1 --patient=10 --feature_flag=chunk --learning_rate=2e-5

# train; use dep
python -W ignore run_BioKMNER_dep.py --data_dir=./data/sample_data --bert_model=/path/to/biobert_dir/ --max_seq_length=150 --max_word_size=150 --do_train --train_batch_size=2  --num_train_epochs 3 --do_eval --warmup_proportion=0.1 --patient=10 --feature_flag=dep --learning_rate=2e-5
