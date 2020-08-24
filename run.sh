#!/bin/sh

mkdir logs
mkdir results

python -W ignore run_NER_kv_softmax_all_dataset_concate_pos.py --data_dir=./data/sample_data/ --bert_model=biobert_pyt --max_seq_length=150 --max_word_size=150 --do_train --train_batch_size=2  --num_train_epochs 10 --do_eval --warmup_proportion=0.2 --patient=10 --feature_flag=pos --learning_rate=2e-5
