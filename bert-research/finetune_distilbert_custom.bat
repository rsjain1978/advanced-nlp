#rem clone transformers repo git clone https://github.com/huggingface/transformers.git
#create a folder data under /examples/distillation
#put dump file in this folder
# pip install -r requirements.txt
# pip install gitpython --upgrade

#python scripts/binarized_data.py --file_path data/dump.txt --tokenizer_type bert --tokenizer_name bert-base-uncased --dump_file data/binarized_text

#python scripts/token_counts.py --data_file data/binarized_text.bert-base-uncased.pickle --token_counts_dump data/token_counts.bert-base-uncased.pickle --vocab_size 30522

python train.py ^
    --student_type distilbert ^
    --student_config training_configs/distilbert-base-uncased.json ^
    --teacher_type bert ^
    --teacher_name bert-base-uncased ^
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm ^
    --freeze_pos_embs ^
    --dump_path serialization_dir/my_first_training ^
    --data_file data/binarized_text.bert-base-uncased.pickle ^
    --token_counts data/token_counts.bert-base-uncased.pickle ^
    --force