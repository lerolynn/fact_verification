# Fact Verification

This is the code repository for fact verification


## Train
```
python run_hover.py --model_type bert --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --do_train --do_lower_case --per_gpu_train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 5.0 --evaluate_during_training --max_seq_length 200 --max_query_length 60 --gradient_accumulation_steps 2 --save_steps 60 --logging_steps 60 --overwrite_cache --data_dir ../data/ --train_file project_train_data.json --predict_file project_train_data.json --output_dir ./output/roberta_zero_shot

```