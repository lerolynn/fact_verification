# Fact Verification

This is the code repository for fact verification

## Requirements
Python 3.7.3
torch 1.7.1
tqdm 4.49.0
transformers 2.6.0
stanza 1.1.1
nltk 3.5
scikit-learn 0.23.2
sense2vec

```
pip install torch==1.7.1 tqdm==4.49.0 transformers==2.6.0 stanza==1.1.1 nltk==3.5 scikit-learn==0.23.2 sense2vec
```

## Train

* `--model_type`: Model type is currently set to bert
  * Set to roberta, comment out line 57 of `run_hover.py`, uncomment line 58 of `run_hover.py` to change model type to roberta
  * Pretrained model also has to be changed to one using the roberta backbone.
* `--model_name_or_path` : set this parameter to the pretrained model to change models. This path can also be changed to the output directory to continue training from there

```
python run_hover.py --model_type bert --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --do_train --do_lower_case --per_gpu_train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 5.0 --evaluate_during_training --max_seq_length 200 --max_query_length 60 --gradient_accumulation_steps 2 --save_steps 60 --logging_steps 60 --overwrite_cache --data_dir ../data/ --train_file project_train_data.json --predict_file project_train_data.json --output_dir ./output/roberta_zero_shot

```

## Evaluate
```
python run_hover.py --model_type bert --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --do_eval --do_lower_case --per_gpu_train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 5.0 --evaluate_during_training --max_seq_length 200 --max_query_length 60 --gradient_accumulation_steps 2 --save_steps 60 --logging_steps 60 --overwrite_cache --data_dir ../data/ --train_file project_train_data.json --predict_file project_train_data.json --output_dir ./output/roberta_zero_shot

```
