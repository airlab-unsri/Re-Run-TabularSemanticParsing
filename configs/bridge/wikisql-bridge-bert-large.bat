@echo off

set data_dir=data\wikisql1.1\
set db_dir=data\wikisql1.1\
set dataset_name=wikisql
set model=bridge
set question_split=True
set query_split=False
set question_only=True
set normalize_variables=False
set denormalize_sql=True
set omit_from_clause=True
set no_join_condition=False
set table_shuffling=True
set use_graph_encoding=False
set use_typed_field_markers=False
set use_lstm_encoder=True
set use_meta_data_encoding=True
set use_picklist=True
set no_anchor_text=False
set anchor_text_match_threshold=0.85
set top_k_picklist_matches=2
set atomic_value_copy=False
set process_sql_in_execution_order=False
set sql_consistency_check=False
set share_vocab=False
set sample_ground_truth=False
set save_nn_weights_for_visualizations=False
set vocab_min_freq=0
set text_vocab_min_freq=0
set program_vocab_min_freq=0
set max_in_seq_len=512
set max_out_seq_len=60

set num_steps=30000
set curriculum_interval=0
set num_peek_steps=400
set num_accumulation_steps=3
set save_best_model_only=True
set train_batch_size=16
set dev_batch_size=24
set encoder_input_dim=1024
set encoder_hidden_dim=512
set decoder_input_dim=512
set num_rnn_layers=1
set num_const_attn_layers=0

set use_oracle_tables=False
set num_random_tables_added=0

set use_additive_features=False

set schema_augmentation_factor=1
set random_field_order=False
set data_augmentation_factor=1
set augment_with_wikisql=False
set num_values_per_field=0
set pretrained_transformer=bert-large-uncased
set fix_pretrained_transformer_parameters=False
set bert_finetune_rate=0.00005
set learning_rate=0.0003
set learning_rate_scheduler=inverse-square
set trans_learning_rate_scheduler=inverse-square
set warmup_init_lr=0.0003
set warmup_init_ft_lr=0
set num_warmup_steps=3000
set emb_dropout_rate=0.3
set pretrained_lm_dropout_rate=0
set rnn_layer_dropout_rate=0.1
set rnn_weight_dropout_rate=0
set cross_attn_dropout_rate=0
set cross_attn_num_heads=8
set res_input_dropout_rate=0.2
set res_layer_dropout_rate=0
set ff_input_dropout_rate=0.4
set ff_hidden_dropout_rate=0.0

set grad_norm=0.3
set decoding_algorithm=beam-search
set beam_size=64
set bs_alpha=1.0

set data_parallel=False

python3 -m src.experiments ^
    %dataset_name% ^
    --data_dir %data_dir% ^
    --db_dir %db_dir% ^
    --dataset_name %dataset_name% ^
    --question_split %question_split% ^
    --query_split %query_split% ^
    --question_only %question_only% ^
    --normalize_variables %normalize_variables% ^
    --denormalize_sql %denormalize_sql% ^
    --omit_from_clause %omit_from_clause% ^
    --no_join_condition %no_join_condition% ^
    --table_shuffling %table_shuffling% ^
    --use_graph_encoding %use_graph_encoding% ^
    --use_typed_field_markers %use_typed_field_markers% ^
    --use_lstm_encoder %use_lstm_encoder% ^
    --use_meta_data_encoding %use_meta_data_encoding% ^
    --use_picklist %use_picklist% ^
    --no_anchor_text %no_anchor_text% ^
    --anchor_text_match_threshold %anchor_text_match_threshold% ^
    --top_k_picklist_matches %top_k_picklist_matches% ^
    --atomic_value_copy %atomic_value_copy% ^
    --process_sql_in_execution_order %process_sql_in_execution_order% ^
    --sql_consistency_check %sql_consistency_check% ^
    --share_vocab %share_vocab% ^
    --sample_ground_truth %sample_ground_truth% ^
    --save_nn_weights_for_visualizations %save_nn_weights_for_visualizations% ^
    --vocab_min_freq %vocab_min_freq% ^
    --text_vocab_min_freq %text_vocab_min_freq% ^
    --program_vocab_min_freq %program_vocab_min_freq% ^
    --max_in_seq_len %max_in_seq_len% ^
    --max_out_seq_len %max_out_seq_len% ^
    --num_steps %num_steps% ^
    --curriculum_interval %curriculum_interval% ^
    --num_peek_steps %num_peek_steps% ^
    --num_accumulation_steps %num_accumulation_steps% ^
    --save_best_model_only %save_best_model_only% ^
    --train_batch_size %train_batch_size% ^
    --dev_batch_size %dev_batch_size% ^
    --encoder_input_dim %encoder_input_dim% ^
    --encoder_hidden_dim %encoder_hidden_dim% ^
    --decoder_input_dim %decoder_input_dim% ^
    --num_rnn_layers %num_rnn_layers% ^
    --num_const_attn_layers %num_const_attn_layers% ^
    --use_oracle_tables %use_oracle_tables% ^
    --num_random_tables_added %num_random_tables_added% ^
    --use_additive_features %use_additive_features% ^
    --schema_augmentation_factor %schema_augmentation_factor% ^
    --random_field_order %random_field_order% ^
    --data_augmentation_factor %data_augmentation_factor% ^
    --augment_with_wikisql %augment_with_wikisql% ^
    --num_values_per_field %num_values_per_field% ^
    --pretrained_transformer %pretrained_transformer% ^
    --fix_pretrained_transformer_parameters %fix_pretrained_transformer_parameters% ^
    --bert_finetune_rate %bert_finetune_rate% ^
    --learning_rate %learning_rate% ^
    --learning_rate_scheduler %learning_rate_scheduler% ^
    --trans_learning_rate_scheduler %trans_learning_rate_scheduler% ^
    --warmup_init_lr %warmup_init_lr% ^
    --warmup_init_ft_lr %warmup_init_ft_lr% ^
    --num_warmup_steps %num_warmup_steps% ^
    --emb_dropout_rate %emb_dropout_rate% ^
    --pretrained_lm_dropout_rate %pretrained_lm_dropout_rate% ^
    --rnn_layer_dropout_rate %rnn_layer_dropout_rate% ^
    --rnn_weight_dropout_rate %rnn_weight_dropout_rate% ^
    --cross_attn_dropout_rate %cross_attn_dropout_rate% ^
    --cross_attn_num_heads %cross_attn_num_heads% ^
    --res_input_dropout_rate %res_input_dropout_rate% ^
    --res_layer_dropout_rate %res_layer_dropout_rate% ^
    --ff_input_dropout_rate %ff_input_dropout_rate% ^
    --ff_hidden_dropout_rate %ff_hidden_dropout_rate% ^
    --grad_norm %grad_norm% ^
    --decoding_algorithm %decoding_algorithm% ^
    --beam_size %beam_size% ^
    --bs_alpha %bs_alpha% ^
    --data_parallel %data_parallel%
