@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Set PYTHONPATH to include the current working directory
SET "PYTHONPATH=%cd%;%PYTHONPATH%"

:: Load the environment variables from the provided file
CALL %1

SET "exp=%2"
SET "gpu=%3"

:: Parse additional arguments
SET "ARGS=%*"
SET "ARGS=%ARGS:~4%"

:: Initialize flags
SET "question_split_flag="
IF /I "%question_split%"=="True" SET "question_split_flag=--question_split"

SET "query_split_flag="
IF /I "%query_split%"=="True" SET "query_split_flag=--query_split"

SET "question_only_flag="
IF /I "%question_only%"=="True" SET "question_only_flag=--question_only"

SET "normalize_variables_flag="
IF /I "%normalize_variables%"=="True" SET "normalize_variables_flag=--normalize_variables"

SET "share_vocab_flag="
IF /I "%share_vocab%"=="True" SET "share_vocab_flag=--share_vocab"

SET "denormalize_sql_flag="
IF /I "%denormalize_sql%"=="True" SET "denormalize_sql_flag=--denormalize_sql"

SET "omit_from_clause_flag="
IF /I "%omit_from_clause%"=="True" SET "omit_from_clause_flag=--omit_from_clause"

SET "no_join_condition_flag="
IF /I "%no_join_condition%"=="True" SET "no_join_condition_flag=--no_join_condition"

SET "table_shuffling_flag="
IF /I "%table_shuffling%"=="True" SET "table_shuffling_flag=--table_shuffling"

SET "use_lstm_encoder_flag="
IF /I "%use_lstm_encoder%"=="True" SET "use_lstm_encoder_flag=--use_lstm_encoder"

SET "use_meta_data_encoding_flag="
IF /I "%use_meta_data_encoding%"=="True" SET "use_meta_data_encoding_flag=--use_meta_data_encoding"

SET "use_graph_encoding_flag="
IF /I "%use_graph_encoding%"=="True" SET "use_graph_encoding_flag=--use_graph_encoding"

SET "sql_consistency_check_flag="
IF /I "%sql_consistency_check%"=="True" SET "sql_consistency_check_flag=--sql_consistency_check"

SET "use_typed_field_markers_flag="
IF /I "%use_typed_field_markers%"=="True" SET "use_typed_field_markers_flag=--use_typed_field_markers"

SET "read_picklist_flag="
IF /I "%read_picklist%"=="True" SET "read_picklist_flag=--read_picklist"

SET "use_picklist_flag="
IF /I "%use_picklist%"=="True" SET "use_picklist_flag=--use_picklist"

SET "no_anchor_text_flag="
IF /I "%no_anchor_text%"=="True" SET "no_anchor_text_flag=--no_anchor_text"

SET "process_sql_in_execution_order_flag="
IF /I "%process_sql_in_execution_order%"=="True" SET "process_sql_in_execution_order_flag=--process_sql_in_execution_order"

SET "sample_ground_truth_flag="
IF /I "%sample_ground_truth%"=="True" SET "sample_ground_truth_flag=--sample_ground_truth"

SET "save_nn_weights_for_visualizations_flag="
IF /I "%save_nn_weights_for_visualizations%"=="True" SET "save_nn_weights_for_visualizations_flag=--save_nn_weights_for_visualizations"

SET "fix_pretrained_transformer_parameters_flag="
IF /I "%fix_pretrained_transformer_parameters%"=="True" SET "fix_pretrained_transformer_parameters_flag=--fix_pretrained_transformer_parameters"

SET "use_oracle_tables_flag="
IF /I "%use_oracle_tables%"=="True" SET "use_oracle_tables_flag=--use_oracle_tables"

SET "atomic_value_copy_flag="
IF /I "%atomic_value_copy%"=="True" SET "atomic_value_copy_flag=--atomic_value_copy"

SET "use_additive_features_flag="
IF /I "%use_additive_features%"=="True" SET "use_additive_features_flag=--use_additive_features"

SET "data_parallel_flag="
IF /I "%data_parallel%"=="True" SET "data_parallel_flag=--data_parallel"

SET "save_best_model_only_flag="
IF /I "%save_best_model_only%"=="True" SET "save_best_model_only_flag=--save_best_model_only"

SET "augment_with_wikisql_flag="
IF /I "%augment_with_wikisql%"=="True" SET "augment_with_wikisql_flag=--augment_with_wikisql"

SET "random_field_order_flag="
IF /I "%random_field_order%"=="True" SET "random_field_order_flag=--random_field_order"

:: Construct the command
SET "cmd=python -m src.experiments %exp% --data_dir %data_dir% --db_dir %db_dir% --dataset_name %dataset_name% ^
    !question_split_flag! !query_split_flag! !question_only_flag! !normalize_variables_flag! !share_vocab_flag! !denormalize_sql_flag! ^
    !omit_from_clause_flag! !no_join_condition_flag! !table_shuffling_flag! !use_lstm_encoder_flag! !use_meta_data_encoding_flag! ^
    !use_graph_encoding_flag! !sql_consistency_check_flag! !use_typed_field_markers_flag! --anchor_text_match_threshold %anchor_text_match_threshold% ^
    !no_anchor_text_flag! !read_picklist_flag! --top_k_picklist_matches %top_k_picklist_matches% !process_sql_in_execution_order_flag! ^
    !sample_ground_truth_flag! !use_oracle_tables_flag! --num_random_tables_added %num_random_tables_added% !atomic_value_copy_flag! ^
    !use_additive_features_flag! !save_nn_weights_for_visualizations_flag! !data_parallel_flag! !save_best_model_only_flag! ^
    --schema_augmentation_factor %schema_augmentation_factor% !random_field_order_flag! --data_augmentation_factor %data_augmentation_factor% ^
    !augment_with_wikisql_flag! --vocab_min_freq %vocab_min_freq% --text_vocab_min_freq %text_vocab_min_freq% ^
    --program_vocab_min_freq %program_vocab_min_freq% --num_values_per_field %num_values_per_field% --max_in_seq_len %max_in_seq_len% ^
    --max_out_seq_len %max_out_seq_len% --model %model% --num_steps %num_steps% --curriculum_interval %curriculum_interval% ^
    --num_peek_steps %num_peek_steps% --num_accumulation_steps %num_accumulation_steps% --train_batch_size %train_batch_size% ^
    --dev_batch_size %dev_batch_size% --encoder_input_dim %encoder_input_dim% --encoder_hidden_dim %encoder_hidden_dim% ^
    --decoder_input_dim %decoder_input_dim% --num_rnn_layers %num_rnn_layers% --num_const_attn_layers %num_const_attn_layers% ^
    --emb_dropout_rate %emb_dropout_rate% --pretrained_lm_dropout_rate %pretrained_lm_dropout_rate% --rnn_layer_dropout_rate %rnn_layer_dropout_rate% ^
    --rnn_weight_dropout_rate %rnn_weight_dropout_rate% --cross_attn_dropout_rate %cross_attn_dropout_rate% --cross_attn_num_heads %cross_attn_num_heads% ^
    --res_input_dropout_rate %res_input_dropout_rate% --res_layer_dropout_rate %res_layer_dropout_rate% --ff_input_dropout_rate %ff_input_dropout_rate% ^
    --ff_hidden_dropout_rate %ff_hidden_dropout_rate% --pretrained_transformer %pretrained_transformer% ^
    !fix_pretrained_transformer_parameters_flag! --bert_finetune_rate %bert_finetune_rate% --learning_rate %learning_rate% ^
    --learning_rate_scheduler %learning_rate_scheduler% --trans_learning_rate_scheduler %trans_learning_rate_scheduler% ^
    --warmup_init_lr %warmup_init_lr% --warmup_init_ft_lr %warmup_init_ft_lr% --num_warmup_steps %num_warmup_steps% --grad_norm %grad_norm% ^
    --decoding_algorithm %decoding_algorithm% --beam_size %beam_size% --bs_alpha %bs_alpha% --gpu %gpu% %ARGS%"

:: Display the command that will be run
echo Running command: %cmd%

:: Execute the command
%cmd%

ENDLOCAL
