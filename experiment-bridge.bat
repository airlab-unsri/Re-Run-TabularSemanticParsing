@echo off
setlocal EnableDelayedExpansion

REM Get the current directory
for /f "delims=" %%a in ('cd') do set "_INTERPOLATION_0=%%a"
set "PYTHONPATH=!_INTERPOLATION_0!;%PYTHONPATH%"

REM Set the environment and other variables
set "%~1exp=%~2"
set "%~1%~2gpu=%~3"
set "%~1%~2%~3ARGS=%*"

REM Prepare flags
set "question_split_flag="
if "%question_split%"=="True" set "question_split_flag=--question_split"

set "query_split_flag="
if "%query_split%"=="True" set "query_split_flag=--query_split"

set "question_only_flag="
if "%question_only%"=="True" set "question_only_flag=--question_only"

set "normalize_variables_flag="
if "%normalize_variables%"=="True" set "normalize_variables_flag=--normalize_variables"

set "share_vocab_flag="
if "%share_vocab%"=="True" set "share_vocab_flag=--share_vocab"

set "denormalize_sql_flag="
if "%denormalize_sql%"=="True" set "denormalize_sql_flag=--denormalize_sql"

set "omit_fram_clause_flag="
if "%omit_from_clause%"=="True" set "omit_fram_clause_flag=--omit_from_clause"

set "no_join_condition_flag="
if "%no_join_condition%"=="True" set "no_join_condition_flag=--no_join_condition"

set "table_shuffling_flag="
if "%table_shuffling%"=="True" set "table_shuffling_flag=--table_shuffling"

set "use_lstm_encoder_flag="
if "%use_lstm_encoder%"=="True" set "use_lstm_encoder_flag=--use_lstm_encoder"

set "use_meta_data_encoding_flag="
if "%use_meta_data_encoding%"=="True" set "use_meta_data_encoding_flag=--use_meta_data_encoding"

set "use_graph_encoding_flag="
if "%use_graph_encoding%"=="True" set "use_graph_encoding_flag=--use_graph_encoding"

set "sql_consistency_check_flag="
if "%sql_consistency_check%"=="True" set "sql_consistency_check_flag=--sql_consistency_check"

set "use_typed_field_markers_flag="
if "%use_typed_field_markers%"=="True" set "use_typed_field_markers_flag=--use_typed_field_markers"

set "read_picklist_flag="
if "%read_picklist%"=="True" set "read_picklist_flag=--read_picklist"

set "use_picklist_flag="
if "%use_picklist%"=="True" set "use_picklist_flag=--use_picklist"

set "no_anchor_text_flag="
if "%no_anchor_text%"=="True" set "no_anchor_text_flag=--no_anchor_text"

set "process_sql_in_execution_order_flag="
if "%process_sql_in_execution_order%"=="True" set "process_sql_in_execution_order_flag=--process_sql_in_execution_order"

set "sample_ground_truth_flag="
if "%sample_ground_truth%"=="True" set "sample_ground_truth_flag=--sample_ground_truth"

set "save_nn_weights_for_visualizations_flag="
if "%save_nn_weights_for_visualizations%"=="True" set "save_nn_weights_for_visualizations_flag=--save_nn_weights_for_visualizations"

set "fix_pretrained_transformer_parameters_flag="
if "%fix_pretrained_transformer_parameters%"=="True" set "fix_pretrained_transformer_parameters_flag=--fix_pretrained_transformer_parameters"

set "use_oracle_tables_flag="
if "%use_oracle_tables%"=="True" set "use_oracle_tables_flag=--use_oracle_tables"

set "atomic_value_copy_flag="
if "%atomic_value_copy%"=="True" set "atomic_value_copy_flag=--atomic_value_copy"

set "use_additive_features_flag="
if "%use_additive_features%"=="True" set "use_additive_features_flag=--use_additive_features"

set "data_parallel_flag="
if "%data_parallel%"=="True" set "data_parallel_flag=--data_parallel"

set "save_best_model_only_flag="
if "%save_best_model_only%"=="True" set "save_best_model_only_flag=--save_best_model_only"

set "augment_with_wikisql_flag="
if "%augment_with_wikisql%"=="True" set "augment_with_wikisql_flag=--augment_with_wikisql"

set "random_field_order_flag="
if "%random_field_order%"=="True" set "random_field_order_flag=--random_field_order"

REM Build the command
set "cmd=python -m src.experiments ^"
    %exp% ^"
    --data_dir !data_dir! ^"
    --db_dir !db_dir! ^"
    --dataset_name !dataset_name! ^"
    !question_split_flag! ^"
    !query_split_flag! ^"
    !question_only_flag! ^"
    !normalize_variables_flag! ^"
    !share_vocab_flag! ^"
    !denormalize_sql_flag! ^"
    !omit_fram_clause_flag! ^"
    !no_join_condition_flag! ^"
    !table_shuffling_flag! ^"
    !use_lstm_encoder_flag! ^"
    !use_meta_data_encoding_flag! ^"
    !use_graph_encoding_flag! ^"
    !sql_consistency_check_flag! ^"
    !use_typed_field_markers_flag! ^"
    !use_picklist_flag! ^"
    --anchor_text_match_threshold !anchor_text_match_threshold! ^"
    !no_anchor_text_flag! ^"
    !read_picklist_flag! ^"
    --top_k_picklist_matches !top_k_picklist_matches! ^"
    !process_sql_in_execution_order_flag! ^"
    !sample_ground_truth_flag! ^"
    !use_oracle_tables_flag! ^"
    --num_random_tables_added !num_random_tables_added! ^"
    !atomic_value_copy_flag! ^"
    !use_additive_features_flag! ^"
    !save_nn_weights_for_visualizations_flag! ^"
    !data_parallel_flag! ^"
    !save_best_model_only_flag! ^"
    --schema_augmentation_factor !schema_augmentation_factor! ^"
    !random_field_order_flag! ^"
    --data_augmentation_factor !data_augmentation_factor! ^"
    --vocab_min_freq !vocab_min_freq! ^"
    --text_vocab_min_freq !text_vocab_min_freq! ^"
    --program_vocab_min_freq !program_vocab_min_freq! ^"
    --num_values_per_field !num_values_per_field! ^"
    --max_in_seq_len !max_in_seq_len! ^"
    --max_out_seq_len !max_out_seq_len! ^"
    --model !model! ^"
    --num_steps !num_steps! ^"
    --curriculum_interval !curriculum_interval! ^"
    --num_peek_steps !num_peek_steps! ^"
    --num_accumulation_steps !num_accumulation_steps! ^"
    --train_batch_size !train_batch_size! ^"
    --dev_batch_size !dev_batch_size! ^"
    --encoder_input_dim !encoder_input_dim! ^"
    --encoder_hidden_dim !encoder_hidden_dim! ^"
    --decoder_input_dim !decoder_input_dim! ^"
    --num_rnn_layers !num_rnn_layers! ^"
    --num_const_attn_layers !num_const_attn_layers! ^"
    --emb_dropout_rate !emb_dropout_rate! ^"
    --pretrained_lm_dropout_rate !pretrained_lm_dropout_rate! ^"
    --rnn_layer_dropout_rate !rnn_layer_dropout_rate! ^"
    --rnn_weight_dropout_rate !rnn_weight_dropout_rate! ^"
    --cross_attn_dropout_rate !cross_attn_dropout_rate! ^"
    --cross_attn_num_heads !cross_attn_num_heads! ^"
    --res_input_dropout_rate !res_input_dropout_rate! ^"
    --res_layer_dropout_rate !res_layer_dropout_rate! ^"
    --ff_input_dropout_rate !ff_input_dropout_rate! ^"
    --ff_hidden_dropout_rate !ff_hidden_dropout_rate! ^"
    --pretrained_transformer !pretrained_transformer! ^"
    !fix_pretrained_transformer_parameters_flag! ^"
    --bert_finetune_rate !bert_finetune_rate! ^"
    --learning_rate !learning_rate! ^"
    --learning_rate_scheduler !learning_rate_scheduler! ^"
    --trans_learning_rate_scheduler !trans_learning_rate_scheduler! ^"
    --warmup_init_lr !warmup_init_lr! ^"
    --warmup_init_ft_lr !warmup_init_ft_lr! ^"
    --num_warmup_steps !num_warmup_steps! ^"
    --grad_norm !grad_norm! ^"
    --decoding_algorithm !decoding_algorithm! ^"
    --beam_size !beam_size! ^"
    --bs_alpha !bs_alpha! ^"
    --gpu !gpu! ^"
    !ARGS!"

REM Execute the command
echo Running: !cmd!
cmd /c !cmd!
