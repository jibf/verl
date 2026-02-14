# python examples/data_preprocess/car_crash.py \
#  --annotation_json_pattern /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/annotations_failed_1times.json \
#  --video_base_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/tracking_visualization/ \
#  --rule_base_crash_detection_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/ \
#  --local_save_dir /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/

# python examples/data_preprocess/car_crash.py \
#  --annotation_json_pattern /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/annotations_failed_1times.json \
#  --video_base_path /shared/jibf/data/CarCrashDataset/tracking_visualization/ \
#  --rule_base_crash_detection_path /shared/jibf/data/CarCrashDataset/rule_base_crash_results/ \
#  --local_save_dir /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/

# mv /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/train.parquet /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/failed_train.parquet
# mv /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/test.parquet /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/failed_test.parquet

# python examples/data_preprocess/car_crash.py \
#     --annotation_json_pattern /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/annotations_passed_12times.json \
#     --video_base_path /shared/jibf/data/CarCrashDataset/tracking_visualization/ \
#     --rule_base_crash_detection_path /shared/jibf/data/CarCrashDataset/rule_base_crash_results/ \
#     --local_save_dir /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/

# mv /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/train.parquet /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/passed_train.parquet
# mv /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/test.parquet /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/passed_test.parquet

# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/train.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/failed_train.parquet:4.0 \
#  /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/passed_train.parquet:1.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/train_with_normal_balanced.parquet \
#  --shuffle \
#  --seed 42

# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/test.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/failed_test.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/passed_test.parquet:1.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/test_with_normal_balanced.parquet \
#  --shuffle \
#  --seed 42


################################ truncate normal video & without rule base proposal ################################
# # 2222 + 117
# python examples/data_preprocess/car_crash.py \
#  --annotation_json_pattern /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/annotations_failed.json \
#  --video_base_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/tracking_visualization/ \
#  --rule_base_crash_detection_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/ \
#  --local_save_dir /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/

# # 1774 + 94
# python examples/data_preprocess/car_crash.py \
#  --annotation_json_pattern /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_data_random_margin0_annotations.json \
#  --video_base_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/tracking_visualization/ \
#  --rule_base_crash_detection_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/ \
#  --local_save_dir /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_random_margin0/

# # 247 + 13
# python examples/data_preprocess/car_crash.py \
#  --annotation_json_pattern /shared/jibf/data/CarCrashDataset/qwen8b_results/annotations_failed.json \
#  --video_base_path /shared/jibf/data/CarCrashDataset/tracking_visualization/ \
#  --rule_base_crash_detection_path /shared/jibf/data/CarCrashDataset/rule_base_crash_results/ \
#  --local_save_dir /shared/jibf/data/CarCrashDataset/qwen8b_results/

# mv /shared/jibf/data/CarCrashDataset/qwen8b_results/train_wo_rulebase.parquet /shared/jibf/data/CarCrashDataset/qwen8b_results/failed_train_wo_rulebase.parquet
# mv /shared/jibf/data/CarCrashDataset/qwen8b_results/test_wo_rulebase.parquet /shared/jibf/data/CarCrashDataset/qwen8b_results/failed_test_wo_rulebase.parquet

# # 1415 + 75
# python examples/data_preprocess/car_crash.py \
#     --annotation_json_pattern /shared/jibf/data/CarCrashDataset/qwen8b_results/annotations_passed.json \
#     --video_base_path /shared/jibf/data/CarCrashDataset/tracking_visualization/ \
#     --rule_base_crash_detection_path /shared/jibf/data/CarCrashDataset/rule_base_crash_results/ \
#     --local_save_dir /shared/jibf/data/CarCrashDataset/qwen8b_results/

# mv /shared/jibf/data/CarCrashDataset/qwen8b_results/train_wo_rulebase.parquet /shared/jibf/data/CarCrashDataset/qwen8b_results/passed_train_wo_rulebase.parquet
# mv /shared/jibf/data/CarCrashDataset/qwen8b_results/test_wo_rulebase.parquet /shared/jibf/data/CarCrashDataset/qwen8b_results/passed_test_wo_rulebase.parquet



# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/train_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_random_margin0/train_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen8b_results/failed_train_wo_rulebase.parquet:3.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/train_with_random_margin0_normal.parquet \
#  --shuffle \
#  --seed 42

# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_random_margin0/test_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen8b_results/failed_test_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen8b_results/passed_test_wo_rulebase.parquet:1.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_with_random_margin0_normal.parquet \
#  --shuffle \
#  --seed 42


################################ truncate normal video & without rule base proposal ################################
# # 1774 + 94
# python examples/data_preprocess/car_crash.py \
#  --annotation_json_pattern /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_data_margin0_annotations.json \
#  --video_base_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/tracking_visualization/ \
#  --rule_base_crash_detection_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/ \
#  --local_save_dir /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_margin0/

# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/train_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_margin0/train_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen8b_results/failed_train_wo_rulebase.parquet:3.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/train_with_margin0_normal.parquet \
#  --shuffle \
#  --seed 42

# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/MM_AU/CAP-DATA_chunks/normal_margin0/test_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen8b_results/failed_test_wo_rulebase.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen8b_results/passed_test_wo_rulebase.parquet:1.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_with_margin0_normal.parquet \
#  --shuffle \
#  --seed 42

################################# ppositive only ################################

python examples/data_preprocess/car_crash.py \
 --annotation_json_pattern /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/positive_data_failed.json \
 --video_base_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/tracking_visualization/ \
 --rule_base_crash_detection_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/ \
 --local_save_dir /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/

mv /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/train_wo_rulebase.parquet /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_positive_wo_rule_involved_id_only_train.parquet
mv /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/test_wo_rulebase.parquet /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_positive_wo_rule_involved_id_only_test.parquet

python examples/data_preprocess/car_crash.py \
    --annotation_json_pattern /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/other_other_collision_failed.json \
    --video_base_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/tracking_visualization/ \
    --rule_base_crash_detection_path /shared/jibf/data/MM_AU/CAP-DATA_chunks/rule_base_det_results/ \
    --local_save_dir /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/

mv /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/train_wo_rulebase.parquet /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_other_positive_wo_rule_involved_id_only_train.parquet
mv /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/test_wo_rulebase.parquet /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_other_positive_wo_rule_involved_id_only_test.parquet

python examples/data_preprocess/merge_parquet.py \
 --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_positive_wo_rule_involved_id_only_train.parquet:1.0 \
 /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_other_positive_wo_rule_involved_id_only_train.parquet:1.0 \
 --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_wo_rule_involved_id_only_positve_only.parquet \
 --shuffle \
 --seed 42

# python examples/data_preprocess/merge_parquet.py \
#  --input /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/test.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/failed_test.parquet:1.0 \
#  /shared/jibf/data/CarCrashDataset/qwen3_4b_8b_12runs/passed_test.parquet:1.0 \
#  --output /shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/test_with_normal_balanced.parquet \
#  --shuffle \
#  --seed 42