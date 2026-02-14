for step in 590 570 560 540 520 470 490
do
# mv /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp6_dapo_12runs_data/global_step_${step}/actor/huggingface /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp6_dapo_12runs_data/global_step_${step}/actor/huggingface_backup
# cp -r /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp5_dapo/global_step_10/actor/huggingface/ /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp6_dapo_12runs_data/global_step_${step}/actor/
python -m verl.model_merger merge \
 --backend fsdp \
--local_dir  /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp16_dapo_positive_only_wo_rulebase_involved_id_only/global_step_${step}/actor/ \
 --target_dir  /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp16_dapo_positive_only_wo_rulebase_involved_id_only/global_step_${step}/huggingface_model
 
done

#  --local_dir  /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp15_dapo_positive_only_wo_rulebase_id_reward/global_step_${step}/actor/ \
#  --target_dir  /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp15_dapo_positive_only_wo_rulebase_id_reward/global_step_${step}/huggingface_model

# --local_dir  /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp16_dapo_positive_only_wo_rulebase_involved_id_only/global_step_${step}/actor/ \
#  --target_dir  /shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp16_dapo_positive_only_wo_rulebase_involved_id_only/global_step_${step}/huggingface_model

