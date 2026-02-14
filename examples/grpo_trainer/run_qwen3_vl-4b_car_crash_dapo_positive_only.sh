#!/bin/bash
# Training script for car crash detection with Qwen3-VL-2B
set -x
ENGINE=${1:-vllm}



HF_MODEL_PATH="/shared/jibf/models/Qwen3-VL-4B-Thinking"

# train_path=/dev-shared/binfei/data/verl_datasets/training_MMAU1_CarCrashnormal3_.parquet
# train_path=/shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_wo_rule_atfault_positve_only.parquet
train_path=/shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen3_4b_8b_12runs/failed_wo_rule_involved_id_only_positve_only.parquet
test_path=/shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_wo_rulebase.parquet
SAVE_DIR=/shared/jibf/results/CarCrash/qwen3_vl_4b_thinking_exp16_dapo_positive_only_wo_rulebase_involved_id_only

# Start GPU monitoring in background
GPU_LOG="${SAVE_DIR}/gpu_monitor.csv"
mkdir -p "$SAVE_DIR"
python3 gpu_monitor.py --output "$GPU_LOG" --interval 0.5 &
GPU_MONITOR_PID=$!
echo "Started GPU monitoring (PID: $GPU_MONITOR_PID), logging to $GPU_LOG"

# Trap to ensure GPU monitor is killed on script exit
trap "echo 'Stopping GPU monitor...'; kill $GPU_MONITOR_PID 2>/dev/null; wait $GPU_MONITOR_PID 2>/dev/null; echo 'Analyzing GPU logs...'; python3 analyze_gpu_log.py --input '$GPU_LOG'" EXIT

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=16 \
    data.max_prompt_length=65536 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.val_batch_size=8 \
    ++data.val_max_samples=32 \
    data.shuffle=True \
    data.dataloader_num_workers=32 \
    +data.persistent_workers=True \
    +data.prefetch_factor=2 \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.overlong_buffer.enable=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +data.gen_batch_size=32              \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric=score \
    +algorithm.filter_groups.max_num_gen_batches=3 \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name='qwen3_vl_4b_thinking' \
    trainer.experiment_name='qwen3_vl_4b_car_crash_exp16_dapo_positive_only_wo_rulebase_involved_id_only' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$SAVE_DIR \
    ++trainer.rollout_data_dir=$SAVE_DIR/rollout_response \
    trainer.test_freq=1000 \
    trainer.total_epochs=10 $@
