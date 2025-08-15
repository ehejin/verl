set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_PATH="/lfs/ampere4/0/echoi1/digitial-human-lm/data/reddit"
VERL_PATH="/lfs/ampere4/0/echoi1/digitial-human-lm/verl"
OUTPUT_DIR="/dfs/scratch0/echoi1/verl/grpo"

export CUDA_VISIBLE_DEVICES=3,5,6,7
export NEW_HF_CACHE=/dfs/scratch0/echoi1/hf-cache

export HF_HOME="$NEW_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$NEW_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$NEW_HF_CACHE/hub"
export HF_DATASETS_CACHE="$NEW_HF_CACHE/datasets"
export XDG_CACHE_HOME="$NEW_HF_CACHE"    
export VLLM_DOWNLOAD_DIR="$NEW_HF_CACHE/hub"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path="$VERL_PATH/recipe/usim/reward.py"   \
    custom_reward_function.name="compute_reward" \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=400 \
    data.max_prompt_length=4000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.chat_template_path="$VERL_PATH/recipe/usim/character_template.txt"\
    actor_rollout_ref.model.path="Qwen/Qwen2.5-7B" \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=400 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen2_5_vl_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@