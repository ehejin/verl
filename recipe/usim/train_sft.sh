set -e
set -x  # Print commands

OUTPUT_DIR="/dfs/scratch0/echoi1/verl/sft"
DATA_DIR="/lfs/ampere4/0/echoi1/digitial-human-lm/data/reddit_sft"
VERL_DIR="/lfs/ampere4/0/echoi1/digitial-human-lm/verl"

export CUDA_VISIBLE_DEVICES=7,0
export NEW_HF_CACHE=/dfs/scratch0/echoi1/hf-cache

export HF_HOME="$NEW_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$NEW_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$NEW_HF_CACHE/hub"
export HF_DATASETS_CACHE="$NEW_HF_CACHE/datasets"
export XDG_CACHE_HOME="$NEW_HF_CACHE"    
export VLLM_DOWNLOAD_DIR="$NEW_HF_CACHE/hub"

echo "Starting SFT Training"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# Launch training
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    +data.chat_template_path="$VERL_DIR/recipe/usim/character_template.txt"\
    data.multiturn.enable=false \
    data.max_length=6000 \
    data.train_batch_size=2 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="Qwen/Qwen2.5-7B" \
    model.fsdp_config.model_dtype=fp32 \
    model.enable_gradient_checkpointing=false \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=cosine \
    trainer.total_epochs=3 \
    trainer.project_name=dtwin_sft \
    trainer.experiment_name=medium_dataset_lora \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=402 \
    trainer.test_freq=402 \
    trainer.max_ckpt_to_keep=1 \
    trainer.n_gpus_per_node=2 \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear

echo "Training completed"
echo "Model saved to: $OUTPUT_DIR"