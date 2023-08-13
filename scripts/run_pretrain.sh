#!/bin/bash

OUTPUT_DIR="ablation-kstep-gcn-output/pretrained/GOProteinGCN"
PRETRAIN_DATA_DIR="datasets/ProteinKG25"

# protein sequence setting
IN_MEMORY=true
MAX_PROTEIN_SEQ_LENGTH=1024

# Knowledge pretrained model path
KNOWLEDGE_BERT_MODEL_PATH=None

# Go setting
USE_DESC=true
MAX_TEXT_SEQ_LENGTH=128
MAX_RELATION_TEXT_SEQ_LENGTH=64
PROTEIN_GO_NUM_WORKERS=1
GO_GO_NUM_WORKERS=1
PROTEIN_SEQ_NUM_WORKERS=1

# negative sampling, not used
NUM_PROTEIN_GO_NEG_SAMPLE=1
NUM_GO_GO_NEG_SAMPLE=1
NEGTIVE_SAMPLING_FN="simple_random"
PROTEIN_GO_SAMPLE_HEAD=false
PROTEIN_GO_SAMPLE_TAIL=true
GO_GO_SAMPLE_HEAD=true
GO_GO_SAMPLE_TAIL=true

# Protein sequence pretrained model
ENCODER_MODEL_NAME="Rostlab/prot_bert"

# TextModel
TEXT_MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
GO_ENCODER_CLS="bert"
PROTEIN_ENCODER_CLS="bert"

# Knowledge model (GCN)
GCN_INCLUDE_RELATIONS=true
KNOWLEDGE_LAYERS_STEP=10

# Train
MAX_STEPS=50000
# per gpu batch size
BATCH_SIZE=16

ACCUMULATION_STEPS=8
SCHEDULER_TYPE="linear"
WEIGHT_DECAY=0.02
OPTIMIZE_MEMORY=true

# Loss
MLM_LAMBDA=1.0
MLM_LEARNING_RATE=1e-5
LM_WARMUP_RATIO=0.167
PFI_LAMBDA=1.0

# use PFI Task
USE_PFI=false

# Run on machine with 4 GPUs or add --num_gpus=4 after deepspeed
deepspeed run_pretrain.py \
  --do_train \
  --output_dir $OUTPUT_DIR \
  --pretrain_data_dir $PRETRAIN_DATA_DIR \
  --in_memory $IN_MEMORY \
  --max_protein_seq_length $MAX_PROTEIN_SEQ_LENGTH \
  --model_protein_seq_data false \
  --model_protein_go_data true \
  --model_go_go_data false \
  --use_desc $USE_DESC \
  --use_pfi $USE_PFI \
  --max_text_seq_length $MAX_TEXT_SEQ_LENGTH \
  --dataloader_protein_go_num_workers $PROTEIN_GO_NUM_WORKERS \
  --dataloader_go_go_num_workers $GO_GO_NUM_WORKERS \
  --dataloader_protein_seq_num_workers $PROTEIN_SEQ_NUM_WORKERS \
  --num_protein_go_neg_sample $NUM_PROTEIN_GO_NEG_SAMPLE \
  --num_go_go_neg_sample $NUM_GO_GO_NEG_SAMPLE \
  --negative_sampling_fn $NEGTIVE_SAMPLING_FN \
  --protein_go_sample_head $PROTEIN_GO_SAMPLE_HEAD \
  --protein_go_sample_tail $PROTEIN_GO_SAMPLE_TAIL \
  --go_go_sample_head $GO_GO_SAMPLE_HEAD \
  --go_go_sample_tail $GO_GO_SAMPLE_TAIL \
  --encoder_model_file_name $ENCODER_MODEL_NAME \
  --text_model_file_name $TEXT_MODEL_NAME \
  --knowledge_model_file_name $KNOWLEDGE_BERT_MODEL_PATH \
  --gcn_include_relations $GCN_INCLUDE_RELATIONS \
  --knowledge_layers_step $KNOWLEDGE_LAYERS_STEP \
  --go_encoder_cls $GO_ENCODER_CLS \
  --protein_encoder_cls $PROTEIN_ENCODER_CLS \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --weight_decay $WEIGHT_DECAY \
  --optimize_memory $OPTIMIZE_MEMORY \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --lr_scheduler_type $SCHEDULER_TYPE \
  --mlm_lambda $MLM_LAMBDA \
  --lm_learning_rate $MLM_LEARNING_RATE \
  --lm_warmup_ratio $LM_WARMUP_RATIO \
  --pfi_lambda $PFI_LAMBDA \
  --seed 2021 \
  --deepspeed dp_config.json \
  --fp16 \
  --dataloader_pin_memory
