#!/bin/bash

ARGS=`getopt -o "t:d:e:m:o:b:g:s:w:a:f:" -l "task_name:,data_cat:,epoch:,mean_output:,optimizer:,per_device_batch_size:,gradient_accumulation_steps:,eval_step:,warmup_ratio:,eval_batchsize:,frozen_bert:,model:,do_train:,learning_rate:,seed:,output_file:,tokenizer:" -n "test.sh" -- "$@"`

eval set -- "$ARGS"

while true ; do
        case "$1" in
                -c|--model) MODEL=$2 ; shift 2 ;;
                -t|--task_name) TASK_NAME=$2 ; shift 2 ;;
                -e|--epoch) EPOCHS=$2 ; shift 2 ;;
                -m|--mean_output) MEAN_OUTPUT=$2 ; shift 2 ;;
                -o|--optimizer) OPTIMIZER=$2 ; shift 2 ;;
                -b|--per_device_batch_size) BS=$2 ; shift 2 ;;
                -g|--gradient_accumulation_steps) GS=$2 ; shift 2 ;;
                -s|--eval_step) ES=$2 ; shift 2 ;;
                -a|--eval_batchsize) EB=$2 ; shift 2 ;;
                -w|--warmup_ratio) WR=$2 ; shift 2 ;;
                -f|--frozen_bert) FROZEN_BERT=$2 ; shift 2 ;;
                --do_train) DO_TRAIN=$2 ; shift 2 ;;
                --learning_rate) LR=$2 ; shift 2 ;;
                --seed) SEED=$2 ; shift 2 ;;
                --output_file) OI=$2 ; shift 2 ;;
                --tokenizer) TOKENIZER=$2 ; shift 2 ;;
                --) shift ; break ;;
                *) echo "Internal error!" ; exit 1 ;;
        esac
done

if [ -z "${LR}" ]
then
    LR=3e-5
fi

echo $LR

DATA_DIR=datasets/downstream_datasets
OUTPUT_DIR=gcn-output/$TASK_NAME-$SEED-$OI

python run_downstream.py \
  --task_name $TASK_NAME \
  --data_dir $DATA_DIR \
  --do_train $DO_TRAIN \
  --do_predict True \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $EB \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --warmup_ratio $WR \
  --logging_steps $ES \
  --eval_steps $ES \
  --output_dir $OUTPUT_DIR \
  --seed $SEED \
  --save_steps 200 \
  --optimizer $OPTIMIZER \
  --frozen_bert $FROZEN_BERT \
  --mean_output $MEAN_OUTPUT
