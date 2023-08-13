mkdir -p gcn-output/ss8

nohup sh run_main.sh \
      --model "gcn-output/pretrained/GOProteinGCN/checkpoint-50000/encoder" \
      --tokenizer "Rostlab/prot_bert" \
      --output_file ss8-GOProteinGCN \
      --task_name ss8 \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 1 \
      --gradient_accumulation_steps 32 \
      --eval_step 50 \
      --eval_batchsize 4 \
      --warmup_ratio 0.08 \
      --learning_rate 3e-5 \
      --seed 3 \
      --frozen_bert False > gcn-output/ss8/GOProteinGCN.out 2>&1