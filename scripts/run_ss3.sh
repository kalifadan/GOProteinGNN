mkdir -p gcn-output/ss3

nohup sh run_main.sh \
      --model "gcn-output/pretrained/GOProteinGCN/checkpoint-50000/encoder" \
      --tokenizer "Rostlab/prot_bert" \
      --output_file ss3-GOProteinGCN \
      --task_name ss3 \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 16 \
      --eval_step 50 \
      --eval_batchsize 4 \
      --warmup_ratio 0.08 \
      --learning_rate 5e-5 \
      --seed 3 \
      --frozen_bert False > gcn-output/ss3/GOProteinGCN.out 2>&1
