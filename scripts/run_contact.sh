mkdir -p gcn-output/contact

nohup sh run_main.sh \
      --model "gcn-output/pretrained/GOProteinGCN/checkpoint-50000/encoder" \
      --tokenizer "Rostlab/prot_bert" \
      --output_file contact-GOProteinGCN \
      --task_name contact \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --eval_step 50 \
      --eval_batchsize 1 \
      --warmup_ratio 0.08 \
      --learning_rate 3e-5 \
      --seed 3 \
      --frozen_bert False > gcn-output/contact/GOProteinGCN.out 2>&1
