mkdir -p gcn-output/remote_homology

nohup sh run_main.sh \
      --model "gcn-output/pretrained/GOProteinGCN/checkpoint-50000/encoder" \
      --tokenizer "Rostlab/prot_bert" \
      --output_file gcn-remote_homology \
      --task_name remote_homology \
      --do_train True \
      --epoch 10 \
      --mean_output False \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 16 \
      --eval_step 50 \
      --eval_batchsize 8 \
      --warmup_ratio 0.08 \
      --learning_rate 4e-5 \
      --seed 5 \
      --frozen_bert False > gcn-output/remote_homology/GOProteinGCN.out 2>&1