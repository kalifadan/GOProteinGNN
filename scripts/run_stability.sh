mkdir -p gcn-output/stability

nohup sh run_stability.sh \
      --model "gcn-output/pretrained/GOProteinGCN/checkpoint-50000/encoder" \
      --tokenizer "Rostlab/prot_bert" \
      --evaluation_strategy "steps" \
      --metric_for_best_model "eval_spearmanr" \
      --output_file stability-GOProteinGCN \
      --task_name stability \
      --do_train True \
      --epoch 5 \
      --mean_output True \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 32 \
      --eval_step 500 \
      --eval_batchsize 16 \
      --warmup_ratio 0.08 \
      --learning_rate 1e-5 \
      --seed 3 \
      --frozen_bert False > gcn-output/stability/GOProteinGCN.out 2>&1
