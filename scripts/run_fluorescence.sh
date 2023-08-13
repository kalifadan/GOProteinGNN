mkdir -p gcn-output/fluorescence

nohup sh run_main.sh \
      --model "gcn-output/pretrained/GOProteinGCN/checkpoint-50000/encoder" \
      --tokenizer "Rostlab/prot_bert" \
      --output_file fluorescence-GOProteinGCN \
      --task_name fluorescence \
      --do_train True \
      --epoch 15 \
      --mean_output True \
      --optimizer Adam \
      --per_device_batch_size 4 \
      --gradient_accumulation_steps 16 \
      --eval_step 50 \
      --eval_batchsize 32 \
      --warmup_ratio 0.0 \
      --learning_rate 1e-3 \
      --seed 3 \
      --frozen_bert True > gcn-output/fluorescence/GOProteinGCN.out 2>&1