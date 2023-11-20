# a1_batch_reference a2_peft b0_infer_on_sft_model
file=tests/inference/b0_infer_on_sft_model.py
nohup python $file \
    > $file.log 2>&1 &