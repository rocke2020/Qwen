# a1_batch_reference
file=test_misc/inference/b0_infer_on_sft_model.py
file=test_misc/inference/vllm_wrapper.py
nohup python $file \
    > $file.log 2>&1 &