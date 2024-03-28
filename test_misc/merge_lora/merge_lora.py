from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = '/mnt/nas/yuzhao/model/Llama-2-13b-hf'
    adapter_name_or_path = '/mnt/nas/yuzhao/code/llm/Firefly/output/Llama-2-13b-hf/2023-08-08'
    save_path = 'checkpoints/Llama-2-13b-chat-hf-qlora-sft-merge'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    
    merge_lora_to_base_model()
