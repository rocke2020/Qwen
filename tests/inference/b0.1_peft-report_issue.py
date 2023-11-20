import torch
import transformers
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, GenerationConfig

model_name_or_path = "/mnt/nas1/models/qwen/Qwen-7B-Chat-Int8"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=768,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eod_id
generation_config = GenerationConfig.from_pretrained(
    model_name_or_path, trust_remote_code=True
)

peft_model_path = "/mnt/nas1/models/qwen/Qwen-7B-Chat-int8-moss-small/checkpoint-1500"
load_basic_model = 1
if load_basic_model:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
else:
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_path, trust_remote_code=True
    )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
model.generation_config = generation_config

query = "重点领域产业关键与共性技术攻关工程有哪些?"
response, history = model.chat(tokenizer, query, history=None)
print(response)


# all_raw_text = ["重点领域产业关键与共性技术攻关工程有哪些"]
# batch_raw_text = []
# for q in all_raw_text:
#     raw_text, _ = make_context(
#         tokenizer,
#         q,
#         system="You are a helpful assistant.",
#         max_window_size=model.generation_config.max_window_size,
#         chat_format=model.generation_config.chat_format,
#     )
#     batch_raw_text.append(raw_text)

# batch_input_ids = tokenizer(batch_raw_text, padding="longest")
# batch_input_ids = torch.LongTensor(batch_input_ids["input_ids"]).to(model.device)
# ic(batch_input_ids.size())
# batch_out_ids = model.generate(
#     batch_input_ids,
#     return_dict_in_generate=False,
#     generation_config=model.generation_config,
# )
# padding_lens = [
#     batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item()
#     for i in range(batch_input_ids.size(0))
# ]
# ic(padding_lens)
# batch_response = [
#     decode_tokens(
#         batch_out_ids[i][padding_lens[i] :],
#         tokenizer,
#         raw_text_len=len(batch_raw_text[i]),
#         context_length=(batch_input_ids[i].size(0) - padding_lens[i]),
#         chat_format="chatml",
#         verbose=False,
#         errors="replace",
#     )
#     for i in range(len(all_raw_text))
# ]
# print(batch_response)


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
