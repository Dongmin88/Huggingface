from transformers import MllamaForConditionalGeneration,MllamaProcessor
import torch
from PIL import Image
import requests

model = MllamaForConditionalGeneration.from_pretrained(
  'Bllossom/llama-3.2-Korean-Bllossom-AICA-5B',
  torch_dtype=torch.bfloat16,
  device_map='auto'
)
processor = MllamaProcessor.from_pretrained('Bllossom/llama-3.2-Korean-Bllossom-AICA-5B')

# url = "https://cdn.discordapp.com/attachments/1156141391798345742/1313407928287494164/E18489E185B3E1848FE185B3E18485E185B5E186ABE18489E185A3E186BA202021-11-1620E1848BE185A9E18492E185AE2011.png?ex=675005f4&is=674eb474&hm=fc9c4231203f53c27f6edd2420961c182dd4a1ed14d4b73e04127f11393729af&"
# image = Image.open(requests.get(url, stream=True).raw)

messages = [
  {'role': 'user','content': [
    {'type': 'text','text': '자연어처리 15주치 커리큘럼을 짜줘'}
    ]},
  ]

input_text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

inputs = processor(
    images=None,
    text=input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs,max_new_tokens=256,temperature=0.1,eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),use_cache=False)
print(processor.decode(output[0]))
