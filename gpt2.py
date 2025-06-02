from transformers import GPT2LMHeadModel, BertTokenizer
import torch

# 使用 GPT2 中文模型
model_name = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "当我醒来，发现自己变成了一本书"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成文本
output = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9
)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print("续写结果：")
print(result)
