from transformers import BertTokenizer, BertForSequenceClassification, pipeline

model_name = "uer/roberta-base-finetuned-dianping-chinese"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

review_movie = "这部电影太精彩了，节奏紧凑，毫无冷场，完全沉浸其中！"
review_food = "这家店口味稳定，已经回购好几次了，值得信赖！"

result_movie = classifier(review_movie)[0]
result_food = classifier(review_food)[0]

print(f"影评分类结果: {result_movie['label']}（置信度: {result_movie['score']:.2f}）")
print(f"外卖评价分类结果: {result_food['label']}（置信度: {result_food['score']:.2f}）")
