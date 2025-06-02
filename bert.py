from transformers import BertTokenizer, BertForSequenceClassification, pipeline

model_name = "uer/roberta-base-finetuned-dianping-chinese"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

review_movie = "是近年来最值得一看的国产佳作，强烈推荐！"
review_food = "食物完全凉了，吃起来像隔夜饭，体验极差。"

result_movie = classifier(review_movie)[0]
result_food = classifier(review_food)[0]

print(f"影评分类结果: {result_movie['label']}（置信度: {result_movie['score']:.2f}）")
print(f"外卖评价分类结果: {result_food['label']}（置信度: {result_food['score']:.2f}）")
