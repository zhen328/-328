pip install jieba                # 中文分词
pip install transformers         # Hugging Face模型库
pip install spacy                # 工业级NLP
pip install sentence-transformers# 文本嵌入
python -m spacy download zh_core_web_sm  # spaCy中文模型
from transformers import pipeline

# 加载预训练模型
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

results = classifier([
    "这款手机拍照效果太惊艳了！",
    "售后服务极差，再也不买这个品牌了",
    "中规中矩，没有特别亮点"
])

for result in results:
    print(f"文本: {result['label']} (置信度: {result['score']:.4f})")
    文本: positive (置信度: 0.9987)
文本: negative (置信度: 0.9892)
文本: negative (置信度: 0.7231)  # 中性文本可能被归类为负面


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

sentences = [
    "深度学习需要大量计算资源",
    "神经网络训练依赖GPU加速",
    "今天天气真好"
]

embeddings = model.encode(sentences)

# 计算相似度矩阵
sim_matrix = cosine_similarity(embeddings)
print("相似度矩阵:\n", sim_matrix.round(2))


相似度矩阵:
 [[1.   0.82 0.12]
 [0.82 1.   0.09]
 [0.12 0.09 1.  ]]


![85168c7fcb811e5eae298fef0a10343](https://github.com/user-attachments/assets/e4c88db5-bae3-4169-b7e7-e59fed7ee7a6)
![0b621db55582accfbb6764987acfb8c](https://github.com/user-attachments/assets/1e77ed27-efd3-42f3-832b-4cee8ff08a23)



