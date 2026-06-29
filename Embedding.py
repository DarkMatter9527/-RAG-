from langchain.embeddings.base import Embeddings
from typing import List
import requests
import json
from transformers import BertTokenizer, BertModel
import torch
count=0
def normal(vector):
    ss=sum([s**2 for s in vector])**0.5
    return [round(s/ss,8) for s in vector]
 
class CustomEmbeddings(Embeddings):
    def __init__(self):
        pass
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
 
        """将文档转换为嵌入向量"""
        results=[ self.embed_query(text) for text in texts]
        return results
    
    def embed_query(self, text: str) -> List[float]:
        global count
        count+=1
        if count%100==0:
            print (count)
        token=tokenizer([text], padding=True, truncation=True,return_tensors='pt').to(device)
        vector=model(**token)[1].tolist()[0]
        vector=normal(vector)
 
        return vector
device = torch.device("cuda:0")
#标准bert模型，最为向量模型
model_path="E:\\code\\bge\\bge_recall"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path).to(device)