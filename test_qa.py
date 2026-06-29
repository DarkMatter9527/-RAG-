from whoosh.qparser import QueryParser
from whoosh.index import open_dir
import jieba
 
from  extract_feature import *

 
import cal_score
 
import faiss
import numpy as np
 
 
import jieba
 
import json
import clip_feature
import describe
import qwen3_embedding

ix = open_dir("indexdir")
text_index = faiss.read_index("text_index.faiss")
img_index = faiss.read_index("img_index.faiss")
with open("img_path_dict") as f:
    img_path_dict=json.load(f)
with open("text_path_dict") as f:
    text_path_dict=json.load(f)

def tfidf_search(query):
    # 创建使用结巴分词的分析器
    words=jieba.cut(query)
    parser = QueryParser("content", schema=ix.schema)
    query = parser.parse(text=" OR ".join(words))
    # 执行搜索
    results=[]
    with ix.searcher() as searcher:
        hits = searcher.search(query)
        for hit in hits:
            results.append([hit["id"],hit.score])
    max_score=max(results,key=lambda s:s[1])[1]
    results=[[id,score/max_score] for id,score in results]
    return results
def convert_d2cos(d):
    return 1-d**2/2
def clip_search(query):
    vector=clip_feature.extract_text_vector(query)
    distance,ids=img_index.search(np.array([vector]), 20)
    #print (distance)
    result=[[img_path_dict[str(s)],convert_d2cos(d)] for s,d in zip(ids[0],distance[0])]
    return result
def text_search(query):
    vector=qwen3_embedding.extract_qwen_embedding(query)
    distance,ids=text_index.search(np.array([vector]), 20)
    #print (distance)
    #result=[text_path_dict[str(s)] for s in ids[0]]
    result=[[text_path_dict[str(s)],convert_d2cos(d)] for s,d in zip(ids[0],distance[0])]
    return result
def build_id_to_docnum_map(ix):
    id_map = {}
    with ix.reader() as reader:
        # 遍历所有文档
        for docnum in range(reader.doc_count_all()):
 
            doc = reader.stored_fields(docnum)
 
            id_map[doc["id"]] = doc["audio_text"]
    return id_map

id_document_map=build_id_to_docnum_map(ix)
 
 

#query="印度的飞机在哪失事了？"
#query="人工智能能做什么？"
#query="国家主席在何时何地会见了新加坡总理"
#query="消费者国补是干嘛的"
def qa(query):
    #多路召回
    item_list1=tfidf_search(query)
    item_list2=clip_search(query)
    item_list3=text_search(query)

    #多路merge
    item_score={}
    for s,score in item_list1+item_list2+item_list3:
        item_score[s]=item_score.get(s,0)+score
    item_score=sorted(item_score.items(),key=lambda s:s[1],reverse=True)[0:3]
    #print (item_score)
    result_score=[]
    for item,score in item_score:
        #获取音频对应的文字
        audio_text=id_document_map[item]
        #print (audio_text)
        prompt=f"背景信息{audio_text}问题{query} 如果不图片和问题不相关，直接输出不相关，不要输出无关内容"
        #item召回的视频
        #qwen2.5 VL 做问答
        result=describe.extract_text_feature(item,prompt)
        #qwen2.5 reranker
        score=cal_score.reranker(query,result)
        result_score.append([result,item,score])
        if score>=0.9:
            break
    answer,item,score=sorted(result_score,key=lambda s:s[-1],reverse=True)[0]
    print (answer)
    print (item)
    item=item.replace("F:\video","static\videos")
    return answer+"\n"+item
    #print (score,result)
#print (item_score)
