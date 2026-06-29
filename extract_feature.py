
import extract_frame

from PIL import Image


import faiss
import numpy as np
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import jieba
from whoosh.analysis import Tokenizer, Token
import os
from tqdm import tqdm
import json


class JiebaTokenizer(Tokenizer):
    def __call__(self, text, positions=True,**kwargs):
        tokens = jieba.cut(text)
        for token in tokens:
            t = Token()
            t.text = token
            t.pos = kwargs.get("pos", 0)  # 设置位置信息（可选）
            yield t




def extract_text(path):
    text=describe.extract_text_feature(path)
    return text


def extract_img_vector(path):
    results=[]
    type="video" if path.endswith("mp4") else "image"
    print (type)
    if type=="video":
        frames=extract_frame.extract_keyframes(path)
    else:
        frames=[Image.open(path)]
    for img in frames:
        vector=clip_feature.extract_img_vector(img)
        results.append(vector)
    return results


#提取入库特征
#3路召回特征
if __name__ == "__main__":
    import clip_feature
    import describe
    import qwen3_embedding
    import extract_audio_text
    img_index = faiss.IndexFlatL2(768)
    text_index = faiss.IndexFlatL2(1024)


    text_embeddings=[]
    img_embeddings=[]
    text_path_dict={}
    img_path_dict={}
    # 创建使用结巴分词的分析器
    jieba_analyzer = JiebaTokenizer()



    # 定义索引结构
    schema = Schema(
        id=ID(unique=True, stored=True),     # 文档ID，唯一且存储
        content=TEXT(stored=True,analyzer=jieba_analyzer),             # 文档内容，不存储（仅索引）
        audio_text=TEXT(stored=True,analyzer=jieba_analyzer)
    )

    # 创建索引目录
    import os
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    # 创建索引对象
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    path="F:\\video\\"
    file_content=[]
    for root, dirs, files in os.walk(path):
        # 计算当前目录的深度，用于缩进显示
        for file in tqdm(files):
            file_name=path+file
            #feature1
            img_vector_list=extract_img_vector(file_name)
            for vector in img_vector_list:
                img_path_dict[len(img_path_dict)]=file_name
                img_embeddings.append(vector)
            #feature2
            text=extract_text(file_name)

            if file_name.endswith("mp4"):
                audio_text=extract_audio_text.extract_audio_text(file_name)
            else:
                audio_text=""
            file_content.append(file+"\t"+text+"\t"+audio_text)
            writer.add_document(id=file_name,content=text,audio_text=audio_text)
            #feature3
            text_vector=qwen3_embedding.extract_qwen_embedding(text)
            text_path_dict[len(text_path_dict)]=file_name
            text_embeddings.append(text_vector)
            #feature4



    img_index.add(np.array(img_embeddings))
    text_index.add(np.array(text_embeddings))
    faiss.write_index(img_index, "img_index.faiss")
    faiss.write_index(text_index, "text_index.faiss")
    writer.commit()
    with open("file_content","w",encoding="utf-8") as f:
        f.writelines("\n".join(file_content))
    with open("text_path_dict","w") as f:
        json.dump(text_path_dict,f,ensure_ascii=False)
    with open("img_path_dict","w") as f:
        json.dump(img_path_dict,f,ensure_ascii=False)
    # 从文件加载索引
    #index = faiss.read_index("my_index.faiss")