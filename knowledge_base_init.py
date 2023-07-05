from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst

def main():
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_konwledge_base(embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    create_knowledge_base: bool = input('是否创建知识库？True or False\n')
    if create_knowledge_base == 'True':
        vs_path = os.path.join(os.path.dirname(__file__),
                          'knowledge_base',
                          input("Please enter the repository name you created 请输入你创建的知识库名字：\n"))
    else:
        vs_path = os.path.join(os.path.dirname(__file__),
                          'knowledge_base',
                          input("Input your knowledge base path 请输入已有知识库名字：\n"))
    create_knowledge_base_list: bool = input('是否选择文件夹？True or False\n')
    if create_knowledge_base_list == 'True':
        filepath = input("Input your local knowledge folder path 请输入本地知识文件夹路径：\n")
        import glob
        file_list = [i for j in File_Types for i in glob.glob(filepath +"*."+j) ]
        local_doc_qa.knowledge_add(file_list,vs_path)
    else:
        filepath = input("Input your local knowledge file path 请输入本地知识文件路径：\n")
        local_doc_qa.knowledge_add(filepath,vs_path)
        
if __name__ == "__main__":
    main()