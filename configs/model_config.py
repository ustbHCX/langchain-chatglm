import torch.cuda
import logging
import os

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

embedding_model_dict = {
    "text2vec": "../text2vec-large-chinese",
}

EMBEDDING_MODEL = "text2vec"

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

llm_model_dict = {
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": '/root/autodl-tmp/model/chatglm-6b',
        "provides": "ChatGLM"
    },
}

LLM_MODEL = "chatglm-6b"

BF16 = False

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

EMBEDDING_MODEL = "text2vec"

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")

    # query      查询内容
    # vs_path    知识库路径
    # chunk_conent   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度


# 缓存知识库数量
CACHED_VS_NUM = 1

# 文本分句长度
SENTENCE_SIZE = 250

# 匹配后单段上下文长度
CHUNK_SIZE = 500

# 传入LLM的历史记录长度
LLM_HISTORY_LEN = 3

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 5

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 300

# LLM streaming reponse
STREAMING = True

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，专业的简洁的来回答用户的问题。不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

File_Types = ['txt','pdf','md','csv']