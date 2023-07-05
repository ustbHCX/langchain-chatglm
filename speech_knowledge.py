# -*- coding: utf-8 -*-
from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = False

from speechbrain.pretrained import EncoderDecoderASR
import torch
import torchaudio


def voice_into_word():
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-aishell",
                                               savedir="pretrained_models/asr-transformer-aishell")

    audio_1 = r"./test.wav"
    ddd = torchaudio.list_audio_backends()

    snt_1, fs = torchaudio.load(audio_1)
        # 将双声道转换为单声道（取平均值）
    waveform_mono = torch.mean(snt_1, dim=0, keepdim=True)
    wav_lens = torch.tensor([1.0])
    res = asr_model.transcribe_batch(waveform_mono, wav_lens)

    word = res[0][0].replace(' ', '')

    return word


def main():

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    vs_path = os.path.join(os.path.dirname(__file__),
                          'knowledge_base',
                          input("Input your knowledge base path 请输入已有知识库名字：\n"))
    history = []
    # query = input("Input your question 请输入问题：")
    query = voice_into_word()
    print(query,'\n')
    last_print_len = 0
    for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=STREAMING):
        if STREAMING:
            print(resp["result"][last_print_len:], end="", flush=True)
            last_print_len = len(resp["result"])
        else:
            print(resp["result"])
    if REPLY_WITH_SOURCE:
        source_text = [f"""出处 [{inum + 1}] {os.path.split(doc[0].metadata['source'])[-1]}：\n\n{doc[0].page_content}\n\n"""
                           # f"""相关度：{doc.metadata['score']}\n\n"""
                           for inum, doc in
                           enumerate(resp["source_documents"])]
        print("\n" + "\n".join(source_text))


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    main()
    # print(voice_into_word())