import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import torch
import transformers
from transformers import (AutoConfig, AutoModel,
                          AutoTokenizer)
from configs.model_config import LLM_DEVICE


class LoaderCheckPoint:
    """
    加载自定义 model CheckPoint
    """
    # remote in the model on loader checkpoint
    no_remote_model: bool = False
    # 模型名称
    model_name: str = None
    tokenizer: object = None
    # 模型全路径
    model_path: str = None
    model: object = None
    model_config: object = None
    
    load_in_8bit: bool = False
    bf16: bool = False
    params: object = None
    device_map: Optional[Dict[str, int]] = None
    llm_device = LLM_DEVICE

    def __init__(self, params: dict = None):
        """
        模型初始化
        :param params:
        """
        self.model = None
        self.tokenizer = None
        self.params = params or {}
        self.model_name = params.get('model_name', False)
        self.model_path = params.get('model_path', None)
        self.no_remote_model = params.get('no_remote_model', False)

        self.load_in_8bit = params.get('load_in_8bit', False)
        self.bf16 = params.get('bf16', False)

    def _load_model_config(self, model_name):

        if self.model_path:
            checkpoint = Path(f'{self.model_path}')
        else:
            if not self.no_remote_model:
                checkpoint = model_name
            else:
                raise ValueError(
                    "本地模型local_model_path未配置路径"
                )

        model_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)

        return model_config

    def _load_model(self, model_name):
        """
        加载自定义位置的model
        :param model_name:
        :return:
        """
        print(f"Loading {model_name}...")
        t0 = time.time()

        if self.model_path:
            checkpoint = Path(f'{self.model_path}')
        else:
            if not self.no_remote_model:
                checkpoint = model_name
            else:
                raise ValueError(
                    "本地模型local_model_path未配置路径"
                )
        LoaderClass = AutoModel

        if not any([self.llm_device.lower() == "cpu",
                    self.load_in_8bit]):

            if torch.cuda.is_available() and self.llm_device.lower().startswith("cuda"):
                model = (
                    LoaderClass.from_pretrained(checkpoint,
                                                config=self.model_config,
                                                torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
                                                trust_remote_code=True)
                    .half()
                    .cuda()
                )
            else:
                model = (
                    LoaderClass.from_pretrained(
                        checkpoint,
                        config=self.model_config,
                        trust_remote_code=True)
                    .float()
                    .to(self.llm_device)
                )

        elif self.load_in_8bit:
            try:
                from accelerate import init_empty_weights
                from accelerate.utils import get_balanced_memory, infer_auto_device_map
                from transformers import BitsAndBytesConfig

            except ImportError as exc:
                raise ValueError(
                    "Could not import depend python package "
                    "Please install it with `pip install transformers` "
                    "`pip install bitsandbytes``pip install accelerate`."
                ) from exc

            params = {"low_cpu_mem_usage": True}

            if not self.llm_device.lower().startswith("cuda"):
                raise SystemError("8bit 模型需要 CUDA 支持，或者改用量化后模型！")
            else:
                params["device_map"] = 'auto'
                params["trust_remote_code"] = True
                params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True,
                                                                   llm_int8_enable_fp32_cpu_offload=False)

            with init_empty_weights():
                model = LoaderClass.from_config(self.model_config,trust_remote_code = True)
            model.tie_weights()
            if self.device_map is not None:
                params['device_map'] = self.device_map
            else:
                params['device_map'] = infer_auto_device_map(
                    model,
                    dtype=torch.int8,
                    no_split_module_classes=model._no_split_modules
                )
            try:

                model = LoaderClass.from_pretrained(checkpoint, **params)
            except ImportError as exc:
                raise ValueError(
                    "如果开启了8bit量化加载,项目无法启动，参考此位置，选择合适的cuda版本，https://github.com/TimDettmers/bitsandbytes/issues/156"
                ) from exc
        # Custom
        else:

            print(
                "Warning: self.llm_device is False.\nThis means that no use GPU  bring to be load CPU mode\n")
            params = {"low_cpu_mem_usage": True, "torch_dtype": torch.float32, "trust_remote_code": True}
            model = LoaderClass.from_pretrained(checkpoint, **params).to(self.llm_device, dtype=float)

        # Loading the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

        print(f"Loaded the model in {(time.time() - t0):.2f} seconds.")
        return model, tokenizer

    def clear_torch_cache(self):
        gc.collect()
        if self.llm_device.lower() != "cpu":
            if  torch.has_cuda:
                device_id = "0" if torch.cuda.is_available() else None
                CUDA_DEVICE = f"{self.llm_device}:{device_id}" if device_id else self.llm_device
                with torch.cuda.device(CUDA_DEVICE):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            else:
                print("未检测到 cuda，暂不支持清理显存")

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = self.tokenizer = None
        self.clear_torch_cache()

    def set_model_path(self, model_path):
        self.model_path = model_path

    def reload_model(self):
        self.unload_model()
        self.model_config = self._load_model_config(self.model_name)

        self.model, self.tokenizer = self._load_model(self.model_name)

        self.model = self.model.eval()