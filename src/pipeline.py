from llm_sdk import Small_LLM_Model
from src.vocab import Vocab

# from importlib import Path

# import json
# import numpy as np


# from pydantic import BaseModel, ConfigDict, Field
# import torch

from src.io_utils import load_function_definitions, load_prompt_items


class Pipeline:
    def __init__(
        self,
        functions_path: str,
        input_path: str,
        output_path: str,
        model: str = "",
    ) -> None:
        self.functions_path: str = functions_path
        self.input_path: str = input_path
        self.output_path: str = output_path
        self._model: Small_LLM_Model = (
            Small_LLM_Model(model_name=model)
            if model != ""
            else Small_LLM_Model()
        )

    def run(self) -> None:
        prompt_items = load_prompt_items(self.input_path)
        function_definitions = load_function_definitions(self.functions_path)
        vocab = Vocab(self._model.get_path_to_vocab_file())
        vocab.print_vocab()

        # from pprint import pprint

        # for item in prompt_items:
        #     pprint(item)

        # for item in function_definitions:
        #     pprint(item)

        # input_text = json.dumps(
        #     [item.model_dump(mode="json") for item in prompt_items],
        #     ensure_ascii=False,
        # )
        # functions_text = json.dumps(
        #     [fn.model_dump(mode="json") for fn in function_definitions],
        #     ensure_ascii=False,
        # )
        # print(input_text, functions_text, sep="\n\n\n")
        # input_ids = self.model.encode(text=input_text)[0].tolist()
        # function_ids = self.model.encode(text=functions_text)[0].tolist()
        # input_ids.extend(function_ids)
        # generated_ids: list[int] = []
        # for _ in range(50):
        #     logits: list[float] = self.model.get_logits_from_input_ids(
        #         input_ids=input_ids
        #     )
        #     next_token_id = int(np.argmax(logits))

        #     generated_ids.append(next_token_id)
        #     input_ids.append(next_token_id)

        # generated_text = self.model.decode(generated_ids)
        # print(generated_text)
