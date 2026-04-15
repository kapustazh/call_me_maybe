from typing import Any
from llm_sdk import Small_LLM_Model

# from pydantic import BaseModel, ConfigDict, Field
# import torch
import numpy as np


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
        self.model: Small_LLM_Model = (
            Small_LLM_Model(model_name=model)
            if model != ""
            else Small_LLM_Model()
        )

    def run(self) -> None:
        with open(file=self.input_path, mode="r", encoding="utf-8") as f:
            input_text: str = f.read()

        with open(file=self.functions_path, mode="r", encoding="utf-8") as f:
            functions_text: str = f.read()

        input_ids = self.model.encode(text=input_text)[0].tolist()
        functions_text = self.model.encode(text=functions_text)[0].tolist()
        generated_ids: list[int] = []
        for _ in range(10):
            logits: list[float] = self.model.get_logits_from_input_ids(
                input_ids=input_ids
            )
            next_token_id = int(np.argmax(logits))

            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)

        generated_text = self.model.decode(generated_ids)
        print(generated_text)
