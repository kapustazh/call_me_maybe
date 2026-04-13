from llm_sdk import Small_LLM_Model
from src.pipeline import (
    Tokenization,
    TokenSelection,
    LLMProcessing,
)


def build_inference(
    prompt: str,
    llm_sdk: Small_LLM_Model,
    max_new_tokens: int = 32,
) -> str:
    tokenization = Tokenization(model=llm_sdk)
    llm = LLMProcessing(model=llm_sdk)
    selector = TokenSelection()
    context_ids = tokenization.execute(prompt=prompt)
    generated_ids: list[int] = []
    # matches = 0
    for i in range(max_new_tokens):
        logits = llm.execute(context_ids)
        next_token_id = selector.execute(logits=logits)

        generated_ids.append(next_token_id)
        context_ids.append(next_token_id)

    generated_text = tokenization.decode(generated_ids)
    print("generated_text: ", generated_text)
