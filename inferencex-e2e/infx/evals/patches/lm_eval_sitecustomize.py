"""Runtime compatibility hooks for lm-eval 0.4.9.2."""

import json
from typing import Any

from lm_eval.models.openai_completions import LocalChatCompletion


def _parse_generations(outputs: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG001
    results = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for output in outputs or []:
        try:
            choices = output.get("choices", [])
            parsed = ["" for _ in choices]
            for choice in choices:
                index = choice.get("index", 0)
                message = choice.get("message") or {}
                content = message.get("content")
                if content in (None, "", []):
                    content = message.get("reasoning_content") or ""
                parsed[index] = content
        except Exception:  # noqa: BLE001
            parsed = [""]
        results.extend(parsed)
    return results


LocalChatCompletion.parse_generations = staticmethod(_parse_generations)

try:
    from lm_eval.models.api_models import JsonChatStr, TemplateAPI
except ImportError:
    JsonChatStr = None
    TemplateAPI = None

if TemplateAPI is not None and JsonChatStr is not None:

    def _apply_chat_template(
        self: Any,
        chat_history: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> Any:
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        if self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        return JsonChatStr(json.dumps(list(chat_history), ensure_ascii=False))

    TemplateAPI.apply_chat_template = _apply_chat_template
