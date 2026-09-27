"""One model builder, through the LangSmith gateway.

The gateway means no provider secret is needed and every call is rate-limited
and traced in one place. The key it spends against is the `lsv2_sk_` gateway
key, which is a different credential from the `lsv2_pt_` workspace key used for
tracing -- swapping them fails at the first model call with a 402, not a 401,
which reads as a billing problem rather than an auth one.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

GATEWAY = "https://gateway.smith.langchain.com/openai/v1"


#: Reasoning effort. `minimal` rather than unset, because the gateway rejects
#: function tools on /v1/chat/completions unless reasoning_effort is 'none':
#:
#:   "Function tools with reasoning_effort are not supported for gpt-5.6-terra
#:    in /v1/chat/completions. To use function tools, use /v1/responses or set
#:    reasoning_effort to 'none'."
#:
#: Going through the Responses API instead keeps reasoning available, which the
#: planning half of this agent needs -- deciding to aggregate rather than page
#: is exactly the judgement that degrades at 'none'.
#: `low`, not `minimal`: the gateway rejects minimal for this model --
#: "Supported values are: 'none', 'low', 'medium', 'high', 'xhigh', 'max'".
REASONING_EFFORT = os.environ.get("MODEL_REASONING_EFFORT", "low")


def build_model(model_id: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id or os.environ.get("MODEL_ID", "gpt-5.6-terra"),
        base_url=os.environ.get("OPENAI_BASE_URL", GATEWAY),
        api_key=os.environ.get("LANGSMITH_API_KEY_GATEWAY") or os.environ["OPENAI_API_KEY"],
        temperature=temperature,
        use_responses_api=True,
        model_kwargs={"reasoning": {"effort": REASONING_EFFORT}},
    )
