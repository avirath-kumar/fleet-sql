"""Make the last message the prose answer, not the JSON that produced it.

`response_format=Answer` makes the structured output the final message, so the
Messages view of every trace ends in a blob:

    {"answer":"The 747-8 fleet has 12 aircraft...","counts":[{"label":...

The answer is in there, under `answer`, wrapped in the fields around it. Nobody
opening a trace to see what the agent said should have to read JSON.

This REPLACES that message's content with the prose by returning a message with
the same id -- `add_messages` treats a repeated id as an overwrite rather than
an append. Appending a second message does not help: the blob is still there,
still above the readable one.

Nothing is lost. `structured_response` is untouched on the run's outputs, and
that is what the evaluators read -- `counts` in particular, which is how
answer_is_correct checks a figure against the database. The split the trace
wants is exactly the split that already exists: prose in `messages`, everything
structured beside it.
"""
from __future__ import annotations

from langchain.agents.middleware import after_agent
from langchain.messages import AIMessage


def _text(content) -> str:
    """The message's text, whether a string or Responses API content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return str(content or "")


@after_agent
def readable_answer(state, runtime):  # noqa: ANN001, ARG001 - hook signature
    if not isinstance(state, dict):
        return None
    response = state.get("structured_response")
    answer = str((response.get("answer") if isinstance(response, dict)
                  else getattr(response, "answer", "")) or "").strip()
    if not answer:
        # A failed run has no prose to show. Leave the transcript as it is
        # rather than blanking the only record of what happened.
        return None

    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    last_id = getattr(last, "id", None)
    # Only rewrite the structured message itself: an AI turn whose text is the
    # JSON we are about to replace.
    if getattr(last, "type", None) != "ai" or not last_id:
        return None
    if _text(getattr(last, "content", None)).lstrip()[:1] != "{":
        return None
    return {"messages": [AIMessage(id=last_id, content=answer)]}
