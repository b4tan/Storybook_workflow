# Storyteller CLI – System Design Overview
Completed feature ~ 3 hours
## High-Level Flow

1. User sends a message.
2. Intent classifier routes the turn:
   - `story` / `feedback` → story pipeline.
   - `general` → friendly chat response.
3. Story pipeline steps:
   - Build or refine a spec.
   - Generate a story from the spec.
   - Judge approves or returns feedback.
4. Judge feedback (when present) feeds back into the spec builder, otherwise the loop ends and we show the story.

### Simple Block Diagram

```
User
  |
  v
Intent Classifier
  |----> General Responder ----> reply to user
  |
  v
Spec Builder --> Story Generator --> Judge --(pass)--> story to user
    ^                                   |
    |                                   |
    +-----------------------------(feedback, if present)
    Back to refine spec
```

Evaluator–optimizer loop: judge critiques each story; negative feedback is routed back into the spec prompt to refine before regenerating.
General chat prompt stays lightweight to avoid mixing responsibilities with the storytelling path.

## File Structure

- `main.py`: CLI harness that loops over user turns, remembers recent story state, and prints results.
- `graph.py`: prompt helpers, OpenAI wrapper, and LangGraph nodes/edges composing the workflow.
- `schema.py`: Pydantic models for requests, specs, generated stories, and judge results.
- `debug_utils.py`: optional console logging utilities, gated by the `DEBUG_AGENT` environment variable.

