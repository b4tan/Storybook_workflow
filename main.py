import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from graph import build_graph
from debug_utils import debug_log

load_dotenv()

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

If I had 2–3 more hours, I’d add a lightweight web UI to preview the story, visualize the story arc, and show the judge feedback inline. I’d also add basic rate-limiting to protect the API key and prevent accidental bursts. Finally, I’d wire up simple auth (login/logout) and session storage so a user can revise a story across multiple rounds without losing their spec. Next would be to archive the chat histories like chatgpt does.
"""
STATE_KEYS = ("specifications", "story", "judge_evaluation", "judge_feedback")


def build_payload(message: str, history: List[Dict[str, str]], carry: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": message, "history": history}
    for key in STATE_KEYS:
        if key in carry:
            payload[key] = carry[key]
    return payload


def handle_general_response(state: Dict[str, Any], history: List[Dict[str, str]]) -> None:
    response = state.get("general_response", "")
    print("\nBot:", response)
    if response:
        history.append({"role": "assistant", "content": response})


def handle_story_response(state: Dict[str, Any], history: List[Dict[str, str]]) -> None:
    story_text = state.get("story", "")
    judge_feedback = state.get("judge_feedback", "")

    print("\nBot:", story_text)
    if judge_feedback:
        print(f"[Judge feedback] {judge_feedback}")

    if story_text:
        history.append({"role": "assistant", "content": story_text})
    if judge_feedback:
        history.append({"role": "assistant", "content": f"Judge feedback: {judge_feedback}"})


def update_carry(state: Dict[str, Any], carry: Dict[str, Any]) -> None:
    for key in STATE_KEYS:
        if state.get(key) is not None:
            carry[key] = state[key]


def main():
    wf = build_graph()

    print("Storyteller ready! Ask for a story, give feedback, or just chat. Type 'exit' to quit.")

    carry: Dict[str, Any] = {}
    history: List[Dict[str, str]] = []

    while True:
        user_msg = input("\nYou: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_msg})

        payload = build_payload(user_msg, history, carry)
        state = wf.invoke(payload)
        # debug_log("main", "turn_state", state)

        intent = state.get("intent")
        if intent == "general":
            handle_general_response(state, history)
        else:
            handle_story_response(state, history)

        update_carry(state, carry)


if __name__ == "__main__":
    main()
