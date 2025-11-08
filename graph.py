import os, json, openai
from typing import Optional, TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from debug_utils import debug_log
from schema import (
    StoryRequest,
    UserIntent,
    StorySpecifications,
    StoryGenerated,
    JudgeEvaluation,
)

# LLM caller
def call_model(prompt: str, max_tokens=3000, temperature=0.1) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    message = resp.choices[0].message
    content = getattr(message, "content", None)

    return content

# PROMPTS
def prompt_classify_intent(message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    context = ""
    if history:
        recent = history[-6:]
        formatted = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent)
        context = f"\nConversation context:\n{formatted}\n"
    return f"""Your are an expert in classifying user intent. 
    Classify MESSAGE as either "story" (a request for a new or continued story) or "feedback" (instructions to revise/improve the current story/spec) or "general" (a general question or statement unrelated to the story). Return ONLY valid JSON like: {{"intent": "story"}} or {{"intent": "feedback"}} or {{"intent": "general"}}.{context} MESSAGE: {message}"""

def prompt_build_spec(user_request: str, specifications: Optional[str], feedback: Optional[str]) -> str:
    # Format the prior specifications and feedback
    prior = f"\nCURRENT SPEC (JSON):\n{specifications}\n" if specifications else ""
    fb = f"\nUSER FEEDBACK:\n{feedback}\n" if feedback else ""
    
    return f"""You are writing a SPECIFICATION for a children's bedtime story (ages 5–10).
Fill out the following JSON object—return ONLY valid JSON, no extra text.

Schema (keys and types must match EXACTLY):
{{
  "topic": "string",                  // concise topic/title derived from the request
  "tone": "string",                   // e.g., "cozy, reassuring, tense, etc."
  "style": "string",                  // e.g., "simple sentences, gentle imagery, some dialogue"
  "plan": "string",                   // a short walkthrough of the story arc (beginning → small non-scary problem → kind resolution → warm closing)
  "length": 1000                      // integer target word count (approximate)
}}

Constraints:
- Audience are ages 5–10 (no violence, no scares, no bullying, no romance, no medical/legal advice).
- Keep the plan concrete but brief (3–6 sentences max).
- Choose an integer for "length" (typical 350–700 unless the request implies otherwise; integer only).
- Do NOT write the story—only the JSON spec.
- Do NOT include comments in your output (the example above shows comments; your output must not).

USER REQUEST:
{user_request}
{prior}{fb}
Return ONLY the JSON object.
"""

def prompt_generate_story(spec: str) -> str:
    return f"""You are a children's storyteller (ages 5–10).
Using ONLY the following SPECIFICATION STRING, write the story. Do NOT include the spec in your output.

SPEC:
\"\"\"{spec}\"\"\"

Write the full bedtime story now.
At the very end add the token <END> on its own line.
"""

def prompt_judge(story_text: str) -> str:
    return f"""You are a simple reviewer for children's bedtime stories (ages 5–10).
Return ONLY JSON with fields:
{{
  "is_appropriate": true|false,
  "feedback": "brief explanation and suggestions for improvement only if the story is not appropriate",
}}

STORY:
\"\"\"{story_text}\"\"\"
"""

def prompt_general_response(message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    history_text = ""
    if history:
        recent = history[-6:]
        formatted = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent)
        history_text = f"\nRecent conversation:\n{formatted}\n"
    return f"""You are a friendly storyteller assistant. Explain that this program is a storytelling program for children (ages 5-10). Answer any general conversatoinal questions as long as they are polite and appropriate.{history_text}

MESSAGE:
{message}
"""

def classify_intent(message: str, history: Optional[List[Dict[str, str]]] = None) -> UserIntent:
    raw = call_model(prompt_classify_intent(message, history), max_tokens=200, temperature=0.0)
    data = json.loads(raw)
    return UserIntent(**data)

def build_spec(req: StoryRequest, specifications: Optional[str], feedback: Optional[str]) -> StorySpecifications:
    spec_text = call_model(prompt_build_spec(req.request, specifications, feedback), max_tokens=500,temperature=0.4).strip()
    data = json.loads(spec_text)
    return StorySpecifications(**data)

def generate_story(spec: StorySpecifications) -> StoryGenerated:
    spec_json = spec.model_dump_json()
    text = call_model(prompt_generate_story(spec_json), max_tokens=1200, temperature=0.35)
    story = text.split("<END>")[0].strip()
    return StoryGenerated(story=story)

def judge_story(story: StoryGenerated) -> JudgeEvaluation:
    raw = call_model(prompt_judge(story.story), max_tokens=200, temperature=0.0)
    data = json.loads(raw)
    return JudgeEvaluation(**data)

def respond_general(history: Optional[List[Dict[str, str]]], message: str) -> str:
    text = call_model(prompt_general_response(message, history), max_tokens=300, temperature=0.3)
    return text.strip()

# Graph

# Define state graph - keeps track of the state of the story
class State(TypedDict, total=False): # Typed dict for internal states
    message: str # the user message
    intent: str
    story: str # the story so far
    specifications: str # the specifications for the story
    feedback: str # feedback from the judge or user
    judge_evaluation: bool # evaluation from the judge
    judge_feedback: str # feedback from the judge (only if story not appropriate)
    general_response: str
    history: List[Dict[str, str]]

# Nodes
def n_intent(state: State):
    # debug_log("intent", "input_state", state)
    history = state.get("history")
    user_intent = classify_intent(state["message"], history)
    result: Dict[str, Any] = {"intent": user_intent.intent}
    if user_intent.intent == "feedback":
        result["feedback"] = state["message"]
    # debug_log("intent", "output_delta", result)
    return result

def n_spec(state: State):
    # If we already have a spec and intent is feedback, update; else (re)build fresh.
    prior = state.get("specifications")
    fb = state.get("feedback")
    # debug_log(
    #     "spec",
    #     "input_state",
    #     {"message": state["message"], "specifications": prior or "", "feedback": fb or ""},
    # )
    spec_model = build_spec(StoryRequest(request=state["message"]), specifications=prior, feedback=fb)
    result = {"specifications": spec_model.model_dump_json(), "feedback": ""}
    # debug_log("spec", "output_delta", {"specifications": spec_model.model_dump(), "feedback": ""})
    return result

def n_generate(state: State):
    spec = StorySpecifications(**json.loads(state["specifications"]))
    # debug_log("generate", "input_state", {"specifications": spec.model_dump()})
    story = generate_story(spec)
    result = {"story": story.story}
    # debug_log("generate", "output_delta", {"story": story.story})
    return result

def n_judge(state: State):
    # debug_log(
    #     "judge",
    #     "input_state",
    #     {
    #         "story": state.get("story", ""),
    #     },
    # )
    je = judge_story(StoryGenerated(story=state["story"]))
    ok = je.is_appropriate
    fb = "" if ok else je.feedback
    result = {
        "judge_evaluation": ok,       
        "judge_feedback": je.feedback,
        "feedback": fb,
    }
    # debug_log(
    #     "judge",
    #     "output_delta",
    #     {
    #         "judge_evaluation": ok,
    #         "judge_feedback": je.feedback,
    #         "feedback": fb,
    #     },
    # )
    return result

def n_general(state: State):
    # debug_log("general", "input_state", {"message": state["message"]})
    history = state.get("history")
    response = respond_general(history, state["message"])
    result = {"general_response": response}
    # debug_log("general", "output_delta", result)
    return result

def e_after_judge(state: State):
    # If appropriate, we can end; otherwise loop to spec to revise using judge reason as feedback.
    return "OK" if state.get("judge_evaluation", True) else "FIX"

def e_after_intent(state: State):
    return state.get("intent", "story")

def build_graph():
    g = StateGraph(State)
    g.add_node("intent", n_intent)
    g.add_node("spec", n_spec)
    g.add_node("generate", n_generate)
    g.add_node("judge", n_judge)
    g.add_node("general", n_general)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", e_after_intent, {"story": "spec", "feedback": "spec", "general": "general"})
    g.add_edge("spec", "generate")
    g.add_edge("generate", "judge")
    g.add_conditional_edges("judge", e_after_judge, {"OK": END, "FIX": "spec"})
    g.add_edge("general", END)
    compiled = g.compile()
    # debug_log(
    #     "graph",
    #     "plan",
    #     {
    #         "sequence": ["intent", "spec", "generate", "judge"],
    #         "retry_on": "judge -> spec when FIX",
    #         "general_route": "intent -> general -> END",
    #     },
    # )
    return compiled

