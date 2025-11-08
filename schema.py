from typing import Optional, Literal
from pydantic import BaseModel, Field

# Define schemas
# User (Input)
class StoryRequest(BaseModel):
    request: str # the request for the story

# Classify intent from user query
class UserIntent(BaseModel):
    intent: Literal["story", "feedback", "general"] # the intent of the user

# Specifications for the story to generate (Input)
class StorySpecifications(BaseModel):
    topic: str
    tone: str
    length: int = Field(default=1000)
    style: str
    plan: str # walkthrough the story arcs

# Story Generation output will always be a string
# Will be used for input to the judge
class StoryGenerated(BaseModel):
    story: str

# Judge Evaluation (Output)
class JudgeEvaluation(BaseModel): 
    is_appropriate: bool # should be for ages 5-10
    feedback: str = ""

class UserFeedback(BaseModel):
    feedback: str # feedback from the user

