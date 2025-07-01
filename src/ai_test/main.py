import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    InputGuardrail,
    GuardrailFunctionOutput
)

load_dotenv()
API_KEY = os.environ["GEMINI_API_KEY"]

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# Guardrail agent
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

# Math and History agents
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

# Guardrail function 
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context, run_config=config)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework  # block if not homework
    )


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)
#
async def main():
    try:
        result = await Runner.run(triage_agent, "who was the first president of the united states?", run_config=config)
        print("\n[History Question Result]:\n", result.final_output)
    except Exception as e:
        print("\n[Guardrail Blocked History Question]:", str(e))

    try:
        result = await Runner.run(triage_agent, "what is life", run_config=config)
        print("\n[Philosophical Question Result]:\n", result.final_output)
    except Exception as e:
        print("\n[Guardrail Blocked Philosophical Question]:", str(e))

if __name__ == "__main__":
    asyncio.run(main())
