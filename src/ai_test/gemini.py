import asyncio
import os
from dotenv import load_dotenv

from agents import (
    Agent,
    InputGuardrail,
    GuardrailFunctionOutput,
    Runner,
    RunConfig,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from pydantic import BaseModel

# --- Your Gemini Setup ---
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Use the correct base_url for Gemini through OpenAI client
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Specify the Gemini model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", # Or "gemini-1.5-flash", "gemini-1.5-pro" etc.
    openai_client=client
)

# Your run configuration
config = RunConfig(
    model=model,
    model_provider=client, # This seems redundant with openai_client in the model, but let's keep it as per your code
    tracing_disabled=True # Disable tracing if not needed
)

# --- Example Code Definitions (Integrated) ---

class HomeworkOutput(BaseModel):
    """Output schema for the guardrail agent."""
    is_homework: bool
    reasoning: str

# Agent to check if the query is homework
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Determine if the user's request is asking for help with homework or a specific assignment.",
    output_type=HomeworkOutput,
    # Pass the run_config to the agent definition itself,
    # or rely on it being passed to the initial Runner.run.
    # Let's rely on passing it to the initial Runner.run for simplicity.
)

# Specialist agents
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples.",
    # Will inherit run_config from parent run
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
    # Will inherit run_config from parent run
)

# Asynchronous guardrail function
async def homework_guardrail(ctx, agent, input_data: str):
    """
    Guardrail function to check if the input is homework.
    Uses the guardrail_agent internally.
    """
    # Run the guardrail_agent. It will inherit the run_config from ctx.context
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)

    # Tripwire triggers if it is *not* homework
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
        # Optionally, provide a message if tripped:
        # tripwire_message="This appears not to be homework. Please ask a homework-related question."
    )

# Triage agent with handoffs and guardrails
triage_agent = Agent(
    name="Triage Agent",
    # Update instructions to reflect it handles homework questions
    instructions="You are a triage agent that determines the correct specialist tutor agent (Math Tutor or History Tutor) for a user's homework question. If the input is not a homework question, the guardrail should prevent the conversation from proceeding.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
    # Will inherit run_config from parent run
)

# --- Main Execution using Async ---

async def main():
    """Runs the triage agent with example inputs."""

    print("Running with a potential homework question (History):")
    # Pass your 'config' to the initial Runner.run call
    result_history = await Runner.run(
        triage_agent,
        "who was the first president of the united states?",
        run_config=config
    )
    print("\nHistory Question Result:")
    print(f"Final Output: {result_history.final_output}")
    print(f"Handoff Used: {result_history.handoff_used.name if result_history.handoff_used else 'None'}")
    print("-" * 20)


    print("Running with a non-homework question:")
    # Pass your 'config' to the initial Runner.run call
    result_non_homework = await Runner.run(
        triage_agent,
        "what is life",
        run_config=config
    )
    print("\nNon-Homework Question Result:")
    # Note: If the guardrail trips, the final_output might be an error message
    # or a default response depending on how tripping is handled by the library
    # or your agent's default behavior.
    print(f"Final Output: {result_non_homework.final_output}")
    print(f"Handoff Used: {result_non_homework.handoff_used.name if result_non_homework.handoff_used else 'None'}")
    print("-" * 20)

    print("Running with a potential homework question (Math):")
    # Pass your 'config' to the initial Runner.run call
    result_math = await Runner.run(
        triage_agent,
        "Can you help me solve for x in the equation 2x + 5 = 11?",
        run_config=config
    )
    print("\nMath Question Result:")
    print(f"Final Output: {result_math.final_output}")
    print(f"Handoff Used: {result_math.handoff_used.name if result_math.handoff_used else 'None'}")
    print("-" * 20)


if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    asyncio.run(main())