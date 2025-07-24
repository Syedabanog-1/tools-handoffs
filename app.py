import os
import requests
from dotenv import load_dotenv
from agents import (
    Agent,
    function_tool,
    RunConfig,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    Runner,
)
from rich import print

#  Load environment variables
load_dotenv()
set_tracing_disabled(True)

# Gemini API Key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

#  Configure Gemini model
provider = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=GEMINI_API_KEY,
)

model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash"
)

config = RunConfig(
    model=model,
    model_provider=model,
    tracing_disabled=True
)

# Tool 1: Dynamic Location
@function_tool
def get_current_location() -> str:
    """Fetch the user's current city using IP address."""
    try:
        response = requests.get("https://ipinfo.io/json")
        if response.status_code == 200:
            data = response.json()
            return f"{data.get('city')}, {data.get('region')}, {data.get('country')}"
        else:
            return "Unable to determine location"
    except Exception as e:
        return f"Error fetching location: {str(e)}"

#  Tool 2: Simulated Breaking News
@function_tool
def get_breaking_news() -> str:
    """Return a sample breaking news headline."""
    return "Climate Change: The International Court of Justice has ruled that countries are legally obligated to curb emissions and protect the climate."

#  Agent 1: Plant biology expert
plant_agent = Agent(
    name="plant_agent",
    model=model,
    instructions="You are an expert in biology and plants. Answer user questions about biology in a simple way."
)

#  Main Assistant Agent (with tools + handoffs)
assistant = Agent(
    name="main_assistant",
    model=model,
    instructions="You are a smart assistant that can answer questions and use tools for current location and breaking news.",
    tools=[get_current_location, get_breaking_news],
    handoffs=[plant_agent]
)

#  Run the system
result = Runner.run_sync(
    assistant,
    """
    1. What is my current location?
    2. Any breaking news?
    3. What is photosynthesis?
    """,
    run_config=config
)

# Output Results
print("=" * 50)
print("Final Agent Used:", result.last_agent.name)
print(result.new_items)
print("Final Output:\n", result.final_output)
