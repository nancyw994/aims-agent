import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def LMF_LLM(prompt: str) -> str:
    """LFM2.5-1.2B-Thinking (free) via OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/xavierqian/lfm-2.5-1.2b-thinking",
            "X-Title": "lfm-2.5-1.2b-thinking",
        },
        model="liquid/lfm-2.5-1.2b-thinking:free",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""

class Agent:
    def __init__(self, llm_call=None):
        self._llm_call = llm_call if llm_call is not None else LMF_LLM

    def call_llm(self, prompt: str) -> str:
        try:
            return self._llm_call(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

def plan_steps(agent: Agent, motivation: str, data_path: str | None = None) -> list[str]:
    prompt = f"The user wants to: {motivation}.\n"
    if data_path:
        prompt += f"Data file: {data_path}\n"
    prompt += "Provide a short, ordered list of high-level steps (one per line, numbered 1. 2. 3. ...). Output only the list."
    response = agent.call_llm(prompt)
    steps = []
    for line in response.strip().splitlines():
        line = line.strip()
        for sep in (". ", ") ", "- "):
            if sep in line and line.split(sep)[0].strip().isdigit():
                line = line.split(sep, 1)[1]
                break
        if line:
            steps.append(line)
    return steps if steps else [response.strip()]

def parse_args():
    p = argparse.ArgumentParser(description="AI Agent for ML in Materials Science")
    p.add_argument("--motivation", required=True, help="User's goal in natural language")
    p.add_argument("--data", default=None, help="Path to data file (optional)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent = Agent()
    steps = plan_steps(agent, args.motivation, args.data)
    print("Plan:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")