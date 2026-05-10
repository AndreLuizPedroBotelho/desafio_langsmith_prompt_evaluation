"""
Pull de prompts do LangSmith Prompt Hub.
Baixa o prompt v1 e salva localmente em YAML.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langsmith import Client

load_dotenv()

LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"
OUTPUT_DIR = Path("prompts")


def pull_prompt(prompt_name: str) -> dict:
    """Faz pull de um prompt do LangSmith e retorna como dict."""
    print(f"Fazendo pull do prompt: {prompt_name}")
    prompt = hub.pull(prompt_name, api_key=LANGCHAIN_API_KEY)

    messages = []
    for msg in prompt.messages:
        role = msg.__class__.__name__.replace("MessagePromptTemplate", "").lower()
        if role == "system":
            content = msg.prompt.template
        elif role == "human":
            content = msg.prompt.template
        else:
            content = str(msg)
        messages.append({"role": role, "content": content})

    return {
        "name": prompt_name,
        "messages": messages,
    }


def save_prompt_yaml(prompt_data: dict, filepath: Path) -> None:
    """Salva o prompt em formato YAML."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(
            prompt_data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"Prompt salvo em: {filepath}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Pull do prompt v1
    prompt_data = pull_prompt(SOURCE_PROMPT)
    output_path = OUTPUT_DIR / "bug_to_user_story_v1.yml"
    save_prompt_yaml(prompt_data, output_path)

    print("\nConteúdo do prompt:")
    for msg in prompt_data["messages"]:
        print(f"\n[{msg['role'].upper()}]")
        print(msg["content"][:300], "..." if len(msg["content"]) > 300 else "")

    print("\n✅ Pull concluído com sucesso!")


if __name__ == "__main__":
    main()
