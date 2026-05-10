"""
Push optimized prompts to LangSmith Prompt Hub.

This module uploads the optimized prompt (v2) from local YAML file
to LangSmith Prompt Hub with metadata and public visibility.

Usage:
    python src/push_prompts.py

The prompt is read from: prompts/bug_to_user_story_v2.yml
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()


# Configuration
INPUT_PATH = "prompts/bug_to_user_story_v2.yml"
PROMPT_NAME = "bug_to_user_story_v2"


def load_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Carrega arquivo YAML.

    Args:
        file_path: Caminho do arquivo YAML

    Returns:
        Dicionário com conteúdo do YAML ou None se erro
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Erro ao parsear YAML: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo: {e}")
        return None


def load_prompt_from_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load prompt data from a YAML file.

    Args:
        file_path: Path to the YAML file containing the prompt.

    Returns:
        Dictionary containing the prompt data.

    Raises:
        FileNotFoundError: If the YAML file does not exist.

    Example:
        >>> prompt_data = load_prompt_from_yaml("prompts/bug_to_user_story_v2.yml")
        >>> 'system_prompt' in prompt_data
        True
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Optimized prompt file not found at {file_path}")

    data = load_yaml(file_path)
    if data is None:
        raise ValueError(f"Failed to parse YAML file: {file_path}")
    return data


def validate_prompt_structure(prompt_data: Dict[str, Any]) -> List[str]:
    """
    Validate that the prompt structure meets requirements per data-model.md schema.

    Checks for:
    - metadata section with name, description, and techniques (>= 2)
    - system_prompt (non-empty)
    - user_prompt_template with {bug_report} variable
    - examples list with at least 1 item
    - No [TODO] markers

    Args:
        prompt_data: Dictionary containing the prompt data.

    Returns:
        List of validation error messages. Empty list if all validations pass.

    Example:
        >>> errors = validate_prompt_structure({"system_prompt": "You are..."})
        >>> len(errors) > 0
        True
    """
    errors = []

    # Check metadata section
    if "metadata" not in prompt_data:
        errors.append("Missing required field: metadata")
    else:
        metadata = prompt_data["metadata"]
        if "name" not in metadata or not metadata["name"]:
            errors.append("Missing required field: metadata.name")
        if "description" not in metadata or not metadata["description"]:
            errors.append("Missing required field: metadata.description")
        if "techniques" not in metadata:
            errors.append("Missing required field: metadata.techniques")
        elif (
            not isinstance(metadata["techniques"], list)
            or len(metadata["techniques"]) < 2
        ):
            errors.append("metadata.techniques must have at least 2 items")

    # Check system_prompt
    if "system_prompt" not in prompt_data:
        errors.append("Missing required field: system_prompt")
    elif not prompt_data["system_prompt"] or not prompt_data["system_prompt"].strip():
        errors.append("system_prompt cannot be empty")

    # Check user_prompt_template
    if "user_prompt_template" not in prompt_data:
        errors.append("Missing required field: user_prompt_template")
    elif "{bug_report}" not in prompt_data.get("user_prompt_template", ""):
        errors.append("user_prompt_template must contain {bug_report} variable")

    # Check examples
    if "examples" not in prompt_data:
        errors.append("Missing required field: examples")
    elif (
        not isinstance(prompt_data["examples"], list)
        or len(prompt_data["examples"]) < 1
    ):
        errors.append("examples must have at least 1 item")
    else:
        for i, example in enumerate(prompt_data["examples"]):
            if "input" not in example or not example["input"]:
                errors.append(f"examples[{i}] missing required field: input")
            if "output" not in example or not example["output"]:
                errors.append(f"examples[{i}] missing required field: output")

    # Check for [TODO] markers
    prompt_str = str(prompt_data)
    if "[TODO]" in prompt_str:
        errors.append("Prompt contains [TODO] markers that must be resolved")

    return errors


def push_prompt(
    prompt_data: Dict[str, Any], prompt_name: str, is_public: bool = True
) -> str:
    """
    Push a prompt to LangSmith Prompt Hub.

    Args:
        prompt_data: Dictionary containing the prompt data.
        prompt_name: Name for the prompt in the Hub.
        is_public: Whether to make the prompt publicly visible.

    Returns:
        The full prompt identifier (username/prompt_name).

    Raises:
        Exception: If the push fails (authentication, network, etc.).

    Example:
        >>> # When credentials are configured:
        >>> full_name = push_prompt(prompt_data, "my_prompt", is_public=True)
        >>> '/' in full_name
        True
    """
    # Import using the modern LangSmith SDK API
    from langsmith import Client
    from langchain_core.prompts import ChatPromptTemplate

    print("Pushing to LangSmith Hub...")

    # Build the ChatPromptTemplate from our YAML structure
    system_prompt = prompt_data.get("system_prompt", "")
    user_template = prompt_data.get("user_prompt_template", "{bug_report}")

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_template)]
    )

    # Initialize the LangSmith client
    client = Client()

    # Push to hub
    # Note: If you don't have a LangChain Hub handle yet, push as private first
    # then make it public via the web interface at https://smith.langchain.com/prompts
    try:
        result = client.push_prompt(
            prompt_name, object=prompt_template, is_public=is_public
        )
    except Exception as e:
        if "public prompt" in str(e).lower() and "handle" in str(e).lower():
            print("Note: Cannot create public prompt without a LangChain Hub handle.")
            print(
                "Pushing as PRIVATE instead. You can make it public later via the web interface."
            )
            result = client.push_prompt(
                prompt_name, object=prompt_template, is_public=False
            )
        else:
            raise

    return result


def main() -> int:
    """
    Main entry point for the push_prompts script.

    Reads the optimized prompt from YAML, validates it, and uploads to LangSmith Hub.

    Returns:
        Exit code: 0 for success, 1 for error.

    Example:
        >>> # When .env and prompt file are configured correctly:
        >>> exit_code = main()
        >>> exit_code == 0
        True
    """
    try:
        # Step 1: : Load prompt from YAML
        print(f"Reading optimized prompt from {INPUT_PATH}...")
        try:
            prompt_data = load_prompt_from_yaml(INPUT_PATH)
        except FileNotFoundError:
            print(f"❌ Optimized prompt file not found at {INPUT_PATH}")
            print("Please create the optimized prompt first.")
            return 1

        # Step 2: Validate prompt structure
        validation_errors = validate_prompt_structure(prompt_data)
        if validation_errors:
            print("❌ Invalid prompt structure:")
            for error in validation_errors:
                print(f"  - {error}")
            return 1

        # Step 3: Push to LangSmith Hub
        try:
            result = push_prompt(prompt_data, PROMPT_NAME, is_public=True)

            # Extract metadata for display
            metadata = prompt_data.get("metadata", {})
            tags = metadata.get("tags", [])
            techniques = metadata.get("techniques", [])

            print(f"Prompt published: {result}")
            print("Visibility: Public")
            print(f"Tags: {', '.join(tags)}")
            print(f"Techniques: {', '.join(techniques)}")

        except Exception as e:
            error_msg = str(e).lower()
            if "auth" in error_msg or "401" in error_msg or "403" in error_msg:
                print(
                    "❌ Authentication failed. Please check your LANGCHAIN_API_KEY in .env"
                )
            elif (
                "connection" in error_msg
                or "network" in error_msg
                or "timeout" in error_msg
            ):
                print(
                    "❌ Unable to connect to LangSmith Hub. Please check your internet connection."
                )
            else:
                print(f"❌ Failed to push prompt: {e}")
            return 1

        print("✓ Push completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
