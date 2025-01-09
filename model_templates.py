import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display, update_display
import google.generativeai
import gradio as gr


# -------------------- Environment Setup --------------------
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validating API keys
if not (OPENAI_API_KEY and ANTHROPIC_API_KEY and GOOGLE_API_KEY):
    raise EnvironmentError(
        "One or more API keys are missing. Ensure all required API keys are set in your .env file."
    )

# Model Configurations
GPT_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-haiku-20240307"
GEMINI_MODEL = "gemini-1.5-flash"

# System Messages
SYSTEM_MESSAGES = {
    "default": "You are a helpful and friendly assistant.",
    "gpt": "You are a chatbot who is very argumentative; you disagree with anything in the conversation and challenge everything in a snarky way.",
    "claude": "You are a very polite, courteous chatbot. You try to agree with everything the user says or find common ground.",
    "gemini": "You are a chatbot who is very argumentative; you disagree with anything in the conversation and challenge everything in a snarky way.",
}

# Initialize API Clients
openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure(api_key=GOOGLE_API_KEY)


# -------------------- Utility Functions --------------------
def call_gpt(user_prompt, system_message=None, json_format=False):
    """
    Call the GPT model with a user prompt.

    Parameters:
    - user_prompt (str): The user's input.
    - system_message (str): Custom system message (defaults to SYSTEM_MESSAGES['default']).
    - json_format (bool): Whether to request the response in JSON format.

    Returns:
    - str: The GPT model's response.
    """
    system_message = system_message or SYSTEM_MESSAGES["default"]
    response_format = {"type": "json_object"} if json_format else None
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    completion = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, response_format=response_format
    )
    return completion.choices[0].message.content


def stream_gpt(user_prompt, system_message=None):
    """
    Stream GPT model's response for a user prompt.

    Parameters:
    - user_prompt (str): The user's input.
    - system_message (str): Custom system message (defaults to SYSTEM_MESSAGES['default']).

    Yields:
    - str: The response stream chunk by chunk.
    """
    system_message = system_message or SYSTEM_MESSAGES["default"]
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    stream = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, stream=True
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


def call_claude(user_prompt):
    """
    Call the Claude model with a user prompt.

    Parameters:
    - user_prompt (str): The user's input.

    Returns:
    - str: The Claude model's response.
    """
    messages = [{"role": "user", "content": user_prompt}]
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        system=SYSTEM_MESSAGES["claude"],
        messages=messages,
        max_tokens=500,
    )
    return response.content[0].text


def call_gemini(user_prompt):
    """
    Call the Gemini model with a user prompt.

    Parameters:
    - user_prompt (str): The user's input.

    Returns:
    - str: The Gemini model's response.
    """
    gemini = google.generativeai.GenerativeModel(
        model_name=GEMINI_MODEL, system_instruction=SYSTEM_MESSAGES["gemini"]
    )
    response = gemini.generate_content(user_prompt)
    return response.text


def stream_claude(user_prompt):
    """
    Stream the Claude model's response for a user prompt.

    Parameters:
    - user_prompt (str): The user's input.

    Yields:
    - str: The response stream chunk by chunk.
    """
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        temperature=0.7,
        system=SYSTEM_MESSAGES["default"],
        messages=[{"role": "user", "content": user_prompt}],
    )
    with result as stream:
        for text in stream.text_stream:
            yield text or ""


def stream_model(prompt, model):
    """
    Stream the response for a given prompt and model.

    Parameters:
    - prompt (str): The user's input.
    - model (str): The model name ("GPT", "Claude", "Gemini").

    Yields:
    - str: The response stream chunk by chunk.
    """
    if model == "GPT":
        yield from stream_gpt(prompt)
    elif model == "Claude":
        yield from stream_claude(prompt)
    elif model == "Gemini":
        yield call_gemini(prompt)
    else:
        raise ValueError(f"Unknown model: {model}")


# -------------------- Audio Generation --------------------
import base64
from io import BytesIO
from PIL import Image
from IPython.display import Audio


def talker(message):
    response = openai.audio.speech.create(model="tts-1", voice="fable", input=message)
    # response = openai.audio.speech.create())

    audio_stream = BytesIO(response.content)
    output_filename = "output_audio.mp3"
    with open(output_filename, "wb") as f:
        f.write(audio_stream.read())

    # Play the generated audio
    display(Audio(output_filename, autoplay=True))


talker("I wish we weren't sick")


# -------------------- Tooling Framework --------------------
def define_tool_property(arg_name: str, arg_type: str, description: str) -> dict:
    """
    Define a single property for a tool package.

    Parameters:
    - arg_name (str): Name of the argument.
    - arg_type (str): Type of the argument (e.g., "string", "integer").
    - description (str): Description of the argument.

    Returns:
    - dict: A dictionary defining the tool property.
    """
    return {arg_name: {"type": arg_type, "description": description}}


def create_tool_package(
    tool_name: str, tool_description: str, properties: dict
) -> dict:
    """
    Create a tool package for use with LLM integrations.

    Parameters:
    - tool_name (str): Name of the tool.
    - tool_description (str): Description of what the tool does.
    - properties (dict): Dictionary of properties describing the tool's parameters.

    Returns:
    - dict: A dictionary containing the tool package.
    """
    return {
        "name": tool_name,
        "description": tool_description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        },
    }


if __name__ == "__main__":
    user_prompt = "how old is russia"

    # print(call_gemini(user_prompt))
    view = gr.Interface(
        fn=stream_model,
        inputs=[
            gr.Textbox(label="Your message:"),
            gr.Dropdown(["GPT", "Claude", "Gemini"], label="Select model", value="GPT"),
        ],
        outputs=[gr.Markdown(label="Response:")],
        flagging_mode="never",
    )
    # view.launch()
    # print(call_gemini(user_prompt))
    # print(call_gpt(user_prompt, json_format=True))

    # print(call_claude(user_prompt))
