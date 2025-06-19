import json
import os
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


load_dotenv()

client = AsyncOpenAI()
sync_client = OpenAI()


def llm_call(prompt: str,  model: str = "gpt-4.1-mini") -> str:
    messages = []
    messages.append({"role": "user", "content": prompt})
    chat_completion = sync_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return chat_completion.choices[0].message.content


async def llm_call_async(prompt: str,  model: str = "gpt-4.1-mini") -> str:
    messages = []
    messages.append({"role": "user", "content": prompt})
    chat_completion = await client.chat.completions.create(
        model=model,
        messages=messages,
    )
    # print(model,"완료")
    return chat_completion.choices[0].message.content


def ollama_call(prompt: str, model: str = "exaone3.5:latest", return_obj: bool = False) -> tuple | str:
    url = f"{os.getenv('OLLAMA_ENDPOINT', 'localhost')}/api/chat"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()

    json_objs = response.content.decode().strip().split("\n")
    data = [json.loads(obj) for obj in json_objs]
    res_text, final_socket = "", {}
    for socket in data:
        message = socket.get("message", {})
        done = socket.get("done", False)
        if message.get("role") == "assistant":
            res_text += message.get("content", "")
        if done:
            final_socket = socket.copy()
            del final_socket["message"]
    if return_obj:
        return res_text.strip(), final_socket
    return res_text.strip()


if __name__ == "__main__":
    test, _ = ollama_call(prompt="안녕")
    print(test)

__all__ = ["llm_call", "llm_call_async", "ollama_call"]