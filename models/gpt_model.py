from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


load_dotenv()

client = AsyncOpenAI()
sync_client = OpenAI()


def gpt_call(prompt: str,  model: str = "gpt-4.1-mini") -> str:
    messages = []
    messages.append({"role": "user", "content": prompt})
    chat_completion = sync_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return chat_completion.choices[0].message.content


async def gpt_call_async(prompt: str,  model: str = "gpt-4.1-mini") -> str:
    messages = []
    messages.append({"role": "user", "content": prompt})
    chat_completion = await client.chat.completions.create(
        model=model,
        messages=messages,
    )
    # print(model,"완료")
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    test, _ = gpt_call(prompt="안녕")
    print(test)

__all__ = ["gpt_call", "gpt_call_async"]