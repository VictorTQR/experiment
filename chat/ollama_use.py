from langchain_ollama import ChatOllama
from langchain.agents import create_agent



model = ChatOllama(model="qwen3:0.6b", base_url="http://soct.top:11436", reasoning=True)
def test_ollama_nonstream():
    print(model.invoke("你好"))

def test_ollama_stream():
    for chunk in model.stream("你好"):
        if chunk.content_blocks:
            if chunk.content_blocks[0]["type"] == "text":
                print(chunk.content_blocks[0]["text"], end="")
            if chunk.content_blocks[0]["type"] == "reasoning":
                print(chunk.content_blocks[0]["reasoning"], end="")


def get_weather(city: str) -> str:
    """获取城市的天气"""
    return f"{city}的天气是晴朗的"

agent = create_agent(
    model,
    tools=[get_weather]
)

def test_agent():
    print(agent.invoke({"messages": [{"role": "user", "content": "北京的天气"}] }))

def test_agent_stream():
    for token, metadata in agent.stream({"messages": [{"role": "user", "content": "北京的天气"}] }, stream_mode="messages"):
        if token.type == "AIMessageChunk":
            if token.content_blocks:
                content = token.content_blocks[0]
                if content["type"] == "text":
                    print(content["text"])
                if content["type"] == "reasoning":
                    print(content["reasoning"])
                if content["type"] == "tool_call_chunk":
                    print(f"工具调用: \n 工具名：{content["name"]}\n 工具参数：{content["args"]}")
        if token.type == "tool":
            print(f"工具结果: {token.content}")

if __name__ == "__main__":
    # test_ollama_stream()
    # test_agent()
    test_agent_stream()