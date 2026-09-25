from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base
)
completion = client.completions.create(
    model = "Qwen/Qwen3-1.7B",
    prompt = "San Francisco is a"
)
print("Completion result:", completion)

chat_response = client.chat.completions.create(
    model = "Qwen/Qwen3-1.7B",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
)
print("Chat response: ", chat_response)