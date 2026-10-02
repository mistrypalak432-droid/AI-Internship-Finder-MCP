from ollama import chat

response = chat(
    model="phi3:mini",
    messages=[
        {
            "role": "user",
            "content": "What is an internship?"
        }
    ]
)

print(response.message.content)