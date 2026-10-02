from ollama import chat

tools_description = """
You have access to these internship tools:

1. search_internships_tool
   - Use when the user wants to search internships by a keyword or job title.
   - Example: "Find Python internships"

2. filter_internships_tool
   - Use when the user wants to filter internships by role, location,
     stipend, or duration.
   - Example: "Find Python internships in Ahmedabad with stipend above 5000"

3. get_internship_details_tool
   - Use when the user asks for details about a specific internship.

4. recommend_internships_tool
   - Use when the user wants internship recommendations based on skills,
     location, or minimum stipend.
"""

question = input("You: ")

response = chat(
    model="phi3:mini",
    messages=[
        {
            "role": "system",
            "content": tools_description
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

print("\nAI:")
print(response.message.content)