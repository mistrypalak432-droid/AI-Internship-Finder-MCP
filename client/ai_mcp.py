import anyio
import sys
import json
import os

from groq import Groq
from mcp import Client, StdioServerParameters


# ==================================================
# Cloud AI
# ==================================================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "qwen/qwen3.8-27b"


# ==================================================
# Connect to MCP server
# ==================================================

server = StdioServerParameters(
    command=sys.executable,
    args=["server/server.py"]
)


# ==================================================
# Convert MCP tools to standard tool format
# ==================================================

def convert_mcp_tools(mcp_tools):

    tools = []

    for tool in mcp_tools:

        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            }
        })

    return tools


# ==================================================
# Remove thinking output if present
# ==================================================

def clean_answer(answer):

    if not answer:
        return ""

    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]

    return answer.strip()


# ==================================================
# AI call
# ==================================================

def ask_ai(messages, tools=None, tool_choice=None):

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0
    )

    return response


# ==================================================
# Main
# ==================================================

async def main():

    async with Client(server) as client:

        # --------------------------------------------------
        # Discover MCP tools
        # --------------------------------------------------

        tool_result = await client.list_tools()

        print("\nMCP tools discovered:")

        for tool in tool_result.tools:
            print("-", tool.name)

        # --------------------------------------------------
        # Convert MCP tools
        # --------------------------------------------------

        mcp_tools = convert_mcp_tools(
            tool_result.tools
        )

        # --------------------------------------------------
        # Continuous chat
        # --------------------------------------------------

        while True:

            question = input("\nYou: ")

            if question.lower().strip() in ["exit", "quit"]:
                print("Exiting...")
                break

            # --------------------------------------------------
            # Ask AI to select MCP tool
            # --------------------------------------------------

            response = ask_ai(

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an internship assistant. "

                            "Choose the most appropriate MCP tool "
                            "for the user's request. "

                            "Use search for simple keyword searches. "

                            "Use filter when the user specifies "
                            "role, location, stipend, duration, "
                            "or multiple filters. "

                            "Use details when the user asks about "
                            "a specific internship ID. "

                            "Use recommendation when the user asks "
                            "for internships based on their skills. "

                            "For skills, always use comma-separated "
                            "values. "

                            "Example: "
                            "'Python and Data Science' means "
                            "'Python, Data Science'. "

                            "Always call the appropriate tool. "
                            "Do not answer the user directly."
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],

                tools=mcp_tools,
                tool_choice="auto"
            )

            # --------------------------------------------------
            # Check tool call
            # --------------------------------------------------

            message = response.choices[0].message
            tool_calls = message.tool_calls

            # --------------------------------------------------
            # If AI selected a tool
            # --------------------------------------------------

            if tool_calls:

                for call in tool_calls:

                    tool_name = call.function.name

                    arguments = json.loads(
                        call.function.arguments
                    )

                    print("\nAI selected:")
                    print("Tool:", tool_name)
                    print("Arguments:", arguments)

                    # --------------------------------------------------
                    # Call MCP tool
                    # --------------------------------------------------

                    result = await client.call_tool(
                        tool_name,
                        arguments
                    )

                    # --------------------------------------------------
                    # Get MCP result safely
                    # --------------------------------------------------

                    if result.content:

                        mcp_result = result.content[0].text

                    else:

                        mcp_result = "No results were returned."


                    # --------------------------------------------------
                    # Ask AI for final answer
                    # IMPORTANT:
                    # No MCP tools are provided here.
                    # Therefore tool_choice = "none"
                    # --------------------------------------------------

                    final_response = ask_ai(

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer the user's request using "
                                    "ONLY the database result below. "

                                    "Do not add information that is "
                                    "not present in the database result. "

                                    "Do not show reasoning or analysis. "

                                    "Do not show thinking. "

                                    "Do not make subjective judgments. "

                                    "Do not say best, better, or worst. "

                                    "Show internships as a numbered list. "

                                    "For each internship, show company, "
                                    "job title, location, duration, "
                                    "stipend, and match score if available. "

                                    "Keep the answer concise."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"User request: {question}\n\n"
                                    f"Database result:\n{mcp_result}"
                                )
                            }
                        ],

                        # VERY IMPORTANT
                        tools=None,
                        tool_choice="none"
                    )

                    answer = clean_answer(
                        final_response.choices[0].message.content
                    )

                    print("\nAI:")
                    print(answer)

            else:

                answer = clean_answer(
                    message.content
                )

                print("\nAI:")
                print(answer)


# ==================================================
# Start program
# ==================================================

if __name__ == "__main__":
    anyio.run(main)