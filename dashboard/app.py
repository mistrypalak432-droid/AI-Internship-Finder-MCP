import streamlit as st
import pandas as pd
import anyio
import sys
import json
import os

from pathlib import Path
from groq import Groq
from mcp import Client, StdioServerParameters


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Internship Finder",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "internship_data.csv"

df = pd.read_csv(CSV_PATH)


# ============================================================
# GROQ CLOUD AI
# ============================================================

groq_api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

groq_client = Groq(
    api_key=groq_api_key
)

MODEL = "qwen/qwen3.8-27b"


# ============================================================
# MCP SERVER CONFIGURATION
# ============================================================

server = StdioServerParameters(
    command=sys.executable,
    args=["server/server.py"]
)


# ============================================================
# CLEAN AI RESPONSE
# ============================================================

def clean_answer(answer):

    if not answer:
        return ""

    answer = str(answer)

    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]

    if "<think>" in answer:
        answer = answer.split("<think>", 1)[1]

    return answer.strip()


# ============================================================
# CONVERT MCP TO GROQ TOOL FORMAT
# ============================================================

def convert_mcp_tools_to_groq(mcp_tools):

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


# ============================================================
# GROQ AI CALL
# ============================================================

def ask_groq(messages, tools=None, tool_choice=None):

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0
    )

    return response


# ============================================================
# FINAL AI RESPONSE
# ============================================================

def generate_final_answer(question, mcp_result):

    response = ask_groq(

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an internship database assistant. "

                    "Use ONLY the database result provided below. "

                    "If multiple internships are present, "
                    "show all returned internships. "

                    "Do not select only one internship. "

                    "Do not say best, better, or worst. "

                    "Do not make subjective judgments. "

                    "Do not invent information. "

                    "For each internship, show the available "
                    "company, job title, location, duration, "
                    "stipend, and match score. "

                    "Use a numbered list when multiple "
                    "internships are returned. "

                    "The match score is calculated by the "
                    "rule-based recommendation system. "

                    "Do not show reasoning. "

                    "Do not show analysis. "

                    "Do not use <think> tags. "

                    "Keep the answer clear and concise."
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

        # IMPORTANT:
        # Final answer must NOT call another MCP tool
        tools=None,
        tool_choice="none"
    )

    return clean_answer(
        response.choices[0].message.content
    )


# ============================================================
# AI + MCP FUNCTION
# ============================================================

async def ask_ai_mcp(question):

    async with Client(server) as client:

        # ----------------------------------------------------
        # Discover MCP tools
        # ----------------------------------------------------

        tool_result = await client.list_tools()

        groq_tools = convert_mcp_tools_to_groq(
            tool_result.tools
        )

        # ----------------------------------------------------
        # Ask Groq/Qwen which MCP tool to use
        # ----------------------------------------------------

        response = ask_groq(

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
                        "for internships based on their skills, "
                        "interests, or background. "

                        "For skills, use comma-separated values. "

                        "Example: "
                        "'Python and Data Science' means "
                        "'Python, Data Science'. "

                        "Always call the appropriate MCP tool. "

                        "Do not answer the user directly."
                    )
                },

                {
                    "role": "user",
                    "content": question
                }
            ],

            tools=groq_tools,
            tool_choice="auto"
        )

        # ----------------------------------------------------
        # Get tool calls
        # ----------------------------------------------------

        message = response.choices[0].message

        tool_calls = message.tool_calls

        # ----------------------------------------------------
        # Fallback if model returns JSON as text
        # ----------------------------------------------------

        if not tool_calls:

            content = clean_answer(
                message.content
            )

            try:

                data = json.loads(content)

                if (
                    isinstance(data, dict)
                    and data.get("name")
                    and data.get("arguments")
                ):

                    tool_name = data["name"]

                    arguments = data["arguments"]

                    result = await client.call_tool(
                        tool_name,
                        arguments
                    )

                    if result.content:

                        mcp_result = result.content[0].text

                    else:

                        mcp_result = "No results were returned."

                    final_answer = generate_final_answer(
                        question,
                        mcp_result
                    )

                    return final_answer, tool_name

            except (json.JSONDecodeError, TypeError):

                pass

            return clean_answer(
                message.content
            ), None

        # ----------------------------------------------------
        # Execute MCP tool
        # ----------------------------------------------------

        answers = []

        selected_tool = None

        for call in tool_calls:

            selected_tool = call.function.name

            arguments = call.function.arguments

            # Groq normally returns JSON string arguments
            if isinstance(arguments, str):

                try:

                    arguments = json.loads(arguments)

                except json.JSONDecodeError:

                    arguments = {}

            # ------------------------------------------------
            # Call MCP server
            # ------------------------------------------------

            result = await client.call_tool(
                selected_tool,
                arguments
            )

            # ------------------------------------------------
            # Get MCP result
            # ------------------------------------------------

            if result.content:

                mcp_result = result.content[0].text

            else:

                mcp_result = "No results were returned."

            # ------------------------------------------------
            # Generate final answer with Groq
            # ------------------------------------------------

            final_answer = generate_final_answer(
                question,
                mcp_result
            )

            answers.append(final_answer)

        return "\n\n".join(answers), selected_tool


# ============================================================
# STREAMLIT HEADER
# ============================================================

st.title("🎓 AI Internship Finder")

st.write(
    "Find and explore internships using AI, MCP tools, "
    "and internship data."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Filters")


# ------------------------------------------------------------
# Job role
# ------------------------------------------------------------

roles = sorted(
    df["job_title"]
    .dropna()
    .unique()
)

selected_role = st.sidebar.selectbox(
    "Job Role",
    ["All"] + roles
)


# ------------------------------------------------------------
# Location
# ------------------------------------------------------------

locations = sorted(
    df["location"]
    .dropna()
    .unique()
)

selected_location = st.sidebar.selectbox(
    "Location",
    ["All"] + locations
)


# ------------------------------------------------------------
# Minimum stipend
# ------------------------------------------------------------

min_stipend = st.sidebar.number_input(
    "Minimum Stipend",
    min_value=0.0,
    value=0.0,
    step=1000.0
)


# ============================================================
# APPLY DASHBOARD FILTERS
# ============================================================

filtered_df = df.copy()


if selected_role != "All":

    filtered_df = filtered_df[
        filtered_df["job_title"] == selected_role
    ]


if selected_location != "All":

    filtered_df = filtered_df[
        filtered_df["location"] == selected_location
    ]


if min_stipend > 0:

    filtered_df = filtered_df[
        filtered_df["max_stipend"] >= min_stipend
    ]


# ============================================================
# STATISTICS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Internships",
        len(filtered_df)
    )


with col2:

    st.metric(
        "Companies",
        filtered_df["company_name"].nunique()
    )


with col3:

    if len(filtered_df) > 0:

        average_stipend = (
            filtered_df["max_stipend"].mean()
        )

    else:

        average_stipend = 0

    st.metric(
        "Average Stipend",
        f"₹{average_stipend:,.0f}"
    )


# ============================================================
# AI ASSISTANT
# ============================================================

st.divider()

st.header("🤖 AI Internship Assistant")

st.write(
    "Ask the AI to search, filter, find details, "
    "or recommend internships."
)


# ------------------------------------------------------------
# Example questions
# ------------------------------------------------------------

st.caption(
    "Try: "
    "\"Find Python internships\" • "
    "\"Find internships in Ahmedabad with stipend above 10000\" • "
    "\"Recommend internships for Python and Data Science\""
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ------------------------------------------------------------
# Display previous messages
# ------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about internships..."
)


if question:

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "AI is searching internships..."
        ):

            try:

                answer, selected_tool = anyio.run(
                    ask_ai_mcp,
                    question
                )

                if selected_tool:

                    st.caption(
                        f"MCP tool used: `{selected_tool}`"
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = (
                    "Something went wrong while connecting "
                    f"to the AI/MCP server:\n\n`{str(e)}`"
                )

                st.error(error_message)


# ============================================================
# INTERNSHIP RESULTS
# ============================================================

st.divider()

st.subheader("📋 Internship Results")


if filtered_df.empty:

    st.warning(
        "No internships match the selected filters."
    )

else:

    display_df = filtered_df[
        [
            "company_name",
            "job_title",
            "location",
            "duration",
            "max_stipend"
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "company_name": "Company",
            "job_title": "Job Title",
            "location": "Location",
            "duration": "Duration",
            "max_stipend": "Stipend"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANALYTICS
# ============================================================

st.subheader("📊 Internship Analytics")


col1, col2 = st.columns(2)


with col1:

    st.write("### Internships by Location")

    location_counts = (
        filtered_df["location"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        location_counts
    )


with col2:

    st.write("### Internships by Job Role")

    role_counts = (
        filtered_df["job_title"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        role_counts
    )