import anyio
import sys

from mcp import Client, StdioServerParameters


server = StdioServerParameters(
    command=sys.executable,
    args=["server/server.py"]
)


async def main():

    async with Client(server) as client:

        print("Connected to MCP server!")

        # Discover tools
        result = await client.list_tools()

        print("\nAvailable tools:")
        for tool in result.tools:
            print(f"- {tool.name}")

        print("\nSearching for Python internships...")

        response = await client.call_tool(
            "search_internships_tool",
            {
                "keyword": "Python"
            }
        )

        print("\nSearch result:")
        print(response)

        # Call search tool
        print("\nFiltering internships...")

        response = await client.call_tool(
            "filter_internships_tool",
            {
                "role": "Python",
                "location": "Work from home"
            }
        )

        print("\nFilter result:")
        print(response)
        
        print("\nGetting internship details...")

        response = await client.call_tool(
            "get_internship_details_tool",
            {
                "internship_id": 9
            }
        )

        print("\nInternship details:")
        print(response)
        
        print("\nGetting recommendations...")

        response = await client.call_tool(
            "recommend_internships_tool",
            {
                "skills": "Python, Data Science",
                "location": "Work from home",
                "min_stipend": 0,
                "top_k": 5
            }
        )

        print("\nRecommendation result:")
        print(response)

if __name__ == "__main__":
    anyio.run(main)