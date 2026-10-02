from mcp.server import MCPServer

from internship_tools import (
    search_internships,
    filter_internships,
    get_internship_details,
    recommend_internships
)


# Create MCP server
mcp = MCPServer("Internship Finder")


# --------------------------------------------------
# Tool 1: Search internships
# --------------------------------------------------

@mcp.tool()
def search_internships_tool(keyword: str) -> str:
    """
    Use ONLY for simple keyword searches by job title.

    Example:
    - "Find Python internships"
    - "Search for Data Science internships"

    Do NOT use this tool when the user specifies:
    location, stipend, duration, or multiple filters.
    """

    result = search_internships(keyword)

    if result.empty:
        return f"No internships found for '{keyword}'."

    # Return important columns only
    result = result[
        [
            "company_name",
            "location",
            "duration",
            "job_title",
            "max_stipend"
        ]
    ]

    # Limit results
    result = result.head(10)

    return result.to_json(orient="records")


# --------------------------------------------------
# Tool 2: Filter internships
# --------------------------------------------------

@mcp.tool()
def filter_internships_tool(
    role: str = None,
    location: str = None,
    min_stipend: float = None,
    max_stipend: float = None,
    duration: str = None
) -> str:
    """
    Use this tool when the user gives ANY filter.

    Use it for:
    - role
    - location
    - minimum stipend
    - maximum stipend
    - duration

    IMPORTANT:
    Use this tool when the user gives MULTIPLE conditions.

    Examples:
    - "Python internships in Ahmedabad"
    - "Internships with stipend above 5000"
    - "Python internships in Work from home with stipend above 5000"
    - "Data Science internships for 2 months"

    Do NOT use search_internships_tool when filters are present.
    """

    result = filter_internships(
        role=role,
        location=location,
        min_stipend=min_stipend,
        max_stipend=max_stipend,
        duration=duration
    )

    if result.empty:
        return "No internships match the given filters."

    # Return only important information
    result = result[
        [
            "company_name",
            "location",
            "duration",
            "job_title",
            "max_stipend"
        ]
    ]

    # Limit results
    result = result.head(10)

    return result.to_json(orient="records")


# --------------------------------------------------
# Tool 3: Get internship details
# --------------------------------------------------

@mcp.tool()
def get_internship_details_tool(internship_id: int) -> str:
    """
    Use ONLY when the user asks for details of a
    specific internship using its internship ID.

    Example:
    - "Show details of internship 9"
    - "Give me details for internship ID 25"
    """

    result = get_internship_details(internship_id)

    if result is None:
        return f"No internship found with ID {internship_id}."

    return result.to_json()


# --------------------------------------------------
# Tool 4: Recommend internships
# --------------------------------------------------

@mcp.tool()
def recommend_internships_tool(
    skills: str,
    location: str = None,
    min_stipend: float = None,
    top_k: int = 5
) -> str:
    """
    Use this tool when the user asks for RECOMMENDATIONS
    based on their skills, interests, or background.

    Examples:
    - "Recommend internships for Python and Data Science"
    - "Which internships match my Python skills?"
    - "Recommend Data Science internships in Ahmedabad"

    Do NOT use this tool for simple keyword searches
    or ordinary filtering.
    """

    result = recommend_internships(
        skills=skills,
        location=location,
        min_stipend=min_stipend,
        top_k=top_k
    )

    if result.empty:
        return "No suitable internships found."

    # Return important information
    result = result[
        [
            "company_name",
            "location",
            "duration",
            "job_title",
            "max_stipend",
            "match_score"
        ]
    ]

    return result.to_json(orient="records")


# --------------------------------------------------
# Start MCP server
# --------------------------------------------------

if __name__ == "__main__":
    mcp.run()