from pathlib import Path
import pandas as pd


# ==================================================
# Project paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "internship_data.csv"


# ==================================================
# Load internship data
# ==================================================

df = pd.read_csv(CSV_PATH)


# ==================================================
# Tool 1: Search internships
# ==================================================

def search_internships(keyword: str):

    keyword = keyword.lower().strip()

    result = df[
        df["job_title"]
        .fillna("")
        .str.lower()
        .str.contains(keyword, na=False)
    ]

    return result


# ==================================================
# Tool 2: Filter internships
# ==================================================

def filter_internships(
    role: str = None,
    location: str = None,
    min_stipend: float = None,
    max_stipend: float = None,
    duration: str = None
):

    result = df.copy()

    # Role
    if role:
        result = result[
            result["job_title"]
            .fillna("")
            .str.lower()
            .str.contains(role.lower(), na=False)
        ]

    # Location
    if location:
        result = result[
            result["location"]
            .fillna("")
            .str.lower()
            .str.contains(location.lower(), na=False)
        ]

    # Minimum stipend
    if min_stipend is not None:
        result = result[
            result["max_stipend"] >= min_stipend
        ]

    # Maximum stipend
    if max_stipend is not None:
        result = result[
            result["max_stipend"] <= max_stipend
        ]

    # Duration
    if duration:
        result = result[
            result["duration"]
            .fillna("")
            .str.lower()
            .str.contains(duration.lower(), na=False)
        ]

    return result


# ==================================================
# Tool 3: Get internship details
# ==================================================

def get_internship_details(internship_id: int):

    if internship_id < 0 or internship_id >= len(df):
        return None

    return df.iloc[internship_id]


# ==================================================
# Tool 4: Recommend internships
# ==================================================

def recommend_internships(
    skills: str,
    location: str = None,
    min_stipend: float = None,
    top_k: int = 5
):

    result = df.copy()

    # --------------------------------------------------
    # Convert user skills into individual skills
    # --------------------------------------------------

    skill_list = [
        skill.strip().lower()
        for skill in skills.split(",")
        if skill.strip()
    ]

    # --------------------------------------------------
    # Related job keywords for each skill
    # --------------------------------------------------

    skill_categories = {

        "python": [
            "python",
            "machine learning",
            "data science",
            "data analyst",
            "artificial intelligence",
            "ai engineer",
            "ai chatbot",
            "automation"
        ],

        "data science": [
            "data science",
            "data scientist",
            "data analyst",
            "machine learning",
            "artificial intelligence",
            "research analyst",
            "database"
        ],

        "machine learning": [
            "machine learning",
            "ml",
            "artificial intelligence",
            "ai engineer",
            "data science",
            "data scientist"
        ],

        "artificial intelligence": [
            "artificial intelligence",
            "ai",
            "ai engineer",
            "ai chatbot",
            "ai automation",
            "ai tools",
            "machine learning"
        ],

        "ai": [
            "artificial intelligence",
            "ai engineer",
            "ai chatbot",
            "ai automation",
            "ai tools",
            "machine learning"
        ],

        "sql": [
            "database",
            "data analyst",
            "data science"
        ],

        "database": [
            "database",
            "data management",
            "data analyst"
        ],

        "web development": [
            "web development",
            "frontend",
            "front end",
            "backend",
            "back end"
        ],

        "react": [
            "react",
            "frontend",
            "front end",
            "web development"
        ],

        "java": [
            "java",
            "software development"
        ]
    }

    # --------------------------------------------------
    # Calculate recommendation score
    # --------------------------------------------------

    def calculate_score(job_title):

        title = str(job_title).lower()

        score = 0

        for skill in skill_list:

            # Exact match = 2 points
            if skill in title:
                score += 2

            # Related match = 1 point
            else:

                related_keywords = skill_categories.get(
                    skill,
                    []
                )

                for keyword in related_keywords:

                    if keyword in title:
                        score += 1
                        break

        return score

    # Calculate scores
    result["match_score"] = result["job_title"].apply(
        calculate_score
    )

    # --------------------------------------------------
    # Location filter
    # --------------------------------------------------

    if location:

        result = result[
            result["location"]
            .fillna("")
            .str.lower()
            .str.contains(
                location.lower(),
                na=False
            )
        ]

    # --------------------------------------------------
    # Minimum stipend filter
    # --------------------------------------------------

    if min_stipend is not None:

        result = result[
            result["max_stipend"] >= min_stipend
        ]

    # --------------------------------------------------
    # Keep matching internships
    # --------------------------------------------------

    result = result[
        result["match_score"] > 0
    ]

    # --------------------------------------------------
    # Sort by recommendation score
    # --------------------------------------------------

    result = result.sort_values(
        by="match_score",
        ascending=False
    )

    # --------------------------------------------------
    # Return top recommendations
    # --------------------------------------------------

    return result.head(top_k)