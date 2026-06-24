"""
utils/skill_extractor.py
Extracts skills from a Job Description by matching against a predefined skill database.
"""

# ── Skill Database ─────────────────────────────────────────────────────────────
# Covers the skills that appear in candidate_resume_dataset.csv plus common extras.
SKILL_DATABASE = [
    # Programming languages
    "python", "r", "java", "scala", "javascript", "bash",
    # Data / ML frameworks
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "spark", "hadoop",
    # ML / AI concepts
    "machine learning", "deep learning", "nlp", "natural language processing",
    "feature engineering", "statistics", "ai",
    # Cloud & infrastructure
    "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
    "ci/cd", "rest api",
    # Data engineering
    "sql", "etl", "airflow", "dbt", "mlflow",
    # BI / Visualisation
    "tableau", "power bi", "looker", "excel", "data visualization",
    # Other
    "node.js", "react", "backend",
]


def extract_skills_from_jd(job_description: str) -> list[str]:
    """
    Return a list of skills found in *job_description* (case-insensitive).

    Parameters
    ----------
    job_description : str
        Raw JD text entered by the recruiter.

    Returns
    -------
    list[str]
        Deduplicated list of matched skill strings (original casing from DB).
    """
    if not job_description or not job_description.strip():
        return []

    jd_lower = job_description.lower()
    matched: list[str] = []

    for skill in SKILL_DATABASE:
        # Simple substring match; keeps multi-word skills working correctly
        if skill in jd_lower:
            matched.append(skill)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in matched:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def normalise_skills(raw_skills: str) -> list[str]:
    """
    Split a comma-separated skills string from the CSV into a normalised list.

    Parameters
    ----------
    raw_skills : str
        e.g. "Python, SQL, Machine Learning"

    Returns
    -------
    list[str]
        Lower-cased, stripped skill tokens.
    """
    if not raw_skills or not isinstance(raw_skills, str):
        return []
    return [s.strip().lower() for s in raw_skills.split(",") if s.strip()]
