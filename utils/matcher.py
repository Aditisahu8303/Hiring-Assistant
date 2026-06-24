"""
utils/matcher.py
Computes candidate–JD match scores and produces a ranked DataFrame.
"""

import pandas as pd
from utils.skill_extractor import normalise_skills


def compute_match_score(candidate_skills_raw: str, jd_skills: list[str]) -> float:
    """
    Calculate the match score for a single candidate.

    Formula
    -------
        Match Score = (Matched Skills / Total JD Skills) × 100

    Parameters
    ----------
    candidate_skills_raw : str
        Comma-separated skills string from the CSV (e.g. "Python, SQL, AWS").
    jd_skills : list[str]
        Lower-cased skills extracted from the Job Description.

    Returns
    -------
    float
        Score in the range [0.0, 100.0], rounded to 2 decimal places.
        Returns 0.0 when jd_skills is empty.
    """
    if not jd_skills:
        return 0.0

    candidate_skills = set(normalise_skills(candidate_skills_raw))
    jd_skills_set = set(jd_skills)

    matched = candidate_skills & jd_skills_set
    score = (len(matched) / len(jd_skills_set)) * 100
    return round(score, 2)


def rank_candidates(df: pd.DataFrame, jd_skills: list[str]) -> pd.DataFrame:
    """
    Score and rank all candidates in *df* against the extracted JD skills.

    Parameters
    ----------
    df : pd.DataFrame
        The full candidate dataset (must include columns: Candidate_ID,
        Candidate_Name, Current_Role, Experience_Years, Skills).
    jd_skills : list[str]
        Lower-cased skills from the JD (output of skill_extractor).

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with columns:
        Rank | Candidate_ID | Candidate_Name | Current_Role |
        Experience_Years | Match_Score
    """
    required_cols = {"Candidate_ID", "Candidate_Name", "Current_Role",
                     "Experience_Years", "Skills"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["Match_Score"] = df["Skills"].apply(
        lambda skills: compute_match_score(skills, jd_skills)
    )

    ranked = (
        df.sort_values("Match_Score", ascending=False)
        .reset_index(drop=True)
    )
    ranked.index += 1          # 1-based ranking
    ranked.index.name = "Rank"

    result = ranked[
        ["Candidate_ID", "Candidate_Name", "Current_Role",
         "Experience_Years", "Match_Score"]
    ].reset_index()             # brings Rank back as a column

    return result
