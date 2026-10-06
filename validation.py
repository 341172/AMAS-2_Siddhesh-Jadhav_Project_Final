"""Input checks that run before anything is scored or sent to the AI."""
import re

import pandas as pd

from scoring import INDUSTRY_POINTS, SENIORITY_POINTS, SETUP_POINTS, SOURCE_POINTS

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
MAX_NOTES = 2000

REQUIRED_COLUMNS = [
    "company", "contact_name", "email", "job_title", "industry", "monthly_orders", "employees",
    "seniority", "current_setup", "source", "demo_requested", "trial_started", "pricing_views",
    "site_visits", "email_clicks", "webinar_attended", "days_since_last_activity",
]
OPTIONAL_COLUMNS = ["lead_id", "notes"]

RANGES = {
    "monthly_orders": (0, 10_000_000),
    "employees": (1, 1_000_000),
    "pricing_views": (0, 100),
    "site_visits": (0, 1000),
    "email_clicks": (0, 200),
    "days_since_last_activity": (0, 365),
}
CHOICES = {
    "industry": INDUSTRY_POINTS,
    "seniority": SENIORITY_POINTS,
    "current_setup": SETUP_POINTS,
    "source": SOURCE_POINTS,
}
YES = {"1", "true", "yes", "y"}
NO = {"0", "false", "no", "n", ""}


def check_lead(lead: dict) -> list:
    """Return a list of plain-language problems. Empty list means the lead is fine."""
    errors = []
    for f in ("company", "contact_name", "email", "job_title"):
        if not str(lead.get(f, "")).strip():
            errors.append(f"{f.replace('_', ' ').capitalize()} is required.")
    email = str(lead.get("email", "")).strip()
    if email and not EMAIL_RE.match(email):
        errors.append(f"'{email}' is not a valid email address.")
    for f, (lo, hi) in RANGES.items():
        try:
            v = float(lead.get(f))
        except (TypeError, ValueError):
            errors.append(f"{f.replace('_', ' ').capitalize()} must be a number.")
            continue
        if v != int(v):
            errors.append(f"{f.replace('_', ' ').capitalize()} must be a whole number.")
        elif not lo <= v <= hi:
            errors.append(f"{f.replace('_', ' ').capitalize()} must be between {lo:,} and {hi:,} (got {int(v):,}).")
    for f, options in CHOICES.items():
        if lead.get(f) not in options:
            errors.append(f"{f.replace('_', ' ').capitalize()} '{lead.get(f)}' is not one of: {', '.join(options)}.")
    if len(str(lead.get("notes", ""))) > MAX_NOTES:
        errors.append(f"Notes are too long ({len(str(lead['notes']))} characters). Keep them under {MAX_NOTES}.")
    return errors


def _to_bool(v):
    s = str(v).strip().lower()
    if s in YES:
        return True
    if s in NO or s == "nan":
        return False
    return None


def clean_csv(df: pd.DataFrame):
    """Validate an uploaded CSV. Returns (clean_rows, problems, fatal_message)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [], [], "The file is missing these columns: " + ", ".join(missing) + ". Download the template to see the format."
    if len(df) == 0:
        return [], [], "The file has headers but no leads."
    if len(df) > 2000:
        return [], [], f"The file has {len(df):,} rows. Upload 2,000 leads or fewer at a time."

    rows, problems, seen = [], [], {}
    for i, raw in df.iterrows():
        line = i + 2  # header is line 1
        lead = {k: raw.get(k) for k in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
        lead["notes"] = "" if pd.isna(lead.get("notes")) else str(lead["notes"])
        lead["lead_id"] = "" if pd.isna(lead.get("lead_id")) else str(lead["lead_id"])
        for k in ("company", "contact_name", "email", "job_title", "industry", "seniority", "current_setup", "source"):
            lead[k] = "" if pd.isna(lead[k]) else str(lead[k]).strip()
        bad = []
        for k in ("demo_requested", "trial_started", "webinar_attended"):
            b = _to_bool(lead[k])
            if b is None:
                bad.append(f"{k} must be yes/no or 1/0")
            lead[k] = bool(b)
        errs = check_lead(lead) + bad
        key = lead["email"].lower()
        if key and key in seen:
            errs.append(f"duplicate of the lead on line {seen[key]} (same email)")
        if errs:
            problems.append(f"Line {line} ({lead['company'] or 'no company'}): " + "; ".join(errs))
            continue
        seen[key] = line
        for f in RANGES:
            lead[f] = int(float(lead[f]))
        rows.append(lead)
    return rows, problems, ""
