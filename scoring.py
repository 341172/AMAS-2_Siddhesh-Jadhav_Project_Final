"""Transparent, rule-based lead scoring for LeadSense.

The rules decide the tier. AI can only add a small, capped adjustment on top
(see ai.py), so the score stays explainable and works even if Gemini is down.
"""
from dataclasses import dataclass, field

INDUSTRY_POINTS = {
    "D2C / E-commerce brand": 15,
    "EdTech": 12,
    "SaaS / Subscriptions": 12,
    "Travel & Hospitality": 10,
    "Online marketplace": 10,
    "Offline retail going online": 5,
    "Other": 3,
}
SENIORITY_POINTS = {
    "Founder / CXO": 7,
    "VP / Head": 6,
    "Manager": 4,
    "Individual contributor": 1,
}
SETUP_POINTS = {
    "Mostly COD / no gateway": 5,
    "Using a competitor gateway": 3,
    "Built in-house": 1,
}
SOURCE_POINTS = {
    "Referral": 7,
    "Inbound form": 5,
    "Event": 4,
    "Paid ads": 3,
    "Cold outbound": 1,
}
COMPETITOR_DOMAINS = {"razorpay.com", "cashfree.com", "payu.in", "phonepe.com",
                      "paytm.com", "juspay.in", "ccavenue.com"}
PERSONAL_DOMAINS = {"gmail.com", "yahoo.com", "yahoo.co.in", "hotmail.com",
                    "outlook.com", "rediffmail.com", "icloud.com"}
STUDENT_WORDS = ("student", "intern", "trainee", "fresher")
MIN_MONTHLY_ORDERS = 100
RECENCY_HALF_LIFE_DAYS = 30
BORDERLINE_BAND = 3


@dataclass
class Settings:
    fit_weight: float = 0.5   # share of the final score that comes from fit
    hot_cutoff: int = 70
    warm_cutoff: int = 45


@dataclass
class Result:
    fit: float
    engagement_raw: float
    engagement: float
    decay: float
    score: int
    tier: str
    reasons: list = field(default_factory=list)   # (label, points) pairs
    flags: list = field(default_factory=list)     # warnings for the rep
    disqualified_because: str = ""
    borderline: bool = False


def _order_points(orders: int) -> int:
    if orders >= 50000:
        return 15
    if orders >= 10000:
        return 12
    if orders >= 2000:
        return 8
    if orders >= 500:
        return 4
    return 0


def _size_points(employees: int) -> int:
    if 50 <= employees <= 500:
        return 8
    if employees > 500:
        return 6
    if employees >= 11:
        return 5
    return 2


def email_domain(email: str) -> str:
    return email.strip().lower().rsplit("@", 1)[-1] if "@" in str(email) else ""


def tier_for(score: float, s: Settings) -> str:
    if score >= s.hot_cutoff:
        return "Hot"
    if score >= s.warm_cutoff:
        return "Warm"
    return "Cold"


def score_lead(lead: dict, s: Settings = Settings()) -> Result:
    reasons = []

    # ---- Fit: is this the right kind of company and person? (max 50)
    fit_parts = [
        (f"Industry: {lead['industry']}", INDUSTRY_POINTS.get(lead["industry"], 3)),
        (f"{int(lead['monthly_orders']):,} online orders a month", _order_points(int(lead["monthly_orders"]))),
        (f"{int(lead['employees'])} employees", _size_points(int(lead["employees"]))),
        (f"Contact is {lead['seniority']}", SENIORITY_POINTS.get(lead["seniority"], 1)),
        (f"Today: {lead['current_setup']}", SETUP_POINTS.get(lead["current_setup"], 1)),
    ]
    fit = sum(p for _, p in fit_parts)
    reasons += [(f"Fit · {l}", p) for l, p in fit_parts]

    # ---- Engagement: are they actually interested right now? (max 50)
    eng_parts = []
    if lead.get("demo_requested"):
        eng_parts.append(("Requested a demo", 12))
    if lead.get("trial_started"):
        eng_parts.append(("Started a sandbox trial", 10))
    pv = min(int(lead.get("pricing_views", 0)), 4)
    if pv:
        eng_parts.append((f"Viewed pricing {int(lead['pricing_views'])}x", pv * 2))
    visits = min(int(lead.get("site_visits", 0)), 10)
    if visits:
        eng_parts.append((f"{int(lead['site_visits'])} site visits in 30 days", visits * 0.5))
    clicks = min(int(lead.get("email_clicks", 0)), 5)
    if clicks:
        eng_parts.append((f"Clicked {int(lead['email_clicks'])} emails", clicks))
    if lead.get("webinar_attended"):
        eng_parts.append(("Attended a webinar", 3))
    eng_parts.append((f"Source: {lead['source']}", SOURCE_POINTS.get(lead["source"], 1)))
    engagement_raw = sum(p for _, p in eng_parts)

    days = max(0, int(lead.get("days_since_last_activity", 0)))
    decay = 0.5 ** (days / RECENCY_HALF_LIFE_DAYS)
    engagement = engagement_raw * decay
    reasons += [(f"Engagement · {l}", round(p * decay, 1)) for l, p in eng_parts]

    score = s.fit_weight * (fit / 50 * 100) + (1 - s.fit_weight) * (engagement / 50 * 100)
    score = int(round(max(0, min(100, score))))
    tier = tier_for(score, s)

    # ---- Hard rules and flags
    flags, dq = [], ""
    domain = email_domain(lead.get("email", ""))
    title = str(lead.get("job_title", "")).lower()
    if domain in COMPETITOR_DOMAINS:
        dq = f"Email domain {domain} belongs to a competitor."
    elif any(w in title for w in STUDENT_WORDS):
        dq = "Contact looks like a student or intern, not a buyer."
    elif int(lead["monthly_orders"]) < MIN_MONTHLY_ORDERS:
        dq = f"Under {MIN_MONTHLY_ORDERS} orders a month: too small for Checkoutly today."
    if dq:
        tier = "Disqualified"
    if domain in PERSONAL_DOMAINS:
        flags.append("Personal email address: confirm the company before investing time.")
    if days > 60:
        flags.append(f"No activity for {days} days: engagement now counts at {decay:.0%}.")

    borderline = (not dq) and any(abs(score - c) <= BORDERLINE_BAND for c in (s.hot_cutoff, s.warm_cutoff))
    if borderline:
        flags.append("Borderline: within 3 points of a tier cut-off. A small change in the data can flip the tier.")

    return Result(fit=fit, engagement_raw=engagement_raw, engagement=engagement, decay=decay,
                  score=score, tier=tier, reasons=reasons, flags=flags,
                  disqualified_because=dq, borderline=borderline)


def apply_ai_adjustment(result: Result, adjustment: int, s: Settings) -> tuple:
    """Return (final_score, final_tier) after a capped AI adjustment. Disqualified stays disqualified."""
    if result.tier == "Disqualified":
        return result.score, "Disqualified"
    final = int(max(0, min(100, result.score + adjustment)))
    return final, tier_for(final, s)
