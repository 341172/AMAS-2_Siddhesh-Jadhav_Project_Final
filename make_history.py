"""Generate data/historical_leads.csv: 500 SYNTHETIC past leads with outcomes (won = 1/0).

The hidden conversion model is deliberately NOT the same as the app's scorecard
(it weights recent engagement more and penalises personal email), so the
"Does the score work?" tab has realistic gaps to show.
"""
import math
import random

import pandas as pd

from scoring import INDUSTRY_POINTS, SENIORITY_POINTS, SETUP_POINTS, SOURCE_POINTS

random.seed(7)
rows = []
for i in range(500):
    industry = random.choices(list(INDUSTRY_POINTS), weights=[30, 12, 12, 10, 10, 14, 12])[0]
    orders = int(random.lognormvariate(8.3, 1.4))
    employees = max(1, int(random.lognormvariate(3.6, 1.1)))
    seniority = random.choices(list(SENIORITY_POINTS), weights=[25, 25, 35, 15])[0]
    setup = random.choice(list(SETUP_POINTS))
    source = random.choices(list(SOURCE_POINTS), weights=[10, 35, 15, 25, 15])[0]
    demo = random.random() < 0.35
    trial = random.random() < (0.45 if demo else 0.12)
    pricing = random.choices(range(7), weights=[30, 20, 15, 12, 10, 8, 5])[0]
    visits = random.randint(0, 15)
    clicks = random.randint(0, 6)
    webinar = random.random() < 0.15
    days = random.choices([random.randint(0, 14), random.randint(15, 60), random.randint(61, 180)], weights=[50, 30, 20])[0]
    personal = random.random() < 0.18
    z = (-4.2 + 1.3 * demo + 1.1 * trial + 0.25 * pricing + 0.04 * visits + 0.6 * (source == "Referral")
         + 0.5 * (seniority == "Founder / CXO") + 0.4 * (orders >= 2000) + 0.3 * (industry == "D2C / E-commerce brand")
         - 0.02 * days - 0.7 * personal + random.gauss(0, 0.6))
    won = int(random.random() < 1 / (1 + math.exp(-z)))
    rows.append(dict(lead_id=f"H{i+1:03d}", company=f"Past Lead {i+1}", contact_name="Past Contact",
                     email=f"buyer{i+1}@{'gmail.com' if personal else f'pastlead{i+1}.in'}",
                     job_title="Buyer", industry=industry, monthly_orders=orders, employees=employees,
                     seniority=seniority, current_setup=setup, source=source,
                     demo_requested=int(demo), trial_started=int(trial), pricing_views=pricing,
                     site_visits=visits, email_clicks=clicks, webinar_attended=int(webinar),
                     days_since_last_activity=days, won=won))
pd.DataFrame(rows).to_csv("data/historical_leads.csv", index=False)
print("wrote", len(rows), "rows, win rate", sum(r["won"] for r in rows) / len(rows))
