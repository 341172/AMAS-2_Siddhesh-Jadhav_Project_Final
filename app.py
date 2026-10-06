"""LeadSense: sales lead-scoring assistant (use case #7, App format).

Run locally:  streamlit run app.py
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import ai
from scoring import (INDUSTRY_POINTS, SENIORITY_POINTS, SETUP_POINTS, SOURCE_POINTS, Settings,
                     apply_ai_adjustment, score_lead)
from validation import check_lead, clean_csv

BASE = Path(__file__).parent
SAMPLE = BASE / "data" / "sample_leads.csv"
HISTORY = BASE / "data" / "historical_leads.csv"
TIER_COLORS = {"Hot": "#B42318", "Warm": "#B54708", "Cold": "#175CD3", "Disqualified": "#475467"}
SAMPLE_LABELS = {
    "L01": "Strong inbound, clear notes",
    "L02": "Interested, but not the decision maker",
    "L03": "Low fit, low interest",
    "L04": "Competitor employee (edge case)",
    "L05": "Student doing a project (edge case)",
    "L06": "Great lead on a Gmail address",
    "L07": "Was hot, gone quiet for 95 days",
    "L08": "Hinglish call notes",
    "L09": "Tricky negation in notes",
    "L10": "Prompt injection hidden in notes",
    "L11": "Near-twin A (just under Hot)",
    "L12": "Near-twin B (just over Hot)",
    "L13": "Too small to serve (edge case)",
    "L14": "Great fit, zero engagement",
    "L15": "Very engaged, poor fit",
}

st.set_page_config(page_title="LeadSense · Lead scoring", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.ls-title {font-size: 2.1rem; font-weight: 750; letter-spacing: -0.02em; margin-bottom: 0;}
.ls-sub {color: #475467; font-size: 1.02rem; margin-top: .2rem; max-width: 760px;}
.ls-note {background:#F2F4F7; border-left: 3px solid #0E7490; padding:.6rem .9rem; font-size:.88rem; color:#344054; border-radius: 4px;}
.ls-tier {display:inline-block; padding:.35rem .9rem; border-radius: 999px; color:white; font-weight:700; font-size:1.15rem;}
.ls-score {font-size: 3.2rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em;}
.ls-muted {color:#667085; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def load_samples():
    rows, _, _ = clean_csv(pd.read_csv(SAMPLE))
    return {r["lead_id"]: r for r in rows}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyse(lead_json: str, settings_json: str) -> dict:
    """Same lead + same settings = same answer, with no second API call (protects the free quota)."""
    lead, s = json.loads(lead_json), Settings(**json.loads(settings_json))
    return ai.analyse(lead, score_lead(lead, s), s)


def tier_badge(tier: str) -> str:
    return f'<span class="ls-tier" style="background:{TIER_COLORS[tier]}">{tier}</span>'


def split_name(name: str):
    parts = str(name).split()
    return (parts[0] if parts else ""), (" ".join(parts[1:]) if len(parts) > 1 else "")


NEXT_STEPS = {"Hot": "Call within 24 hours and book a technical walkthrough.",
              "Warm": "Send a relevant case study and follow up in 3 days.",
              "Cold": "Add to the monthly nurture email list.",
              "Disqualified": "Do not pursue. Close the lead with the reason noted."}


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def crm_rows(leads, results):
    out = []
    for lead, r in zip(leads, results):
        first, last = split_name(lead["contact_name"])
        top = sorted(r.reasons, key=lambda x: -x[1])[0][0].split(" · ", 1)[-1]
        out.append({
            "Lead ID": lead.get("lead_id", ""), "Company Name": lead["company"], "First Name": first,
            "Last Name": last, "Email": lead["email"], "Job Title": lead["job_title"],
            "Industry": lead["industry"], "Lead Score": r.score, "Lead Status": r.tier,
            "Top Reason": r.disqualified_because or top, "Flags": " | ".join(r.flags),
            "Recommended Next Step": NEXT_STEPS[r.tier], "Scored On": date.today().isoformat(),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- state
ss = st.session_state
samples = load_samples()
ss.setdefault("form", dict(samples["L01"]))
ss.setdefault("lead", None)
ss.setdefault("ai_result", None)
ss.setdefault("ai_error", "")
ss.setdefault("answer", "")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Try a sample lead")
    pick = st.selectbox("Sample", list(samples), format_func=lambda k: f"{k} · {samples[k]['company']} — {SAMPLE_LABELS.get(k, '')}",
                        label_visibility="collapsed")
    if st.button("Load into form", type="primary"):
        ss.form = dict(samples[pick])
        ss.lead, ss.ai_result, ss.ai_error, ss.answer = None, None, "", ""
        st.rerun()

    st.divider()
    st.subheader("Scoring settings")
    fit_weight = st.slider("Weight on company fit (rest is engagement)", 0.2, 0.8, 0.5, 0.05,
                           help="0.5 means fit and engagement count equally.")
    hot = st.slider("Hot if score is at least", 50, 90, 70)
    warm = st.slider("Warm if score is at least", 20, 80, 45)
    if warm >= hot:
        st.error("The Warm cut-off must be lower than the Hot cut-off. Using 45 and 70 until you fix it.")
        hot, warm = 70, 45
    use_ai_adjust = st.toggle("Let verified call notes adjust the score (max ±10)", value=True)
    rep_name = st.text_input("Your name (for draft emails)", value="Sanjay")
    settings = Settings(fit_weight=fit_weight, hot_cutoff=hot, warm_cutoff=warm)

    st.divider()
    if ai.get_key():
        st.success("Gemini key found. AI features are on.")
    else:
        st.warning("No Gemini key. Rule-based scoring, CSV scoring and export still work; AI features are off.")
    st.caption("LeadSense is an AI-assisted tool. Scores and drafts are suggestions for a human rep, not decisions.")

# ---------------------------------------------------------------- header
st.markdown('<p class="ls-title">🎯 LeadSense</p>', unsafe_allow_html=True)
st.markdown('<p class="ls-sub">Lead scoring for a B2B sales team. A transparent scorecard sets the tier, '
            'Gemini reads the call notes and tells the rep what to do next.</p>', unsafe_allow_html=True)
st.markdown('<p class="ls-note">Demo set-up: <b>Checkoutly</b>, a fictional payments and checkout platform selling to Indian '
            'online businesses. All leads are sample data. When you use AI features, lead numbers and call notes are sent to '
            'Google Gemini with emails, phone numbers, PAN/GSTIN, company and contact names masked.</p>', unsafe_allow_html=True)
st.write("")

tab_one, tab_list, tab_check, tab_how = st.tabs(["Score a lead", "Score a list (CSV)", "Does the score work?", "How it works & limits"])

# ================================================================ TAB 1: one lead
with tab_one:
    f = ss.form
    with st.form("lead_form"):
        st.markdown("**Who is the lead?**")
        c1, c2, c3 = st.columns(3)
        company = c1.text_input("Company *", f["company"])
        contact = c2.text_input("Contact name *", f["contact_name"])
        email = c3.text_input("Work email *", f["email"])
        c1, c2, c3 = st.columns(3)
        title = c1.text_input("Job title *", f["job_title"])
        seniority = c2.selectbox("Seniority", list(SENIORITY_POINTS), index=list(SENIORITY_POINTS).index(f["seniority"]))
        industry = c3.selectbox("Industry", list(INDUSTRY_POINTS), index=list(INDUSTRY_POINTS).index(f["industry"]))
        c1, c2, c3 = st.columns(3)
        orders = c1.number_input("Online orders per month", 0, 10_000_000, int(f["monthly_orders"]), step=500)
        employees = c2.number_input("Employees", 1, 1_000_000, int(f["employees"]))
        setup = c3.selectbox("Current payment set-up", list(SETUP_POINTS), index=list(SETUP_POINTS).index(f["current_setup"]))

        st.markdown("**What have they done?**")
        c1, c2, c3, c4 = st.columns(4)
        source = c1.selectbox("Lead source", list(SOURCE_POINTS), index=list(SOURCE_POINTS).index(f["source"]))
        demo = c2.checkbox("Requested a demo", bool(f["demo_requested"]))
        trial = c3.checkbox("Started a trial", bool(f["trial_started"]))
        webinar = c4.checkbox("Attended a webinar", bool(f["webinar_attended"]))
        c1, c2, c3, c4 = st.columns(4)
        pricing = c1.number_input("Pricing page views", 0, 100, int(f["pricing_views"]))
        visits = c2.number_input("Site visits (30 days)", 0, 1000, int(f["site_visits"]))
        clicks = c3.number_input("Email clicks", 0, 200, int(f["email_clicks"]))
        days = c4.number_input("Days since last activity", 0, 365, int(f["days_since_last_activity"]))
        notes = st.text_area("Rep's call notes (English, Hindi or Hinglish)", f.get("notes", ""), height=100, max_chars=2000)
        submitted = st.form_submit_button("Score lead", type="primary")

    if submitted:
        lead = dict(lead_id=f.get("lead_id", ""), company=company.strip(), contact_name=contact.strip(),
                    email=email.strip(), job_title=title.strip(), industry=industry, monthly_orders=int(orders),
                    employees=int(employees), seniority=seniority, current_setup=setup, source=source,
                    demo_requested=demo, trial_started=trial, pricing_views=int(pricing), site_visits=int(visits),
                    email_clicks=int(clicks), webinar_attended=webinar, days_since_last_activity=int(days),
                    notes=notes.strip())
        errors = check_lead(lead)
        if errors:
            ss.lead = None
            st.error("Fix these before scoring:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            ss.form, ss.lead, ss.ai_result, ss.ai_error, ss.answer = lead, lead, None, "", ""
            result = score_lead(lead, settings)
            if result.tier != "Disqualified" and ai.get_key():
                with st.spinner("Gemini is reading the notes and drafting the brief…"):
                    try:
                        ss.ai_result = cached_analyse(json.dumps(lead, sort_keys=True),
                                                      json.dumps(settings.__dict__, sort_keys=True))
                    except ai.AIError as e:
                        ss.ai_error = str(e)
                    except Exception as e:  # never let an AI failure break the page
                        ss.ai_error = f"The AI step failed unexpectedly ({type(e).__name__}). Rule-based results are below."

    lead = ss.lead
    if lead:
        r = score_lead(lead, settings)
        a = ss.ai_result
        adj = a["adjustment"] if (a and use_ai_adjust) else 0
        final, final_tier = apply_ai_adjustment(r, adj, settings)

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"<div class='ls-muted'>{lead['company']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='ls-score'>{final}</div><div class='ls-muted'>out of 100</div>", unsafe_allow_html=True)
            st.markdown(tier_badge(final_tier), unsafe_allow_html=True)
            st.write("")
            if adj:
                st.caption(f"Scorecard {r.score} {'+' if adj > 0 else '−'} {abs(adj)} from verified call notes = {final}")
            else:
                st.caption(f"Scorecard score {r.score}. Call notes did not change it.")
            st.progress(r.fit / 50, text=f"Company fit: {r.fit:.0f} / 50")
            st.progress(min(r.engagement / 50, 1.0), text=f"Engagement: {r.engagement:.1f} / 50")
            if r.decay < 0.999:
                st.caption(f"Engagement counted at {r.decay:.0%}: last activity was {lead['days_since_last_activity']} days ago "
                           "(its value halves every 30 days).")
        with right:
            if r.disqualified_because:
                st.error(f"**Disqualified.** {r.disqualified_because} The AI step is skipped for disqualified leads.")
            for fl in r.flags:
                st.warning(fl)
            if a and a["injection"]:
                st.error("The call notes contain instructions aimed at the AI (possible prompt injection). "
                         "They were ignored, and the notes are not allowed to change the score for this lead.")
            st.markdown("**Why this score**")
            rs = pd.DataFrame([(lbl, pts) for lbl, pts in r.reasons], columns=["Signal", "Points"])
            st.dataframe(rs, hide_index=True, height=min(38 * (len(rs) + 1), 420))

        if r.tier != "Disqualified":
            st.divider()
            st.subheader("AI read-out")
            if ss.ai_error:
                st.info(ss.ai_error)
            if not a:
                tb = ai.template_brief(lead, r)
                st.markdown(f"**Summary (rule-based):** {tb['summary']}")
                st.markdown(f"**Next step:** {tb['next_action']}")
            else:
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**Summary.** {a['summary']}")
                    st.markdown(f"**Next best action.** {a['next_action']}")
                    if a["talking_points"]:
                        st.markdown("**Talking points**\n" + "\n".join(f"- {t}" for t in a["talking_points"]))
                with c2:
                    if a["ai_tier"]:
                        if a["ai_tier"] != final_tier:
                            st.warning(f"**Second opinion: the AI would say {a['ai_tier']}, the scorecard says {final_tier}.** "
                                       f"{a['ai_reason']} Check the reasons before acting.")
                        else:
                            st.success(f"**Second opinion agrees: {a['ai_tier']}.** {a['ai_reason']}")
                    st.caption(f"Model: {a.get('model') or 'Gemini'}")

                if a["bant"]:
                    st.markdown("**What the call notes say** (each claim must quote the notes word for word)")
                    icons = {"positive": "🟢", "negative": "🔴", "unclear": "⚪"}
                    bt = pd.DataFrame([{"Signal": k.capitalize(), "Reading": f"{icons[v['status']]} {v['status']}",
                                        "Quote from notes": v["quote"], "Check": v["verdict"],
                                        "Points": v["points"] if use_ai_adjust and not a["injection"] else 0}
                                       for k, v in a["bant"].items()])
                    st.dataframe(bt, hide_index=True)
                    if not use_ai_adjust:
                        st.caption("Notes adjustment is switched off in the sidebar, so these readings are shown but not scored.")

                if a["email_body"]:
                    with st.expander("Draft outreach email (review before sending; LeadSense never sends anything)"):
                        subj = ai.fill_email(a["email_subject"], lead, rep_name)
                        body = ai.fill_email(a["email_body"], lead, rep_name)
                        st.text_input("Subject", subj)
                        st.text_area("Body", body, height=220)

            st.markdown("**Ask about this lead**")
            q = st.text_input("Question", placeholder="e.g. What objection should I expect, and how do I handle it?",
                              label_visibility="collapsed")
            if st.button("Ask"):
                try:
                    with st.spinner("Thinking…"):
                        ss.answer = ai.ask(q, lead, r, settings)
                except ai.AIError as e:
                    ss.answer = f"⚠️ {e}"
            if ss.answer:
                st.info(ss.answer)

        st.download_button("Export this lead (CRM CSV)", crm_rows([lead], [r]).assign(**{"Lead Score": final, "Lead Status": final_tier})
                           .to_csv(index=False).encode(), file_name=f"lead_{lead['company'].replace(' ', '_')}.csv", mime="text/csv")
        st.caption("Results stay while this tab is open. A browser refresh starts a fresh session, so export anything you need.")

# ================================================================ TAB 2: a list
with tab_list:
    st.markdown("Upload a CSV of leads, or use the 15 sample leads. List scoring uses the rules only, "
                "so it makes **no** AI calls and works for any number of leads.")
    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("Leads CSV", type=["csv"], label_visibility="collapsed")
    c2.download_button("Download the CSV template", SAMPLE.read_bytes(), "leadsense_template.csv", "text/csv")
    use_sample = c2.button("Use the 15 sample leads")
    if use_sample:
        ss.list_source = "sample"
    if up is not None:
        ss.list_source = "upload"

    df_in, fatal = None, ""
    if ss.get("list_source") == "upload" and up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception:
            fatal = "This file could not be read as a CSV. Save it from Excel as 'CSV UTF-8' and try again."
    elif ss.get("list_source") == "sample":
        df_in = pd.read_csv(SAMPLE)

    if fatal:
        st.error(fatal)
    if df_in is not None:
        rows, problems, fatal = clean_csv(df_in)
        if fatal:
            st.error(fatal)
        else:
            if problems:
                with st.expander(f"⚠️ {len(problems)} row(s) skipped. See why."):
                    for p in problems:
                        st.write("- " + p)
            results = [score_lead(x, settings) for x in rows]
            crm = crm_rows(rows, results).sort_values("Lead Score", ascending=False)
            counts = crm["Lead Status"].value_counts()
            m = st.columns(5)
            m[0].metric("Leads scored", len(crm))
            for i, t in enumerate(["Hot", "Warm", "Cold", "Disqualified"]):
                m[i + 1].metric(t, int(counts.get(t, 0)))
            show = st.multiselect("Show tiers", ["Hot", "Warm", "Cold", "Disqualified"], default=["Hot", "Warm", "Cold", "Disqualified"])
            view = crm[crm["Lead Status"].isin(show)]
            st.dataframe(view[["Company Name", "Lead Score", "Lead Status", "Top Reason", "Flags", "Recommended Next Step"]],
                         hide_index=True)
            st.download_button("Export to CRM (HubSpot / Salesforce-style CSV)", view.to_csv(index=False).encode(),
                               "leadsense_crm_export.csv", "text/csv", type="primary")

            if st.button("Write a pipeline summary with AI (1 call)"):
                stats = {"leads": len(crm), "by_tier": {k: int(v) for k, v in counts.items()},
                         "avg_score_by_industry": crm.groupby("Industry")["Lead Score"].mean().round(1).to_dict(),
                         "leads_with_flags": int((crm["Flags"] != "").sum()),
                         "most_common_disqualification": crm.loc[crm["Lead Status"] == "Disqualified", "Top Reason"].value_counts().head(3).to_dict()}
                try:
                    with st.spinner("Summarising…"):
                        st.info(ai.pipeline_summary(stats))
                except ai.AIError as e:
                    st.info(str(e))
                st.caption("Only aggregate counts and averages were sent. No company or contact details.")

# ================================================================ TAB 3: validation
with tab_check:
    st.markdown("A score is only useful if Hot leads really close more often. This tab scores **past** leads whose outcome "
                "is known and checks that. The built-in file is 500 **synthetic** past leads; upload a CRM export with a "
                "`won` column (1/0) to test on real deals.")
    hist_up = st.file_uploader("Past leads with outcomes (optional)", type=["csv"], key="hist")
    try:
        hist = pd.read_csv(hist_up) if hist_up is not None else pd.read_csv(HISTORY)
    except Exception:
        hist = None
        st.error("This file could not be read as a CSV.")
    if hist is not None:
        hist.columns = [str(c).strip().lower() for c in hist.columns]
        if "won" not in hist.columns:
            st.error("The file needs a `won` column with 1 for closed-won and 0 otherwise.")
        else:
            rows, problems, fatal = clean_csv(hist)
            if fatal:
                st.error(fatal)
            else:
                won_by_email = dict(zip(hist["email"].astype(str).str.strip(), hist["won"].astype(int)))
                res = [score_lead(x, settings) for x in rows]
                d = pd.DataFrame({"tier": [x.tier for x in res], "score": [x.score for x in res],
                                  "won": [won_by_email.get(x["email"], 0) for x in rows],
                                  "demo": [bool(x["demo_requested"]) for x in rows]})
                total_wins = max(int(d["won"].sum()), 1)
                order = ["Hot", "Warm", "Cold", "Disqualified"]
                tbl = (d.groupby("tier").agg(Leads=("won", "size"), Won=("won", "sum"))
                       .reindex(order).fillna(0).astype(int))
                tbl["Win rate"] = (tbl["Won"] / tbl["Leads"].replace(0, pd.NA)).astype(float).fillna(0)
                tbl["Share of all wins"] = tbl["Won"] / total_wins
                hot_rate = tbl.loc["Hot", "Win rate"]
                cold_rate = tbl.loc["Cold", "Win rate"]
                c = st.columns(3)
                c[0].metric("Overall win rate", f"{d['won'].mean():.1%}")
                if tbl.loc["Hot", "Leads"]:
                    c[1].metric("Hot win rate", f"{hot_rate:.1%}", f"{(hot_rate / cold_rate if cold_rate else 0):.1f}x Cold")
                else:
                    c[1].metric("Hot win rate", "No Hot leads", "Lower the Hot cut-off", delta_color="off")
                c[2].metric("Wins caught by Hot + Warm", f"{(tbl.loc['Hot', 'Won'] + tbl.loc['Warm', 'Won']) / total_wins:.0%}")
                shown = tbl.copy()
                shown["Win rate"] = shown["Win rate"].map(pct)
                shown["Share of all wins"] = shown["Share of all wins"].map(lambda v: pct(v, 0))
                st.dataframe(shown)
                st.bar_chart(tbl["Win rate"], y_label="Win rate", x_label="Tier")

                st.markdown("**Compare with a simple rule of thumb: 'call everyone who asked for a demo'**")
                demo = d[d["demo"]]
                hotwarm = d[d["tier"].isin(["Hot", "Warm"])]
                cmp = pd.DataFrame({
                    "Leads to call": [len(demo), len(hotwarm)],
                    "Win rate": [demo["won"].mean(), hotwarm["won"].mean()],
                    "Share of all wins caught": [demo["won"].sum() / total_wins, hotwarm["won"].sum() / total_wins],
                }, index=["Rule of thumb: demo requested", "LeadSense: Hot + Warm"])
                cmp["Win rate"] = cmp["Win rate"].map(pct)
                cmp["Share of all wins caught"] = cmp["Share of all wins caught"].map(lambda v: pct(v, 0))
                st.dataframe(cmp)
                st.caption("If the rule of thumb catches more wins with fewer calls, the scorecard weights need fixing. "
                           "Move the sliders in the sidebar and watch these numbers change.")
                if problems:
                    st.caption(f"{len(problems)} row(s) skipped because they failed validation.")

# ================================================================ TAB 4: how it works
with tab_how:
    st.markdown("""
#### How a lead is scored
**Score = fit weight × fit% + (1 − fit weight) × engagement%**, out of 100. Default weight 0.5.

| Company fit (max 50) | Points | Engagement (max 50) | Points |
|---|---|---|---|
| Industry (D2C 15, EdTech/SaaS 12, Travel/Marketplace 10, Offline 5, Other 3) | 0–15 | Requested a demo | 12 |
| Online orders a month (50k+ 15, 10k+ 12, 2k+ 8, 500+ 4) | 0–15 | Started a trial | 10 |
| Employees (50–500 is the sweet spot) | 2–8 | Pricing page views (2 each, max 4) | 0–8 |
| Contact seniority (Founder/CXO 7 … IC 1) | 1–7 | Site visits (0.5 each, max 10) | 0–5 |
| Current set-up (COD-heavy 5, competitor 3, in-house 1) | 1–5 | Email clicks (1 each, max 5) | 0–5 |
| | | Webinar 3 · Source (Referral 7 … Cold outbound 1) | 1–10 |

Engagement fades with time: its value **halves every 30 days** without activity.
**Tiers:** Hot ≥ 70, Warm ≥ 45, else Cold (adjustable). Leads within 3 points of a cut-off are flagged as borderline.
**Disqualified:** competitor email domain, student/intern, or under 100 orders a month. A personal email is flagged, not penalised.

#### Where the AI comes in
1. **Reads call notes** for Budget, Authority, Need, Timeline (English, Hindi, Hinglish). Every claim must quote the notes
   exactly; if the quote is not found, the claim is dropped. Lines that can be read two ways go to a human.
   Verified signals move the score by **at most ±10** (2.5 per signal). Switch this off in the sidebar.
2. **Writes a brief**: summary, next best action, talking points, draft email. Nothing is ever sent.
3. **Gives a second opinion** on the tier and warns you when it disagrees with the scorecard.
4. **Answers questions** about the lead, within strict scope.

#### What LeadSense will not do
- It does not decide who gets called. A rep does.
- It does not judge people by personal traits, and refuses questions that ask it to.
- It does not follow instructions written inside call notes.

#### Privacy
Company name, contact name and email never go to Gemini. Notes are masked (emails, phone numbers, PAN, GSTIN, names) before
sending. On Google's free tier, prompts may be used to improve Google's models, so do not paste real customer data into this demo.

#### If Gemini is down
Scoring, list scoring, validation and export keep working. The AI panel shows a plain message and a rule-based brief instead.
""")
