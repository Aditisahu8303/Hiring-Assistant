"""
app.py - AI-Hiring Assistant
Run with: streamlit run app.py
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from utils.skill_extractor import extract_skills_from_jd

st.set_page_config(page_title="AI-Hiring Assistant", page_icon="🤖", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "resume_screening_dataset.csv")
SKILL_COLUMNS = ["python_skill","sql_skill","machine_learning_skill",
                 "deep_learning_skill","communication_skill","leadership_skill"]
SKILL_DISPLAY = {"python_skill":"Python","sql_skill":"SQL","machine_learning_skill":"ML",
                 "deep_learning_skill":"Deep Learning","communication_skill":"Communication",
                 "leadership_skill":"Leadership"}
JD_SKILL_MAP = {"python":"python_skill","sql":"sql_skill","machine learning":"machine_learning_skill",
                "deep learning":"deep_learning_skill","communication":"communication_skill",
                "leadership":"leadership_skill","nlp":"machine_learning_skill",
                "natural language processing":"machine_learning_skill","statistics":"machine_learning_skill",
                "ai":"machine_learning_skill","data visualization":"communication_skill"}

for k in ["logged_in","page"]:
    if k not in st.session_state:
        st.session_state[k] = False if k=="logged_in" else "Dashboard"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for c in ["cgpa","years_experience","internships_count","projects_count","certifications_count",
              "python_skill","sql_skill","machine_learning_skill","deep_learning_skill",
              "communication_skill","leadership_skill","github_projects","kaggle_activity_score",
              "hackathons_participated","expected_salary_lpa","employment_gap_months"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def compute_scores(df):
    d = df.copy()
    edu = (d["cgpa"]/10.0)*100
    exp = (d["years_experience"]/15.0).clip(upper=1.0)*100
    sk = d[SKILL_COLUMNS].mean(axis=1)/10.0*100
    pr = (d["projects_count"]/20.0).clip(upper=1.0)*100
    ce = (d["certifications_count"]/12.0).clip(upper=1.0)*100
    it = (d["internships_count"]/8.0).clip(upper=1.0)*100
    d["Composite_Score"] = (edu*0.20+exp*0.20+sk*0.25+pr*0.15+ce*0.10+it*0.10).round(2)
    d["Resume_Score"] = (edu*0.25+sk*0.30+exp*0.20+pr*0.15+ce*0.10).round(2)
    return d

def jd_match(df, jd_skills):
    d = df.copy()
    cols = set()
    for s in jd_skills:
        c = JD_SKILL_MAP.get(s.lower())
        if c: cols.add(c)
    if not cols:
        d["JD_Match_Score"] = 0.0
        return d
    cl = sorted(cols, key=lambda x: SKILL_COLUMNS.index(x) if x in SKILL_COLUMNS else 99)
    d["Avg_JD_Skill"] = d[cl].mean(axis=1)
    d["JD_Match_Score"] = ((d["Avg_JD_Skill"]/10.0)*100).round(2)
    return d

# ─── Login ────────────────────────────────────────────────────────────────

def show_login():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br><h1 style='text-align:center;'>🤖 AI-Hiring Assistant</h1>"
                    "<p style='text-align:center;color:gray;'>Sign in to continue</p><br>",
                    unsafe_allow_html=True)
        with st.form("login_form"):
            u = st.text_input("Username", placeholder="Enter username")
            p = st.text_input("Password", type="password", placeholder="Enter password")
            if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                if u=="admin" and p=="admin123":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown("<p style='text-align:center;color:gray;font-size:0.8em;'>Default: admin / admin123</p>",
                    unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────

def show_sidebar():
    labels = {"Dashboard":"📊 Dashboard","Top Candidates":"🏆 Top Candidates",
              "Hiring Funnel":"🔻 Hiring Funnel","Resume Score":"📄 Resume Score",
              "Resume Image Analysis":"🖼️ Resume Image Analysis",
              "Skills Gap":"📉 Skills Gap","Interview Analytics":"🎤 Interview Analytics",
              "Hiring Trends":"📈 Hiring Trends"}
    with st.sidebar:
        st.markdown("### 🤖 AI-Hiring Assistant")
        st.markdown("---")
        vals = list(labels.values())
        keys = list(labels.keys())
        sel = st.radio("Navigate", vals, index=keys.index(st.session_state.page))
        st.session_state.page = keys[vals.index(sel)]
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "Dashboard"

# ─── Page: Dashboard ──────────────────────────────────────────────────────

def show_dashboard(df):
    st.title("📊 Dashboard")
    st.caption("High-level overview of the candidate pool")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("👥 Total Candidates", f"{len(df):,}")
    m2.metric("✅ Selection Rate", f"{df['selected'].value_counts().get('Yes',0)/len(df)*100:.1f}%")
    m3.metric("📅 Avg Experience", f"{df['years_experience'].mean():.1f} yrs")
    m4.metric("🎓 Avg CGPA", f"{df['cgpa'].mean():.2f}")
    m5.metric("⚡ Avg Skill Score", f"{df[SKILL_COLUMNS].mean().mean():.1f}/10")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎓 Degree Distribution")
        dg = df["degree"].value_counts().reset_index()
        dg.columns=["degree","count"]
        fig = px.pie(dg, values="count", names="degree", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("⚡ Skills Radar (Avg)")
        av = [round(df[c].mean(),1) for c in SKILL_COLUMNS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=av+[av[0]],
            theta=[SKILL_DISPLAY[c] for c in SKILL_COLUMNS]+[SKILL_DISPLAY[SKILL_COLUMNS[0]]],
            fill="toself", line_color="#636EFA"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,10])),
                          margin=dict(l=40,r=40,t=10,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    c3,c4 = st.columns(2)
    with c3:
        st.subheader("📍 Location Distribution")
        lc = df["location"].value_counts().head(10).reset_index()
        lc.columns=["location","count"]
        fig = px.bar(lc, x="count", y="location", orientation="h", color="count",
                     color_continuous_scale="Viridis")
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=350,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("🏫 University Tier Distribution")
        tc = df["university_tier"].value_counts().reset_index()
        tc.columns=["tier","count"]
        fig = px.bar(tc, x="tier", y="count", color="tier",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ─── Page: Top Candidates ─────────────────────────────────────────────────

def show_top_candidates(df):
    st.title("🏆 Top Candidates")
    st.caption("Ranked candidates based on composite score across all parameters")
    with st.expander("🔍 Filters", expanded=True):
        fc1,fc2,fc3,fc4 = st.columns(4)
        with fc1: min_exp = st.slider("Min Experience (yrs)", 0, 15, 0)
        with fc2: min_cgpa = st.slider("Min CGPA", 0.0, 10.0, 0.0, 0.5)
        with fc3: deg_f = st.multiselect("Degree", list(df["degree"].unique()),
                                          default=list(df["degree"].unique()))
        with fc4: loc_f = st.multiselect("Location", list(df["location"].unique()), default=[])
    f = df.copy()
    f = f[f["years_experience"]>=min_exp]; f = f[f["cgpa"]>=min_cgpa]
    if deg_f: f = f[f["degree"].isin(deg_f)]
    if loc_f: f = f[f["location"].isin(loc_f)]
    if f.empty: st.warning("No candidates match the selected filters"); return
    s = compute_scores(f).sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    s["Rank"] = range(1, len(s)+1)
    top_n = st.slider("Show Top N candidates", 5, 100, 20, key="tc_top_n")
    d = s.head(top_n)
    sd = d[["Rank","candidate_id","degree","specialization","years_experience","cgpa","Composite_Score"]].copy()
    sd["Composite_Score"] = sd["Composite_Score"].apply(lambda x: f"{x:.1f}%")
    sd.columns = ["Rank","Candidate ID","Degree","Specialization","Experience (yrs)","CGPA","Score (%)"]
    st.subheader(f"Top {top_n} Candidates")
    st.dataframe(sd, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("📊 Score Distribution")
    fig = px.histogram(s, x="Composite_Score", nbins=30, color_discrete_sequence=["#636EFA"])
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("👤 Candidate Details")
    sel = st.selectbox("Select a candidate", s["candidate_id"].tolist())
    if sel:
        r = s[s["candidate_id"]==sel].iloc[0]
        dc1,dc2 = st.columns([1,1])
        with dc1:
            st.markdown("##### Profile")
            st.write(f"**ID:** {r['candidate_id']}")
            st.write(f"**Degree:** {r['degree']} ({r['specialization']})")
            st.write(f"**University:** {r['university_tier']}")
            st.write(f"**Experience:** {int(r['years_experience'])} yrs")
            st.write(f"**CGPA:** {r['cgpa']}")
            st.write(f"**Composite Score:** {r['Composite_Score']}%")
            st.write(f"**Expected Salary:** ₹{r['expected_salary_lpa']} LPA")
        with dc2:
            st.markdown("##### Skills")
            sd2 = {SKILL_DISPLAY[c]: int(r[c]) for c in SKILL_COLUMNS}
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(sd2.values()), y=list(sd2.keys()), orientation="h",
                                marker_color="#636EFA", text=list(sd2.values()), textposition="outside"))
            fig.update_layout(xaxis=dict(range=[0,11], title="Rating (1-10)"),
                              margin=dict(l=20,r=20,t=10,b=10), height=250)
            st.plotly_chart(fig, use_container_width=True)

# ─── Page: Hiring Funnel ──────────────────────────────────────────────────

def show_hiring_funnel(df):
    st.title("🔻 Hiring Funnel")
    st.caption("Candidate progression through hiring stages")
    total = len(df)
    scr = int((df["cgpa"]>=6.0).sum())
    shl = int((df["years_experience"]>=2).sum())
    intv = int(((df["cgpa"]>=6.0)&(df["years_experience"]>=2)).sum())
    sel = int((df["selected"]=="Yes").sum())
    stages = ["Total Applications","Pre-screened (CGPA ≥ 6)","Shortlisted (Exp ≥ 2yr)",
              "Interviewed","Selected"]
    vals = [total, scr, shl, intv, sel]
    safe = total or 1
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📋 Total", f"{vals[0]:,}")
    c2.metric("🔍 Screened", f"{vals[1]:,}", f"{vals[1]/safe*100:.0f}%")
    c3.metric("📝 Shortlisted", f"{vals[2]:,}", f"{vals[2]/safe*100:.0f}%")
    c4.metric("🎤 Interviewed", f"{vals[3]:,}", f"{vals[3]/safe*100:.0f}%")
    c5.metric("✅ Selected", f"{vals[4]:,}", f"{vals[4]/safe*100:.0f}%")
    st.markdown("---")
    fig = go.Figure(go.Funnel(y=stages, x=vals, textinfo="value+percent initial",
        marker=dict(color=["#1E90FF","#00CED1","#32CD32","#FFD700","#FF6347"])))
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    cr = [round(scr/safe*100,1), round(shl/(scr or 1)*100,1),
          round(intv/(shl or 1)*100,1), round(sel/(intv or 1)*100,1)]
    st.subheader("📊 Stage-wise Conversion")
    cd = pd.DataFrame({"Stage":["Applied→Screened","Screened→Shortlisted",
                                "Shortlisted→Interviewed","Interviewed→Selected"],
                       "Conversion Rate (%)": cr})
    st.dataframe(cd, use_container_width=True, hide_index=True)

# ─── Page: Resume Score ───────────────────────────────────────────────────

def show_resume_score(df):
    st.title("📄 Resume Score Analysis")
    st.caption("Detailed scoring breakdown for each candidate")
    s = compute_scores(df)
    c1,c2,c3 = st.columns(3)
    with c1: ms = st.slider("Min Resume Score", 0, 100, 0)
    with c2: df2 = st.multiselect("Degree", list(df["degree"].unique()),
                                    default=list(df["degree"].unique()))
    with c3: tn = st.slider("Top N", 10, 100, 30)
    f = s[(s["Resume_Score"]>=ms)&(s["degree"].isin(df2))]
    f = f.sort_values("Resume_Score", ascending=False).head(tn)
    st.subheader(f"Top {tn} Candidates by Resume Score")
    d = f[["candidate_id","degree","specialization","cgpa","years_experience","Resume_Score"]].copy()
    d["Resume_Score"] = d["Resume_Score"].apply(lambda x: f"{x:.1f}%")
    d.columns = ["Candidate ID","Degree","Specialization","CGPA","Experience (yrs)","Resume Score"]
    st.dataframe(d, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("📊 Score Breakdown by Category")
    sid = st.selectbox("Select a candidate", f["candidate_id"].tolist(), key="rs_select")
    if sid:
        r = s[s["candidate_id"]==sid].iloc[0]
        bd = pd.DataFrame({"Category":["Education (CGPA)","Skills (Avg Rating)","Experience",
                                       "Projects","Certifications"],
                           "Score (%)":[
                               round(r["cgpa"]/10.0*100,1),
                               round(r[SKILL_COLUMNS].mean()/10.0*100,1),
                               round(min(r["years_experience"]/15.0,1.0)*100,1),
                               round(min(r["projects_count"]/20.0,1.0)*100,1),
                               round(min(r["certifications_count"]/12.0,1.0)*100,1)]})
        cc1,cc2 = st.columns([1,1])
        with cc1:
            fig = px.bar(bd, x="Category", y="Score (%)", color="Category", text="Score (%)")
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(showlegend=False, height=350, margin=dict(l=20,r=20,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.markdown("##### Candidate Summary")
            st.write(f"**ID:** {r['candidate_id']}")
            st.write(f"**Degree:** {r['degree']} - {r['specialization']}")
            st.write(f"**CGPA:** {r['cgpa']}/10")
            st.write(f"**Experience:** {int(r['years_experience'])} yrs")
            st.write(f"**Projects:** {int(r['projects_count'])}")
            st.write(f"**Certifications:** {int(r['certifications_count'])}")
            st.write(f"**Overall Resume Score:** **{r['Resume_Score']}%**")

# ─── Page: Resume Image Analysis ──────────────────────────────────────────

def show_resume_image_analysis(df):
    st.title("🖼️ Resume Image Analysis")
    st.caption("Upload an image of your resume and enter the key details for scoring")
    uf = st.file_uploader("Upload your resume image (PNG/JPG)", type=["png","jpg","jpeg"])
    if uf is None:
        st.info("👆 Upload a resume image to get started")
        return
    img = Image.open(io.BytesIO(uf.read()))
    co, ci = st.columns([1,1])
    with co:
        st.image(img, caption="Uploaded Resume", use_container_width=True)
    with ci:
        st.markdown("##### Enter Resume Details")
        st.text_input("Candidate Name", placeholder="e.g. John Doe")
        deg = st.selectbox("Degree", options=list(df["degree"].unique()))
        spec = st.selectbox("Specialization", options=list(df["specialization"].unique()))
        exp = st.number_input("Years of Experience", 0, 50, 2)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0, 0.1)
        proj = st.number_input("Projects Count", 0, 50, 3)
        certs = st.number_input("Certifications Count", 0, 30, 1)
        intern = st.number_input("Internships Count", 0, 20, 1)
        st.markdown("##### Skills (1-10)")
        sk_in = {}
        cols = st.columns(3)
        for i, c in enumerate(SKILL_COLUMNS):
            with cols[i%3]:
                sk_in[c] = st.slider(SKILL_DISPLAY.get(c,c), 1, 10, 5, key=f"img_{c}")
        if not st.button("🔍 Analyze My Resume", type="primary", use_container_width=True):
            return
    edu_s = (cgpa/10.0)*100
    exp_s = min(exp/15.0,1.0)*100
    sk_av = sum(sk_in[c] for c in SKILL_COLUMNS)/len(SKILL_COLUMNS)
    sk_s = (sk_av/10.0)*100
    proj_s = min(proj/20.0,1.0)*100
    cert_s = min(certs/12.0,1.0)*100
    int_s = min(intern/8.0,1.0)*100
    us = round(edu_s*0.20+exp_s*0.20+sk_s*0.25+proj_s*0.15+cert_s*0.10+int_s*0.10,1)
    ur = round(edu_s*0.25+sk_s*0.30+exp_s*0.20+proj_s*0.15+cert_s*0.10,1)
    st.markdown("---")
    st.subheader("📊 Your Resume Score")
    ca,cb,cc,cd = st.columns(4)
    ca.metric("Composite Score", f"{us}%")
    cb.metric("Resume Score", f"{ur}%")
    cc.metric("Skill Avg", f"{sk_av:.1f}/10")
    cd.metric("Experience", f"{exp} yrs")
    st.markdown("---")
    st.subheader("🏆 Top 5 Candidates with Similar Profiles")
    sv = compute_scores(df)
    sv["diff"] = abs(sv["Composite_Score"]-us)
    sim = sv.sort_values("diff").head(5)
    sd = sim[["candidate_id","degree","specialization","years_experience","cgpa","Composite_Score"]].copy()
    sd["Composite_Score"] = sd["Composite_Score"].apply(lambda x: f"{x:.1f}%")
    sd.columns = ["Candidate ID","Degree","Specialization","Experience (yrs)","CGPA","Score (%)"]
    st.dataframe(sd, use_container_width=True, hide_index=True)
    st.subheader("📈 How You Compare vs Dataset Avg")
    av = [round(df[c].mean(),1) for c in SKILL_COLUMNS]
    cd2 = pd.DataFrame({"Skill":[SKILL_DISPLAY[c] for c in SKILL_COLUMNS],
                        "Your Score":[sk_in[c] for c in SKILL_COLUMNS],
                        "Dataset Avg":av})
    fig = go.Figure()
    fig.add_trace(go.Bar(name="You", x=cd2["Skill"], y=cd2["Your Score"], marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="Dataset Avg", x=cd2["Skill"], y=cd2["Dataset Avg"], marker_color="#EF553B"))
    fig.update_layout(barmode="group", height=350, margin=dict(l=20,r=20,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Skills Gap ─────────────────────────────────────────────────────

def show_skills_gap(df):
    st.title("📉 Skills Gap Analysis")
    st.caption("Compare candidate skills against job description requirements")
    jd = st.text_area(label="JD", label_visibility="collapsed",
                      value="Looking for a Data Scientist with Python, SQL, ML, Deep Learning, Communication.",
                      height=120, placeholder="Paste job description here...")
    if "sg_run" not in st.session_state: st.session_state.sg_run = False
    if st.button("🔍 Analyze", type="primary"): st.session_state.sg_run = True
    if not st.session_state.sg_run or not jd.strip():
        prev = extract_skills_from_jd(jd) if jd.strip() else []
        if prev:
            st.info(f"✅ Detected: {', '.join(s.title() for s in prev)}. Click **Analyze** for results.")
        else:
            st.info("Enter a job description and click Analyze.")
        return
    jds = extract_skills_from_jd(jd)
    if not jds:
        st.warning("No recognizable skills found."); st.session_state.sg_run = False; return
    mapped, unmapped = set(), []
    for s in jds:
        c = JD_SKILL_MAP.get(s.lower())
        if c: mapped.add(c)
        else: unmapped.append(s)
    sc1,sc2 = st.columns(2)
    with sc1:
        st.markdown("##### Found Skills")
        for s in jds: st.markdown(f"- {s.title()}")
    with sc2:
        st.markdown("##### Mapped Metrics")
        if mapped:
            for m in sorted(mapped, key=lambda x: SKILL_COLUMNS.index(x) if x in SKILL_COLUMNS else 99):
                st.markdown(f"- ✅ {SKILL_DISPLAY.get(m,m)}")
        if unmapped:
            st.markdown("##### Unavailable in Dataset")
            for u in unmapped: st.markdown(f"- ⚠️ {u.title()}")
    if not mapped:
        st.warning("None of the extracted skills could be mapped."); return
    md = jd_match(df, jds)
    st.markdown("---")
    st.subheader("🏆 Top Candidates by JD Match")
    tj = md.sort_values("JD_Match_Score", ascending=False).head(20)
    dc = ["candidate_id","degree","specialization","years_experience"]+sorted(mapped)+["JD_Match_Score"]
    dd = tj[dc].copy()
    dd["JD_Match_Score"] = dd["JD_Match_Score"].apply(lambda x: f"{x:.1f}%")
    dd.columns = ["Candidate ID","Degree","Specialization","Experience (yrs)"]+\
                 [SKILL_DISPLAY.get(c,c) for c in sorted(mapped)]+["JD Match Score"]
    st.dataframe(dd, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("📊 Skills Gap Heatmap (Top 15)")
    hv = [[r[c] for c in sorted(mapped)] for _,r in tj.head(15).iterrows()]
    if hv and hv[0]:
        fig = go.Figure(data=go.Heatmap(z=hv, y=tj.head(15)["candidate_id"].tolist(),
            x=[SKILL_DISPLAY.get(c,c) for c in sorted(mapped)],
            colorscale="RdYlGn", zmin=0, zmax=10, text=hv, texttemplate="%{text}", textfont={"size":10}))
        fig.update_layout(height=400, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("📈 Aggregate Skill Gap")
    ag = {SKILL_DISPLAY.get(c,c): round(df[c].mean(),1) for c in sorted(mapped)}
    gd = pd.DataFrame(list(ag.items()), columns=["Skill","Avg Rating (out of 10)"])
    fig = px.bar(gd, x="Skill", y="Avg Rating (out of 10)", color="Avg Rating (out of 10)",
                 color_continuous_scale="RdYlGn", range_color=[0,10], text="Avg Rating (out of 10)")
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Interview Analytics ────────────────────────────────────────────

def show_interview_analytics(df):
    st.title("🎤 Interview Analytics")
    st.caption("Patterns and insights from candidate selection")
    sel = df[df["selected"]=="Yes"]
    not_sel = df[df["selected"]=="No"]
    total = len(df)
    sr = len(sel)/total*100
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("✅ Selected", f"{len(sel):,}", f"{sr:.1f}%")
    m2.metric("❌ Not Selected", f"{len(not_sel):,}", f"{100-sr:.1f}%")
    m3.metric("🎓 Avg CGPA (Selected)", f"{sel['cgpa'].mean():.2f}")
    m4.metric("📅 Avg Exp (Selected)", f"{sel['years_experience'].mean():.1f} yrs")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Selection Rate by Degree")
        ds = df.groupby("degree")["selected"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        ds.columns=["degree","selection_rate"]
        ds = ds.sort_values("selection_rate", ascending=False)
        fig = px.bar(ds, x="selection_rate", y="degree", orientation="h",
                     color="selection_rate", color_continuous_scale="Viridis")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Selection Rate by University Tier")
        ts = df.groupby("university_tier")["selected"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        ts.columns=["tier","selection_rate"]
        fig = px.bar(ts, x="tier", y="selection_rate", color="tier",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=350, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    c3,c4 = st.columns(2)
    with c3:
        st.subheader("Selection Rate by Specialization")
        ss = df.groupby("specialization")["selected"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        ss.columns=["specialization","selection_rate"]
        ss = ss.sort_values("selection_rate", ascending=False)
        fig = px.bar(ss.head(10), x="selection_rate", y="specialization", orientation="h",
                     color="selection_rate", color_continuous_scale="Teal")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Key Insights")
        st.markdown(f"""
- **Highest selection rate:** {ds.iloc[0]['degree']} graduates
- **Top university tier:** {ts.iloc[0]['tier']}
- **Best specialization:** {ss.iloc[0]['specialization']}
- **Avg CGPA:** {sel['cgpa'].mean():.2f} (selected) vs {not_sel['cgpa'].mean():.2f} (not selected)
- **Avg Skill Rating:** {sel[SKILL_COLUMNS].mean().mean():.1f}/10 (selected) vs {not_sel[SKILL_COLUMNS].mean().mean():.1f}/10
""")
    st.markdown("---")
    st.subheader("📊 Feature Correlation with Selection")
    cc = SKILL_COLUMNS+["cgpa","years_experience","projects_count",
                        "certifications_count","internships_count","github_projects","kaggle_activity_score"]
    db = df[cc+["selected"]].copy()
    db["sel_bin"] = (db["selected"]=="Yes").astype(int)
    nc = [c for c in cc+["sel_bin"] if c in db.columns]
    corr = db[nc].corr()
    cws = corr[["sel_bin"]].drop("sel_bin").sort_values("sel_bin", ascending=False)
    cws.columns = ["Correlation with Selection"]
    cws.index = [SKILL_DISPLAY.get(c,c) for c in cws.index]
    fig = px.bar(cws, x=cws.index, y="Correlation with Selection", color="Correlation with Selection",
                 color_continuous_scale="RdYlGn", range_color=[-0.5,0.5], text_auto=".3f")
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=10,b=40), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Hiring Trends ──────────────────────────────────────────────────

def show_hiring_trends(df):
    st.title("📈 Hiring Trends")
    st.caption("Market trends and patterns in the candidate pool")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("📅 Experience Distribution")
        fig = px.histogram(df, x="years_experience", nbins=20, color_discrete_sequence=["#636EFA"])
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("💰 Expected Salary Distribution")
        fig = px.histogram(df, x="expected_salary_lpa", nbins=25, color_discrete_sequence=["#00CC96"])
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    c3,c4 = st.columns(2)
    with c3:
        st.subheader("👨‍🎓👩‍🎓 Gender Distribution")
        gc = df["gender"].value_counts().reset_index()
        gc.columns=["gender","count"]
        fig = px.pie(gc, values="count", names="gender", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("📚 Top Specializations")
        sc = df["specialization"].value_counts().head(10).reset_index()
        sc.columns=["specialization","count"]
        fig = px.bar(sc, x="count", y="specialization", orientation="h", color="count",
                     color_continuous_scale="Purples")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    c5,c6 = st.columns(2)
    with c5:
        st.subheader("📊 Experience vs Salary")
        sm = df.sample(min(2000,len(df)))
        fig = px.scatter(sm, x="years_experience", y="expected_salary_lpa", color="selected",
                         opacity=0.5, color_discrete_map={"Yes":"#00CC96","No":"#EF553B"})
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        st.subheader("📈 Skills Trend by Experience")
        bins, labs = [0,2,5,8,12,16], ["0-2","3-5","6-8","9-12","13-15"]
        dt = df.copy()
        dt["grp"] = pd.cut(dt["years_experience"], bins=bins, labels=labs, include_lowest=True)
        tr = dt.groupby("grp", observed=False)[SKILL_COLUMNS].mean().reset_index()
        tr = tr.dropna(subset=SKILL_COLUMNS).reset_index(drop=True)
        fig = go.Figure()
        for i,c in enumerate(SKILL_COLUMNS):
            fig.add_trace(go.Scatter(x=tr["grp"].astype(str), y=tr[c], mode="lines+markers",
                name=SKILL_DISPLAY[c], line=dict(color=px.colors.qualitative.Set2[i%len(px.colors.qualitative.Set2)])))
        fig.update_layout(height=350, margin=dict(l=20,r=20,t=10,b=20), yaxis_title="Avg Rating (1-10)")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("📍 Location-Wise Demand")
    lc = df["location"].value_counts().head(15).reset_index()
    lc.columns=["location","count"]
    fig = px.bar(lc, x="count", y="location", orientation="h", color="count",
                 color_continuous_scale="Teal")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=400, margin=dict(l=20,r=20,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ─── Router ───────────────────────────────────────────────────────────────

try:
    if st.session_state.logged_in:
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found at `{DATA_PATH}`")
            st.stop()
        df = load_data(DATA_PATH)
        st.sidebar.write(f"✅ Data loaded ({len(df):,} rows)")
        show_sidebar()
        pages = {"Dashboard": show_dashboard, "Top Candidates": show_top_candidates,
                 "Hiring Funnel": show_hiring_funnel, "Resume Score": show_resume_score,
                 "Resume Image Analysis": show_resume_image_analysis,
                 "Skills Gap": show_skills_gap, "Interview Analytics": show_interview_analytics,
                 "Hiring Trends": show_hiring_trends}
        fn = pages.get(st.session_state.page)
        if fn:
            fn(df)
        else:
            st.error(f"No function for page: {st.session_state.page}")
    else:
        show_login()
except Exception as e:
    import traceback
    st.error(f"Unexpected error: {e}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
