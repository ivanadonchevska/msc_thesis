import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Bulgarian News Timeline")
st.markdown(
    """
    <style>
    div[data-testid="stPlotlyChart"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 2px solid #1a3460;
    }
    
    .stApp {
        background-color: #f1efeb;
    }
    .stButton > button {
        background-color: #1a3460;
        color: white !important;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #8b1f1f;
        color: white !important;
        border: none;
    }
    .stButton > button p {
        color: white !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #8b1f1f;
        font-weight: 600;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-color: #1a3460 !important;
        border-width: 1.5px !important;
    }
    div[data-testid="stAlert"] {
        background-color: #ede9e3  !important;
        border-left-color: #1a3460 !important;
        color: #1a3460 !important;
    }
    div[data-testid="stAlert"] p {
        color: #1a3460 !important;
    }
    a {
        color: #1a3460 !important;
    }
    a:hover {
        color: #8b1f1f !important;
    }
    hr {
        border-color: #1a3460;
        border-width: 1px;
    }
    p, li {
        color: #1a3460;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #e8e4de;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: #1a3460;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #1a3460;
        color: white;
    }
    div[data-baseweb="tooltip"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        display: none;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


REPO_ROOT = Path(__file__).parent.parent

DOT_SIZE = 14
MAX_PREV_DOTS = 5
MIN_ARTICLES = 1
STORIES_PER_PAGE = 6


@st.cache_data
def load_data():
    df = pd.read_parquet(REPO_ROOT / "data/processed/articles_clean.parquet")
    df["published_at_dt"] = pd.to_datetime(df["published_at_dt"], utc=True)
    with open(REPO_ROOT / "data/processed_system/stories.pkl", "rb") as f:
        stories = pickle.load(f)
    with open(REPO_ROOT / "data/processed_system/dots_louvain.pkl", "rb") as f:
        all_dots = pickle.load(f)
    embs_all = np.load(REPO_ROOT / "data/processed_system/embeddings.npy")

    names_path = REPO_ROOT / "data/processed_system/story_names.pkl"
    story_names = pickle.load(open(names_path, "rb")) if names_path.exists() else {}

    summaries_path = REPO_ROOT / "data/processed_system/dot_summaries.pkl"
    if summaries_path.exists():
        dot_summaries = pickle.load(open(summaries_path, "rb"))

    return df, stories, all_dots, embs_all, story_names, dot_summaries


def day_name(today_dots, embs_all, df):
    indices = [idx for dot in today_dots for idx in dot["indices"]]
    if not indices:
        return None
    embs = embs_all[indices]
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    return df.iloc[indices[sims.argmax()]]["title"]


@st.cache_data
def build_story_meta(_all_dots):
    meta = {}
    for sid, dots in _all_dots.items():
        dates = [d["effective_start"].date() for d in dots]
        meta[sid] = {"start": min(dates), "end": max(dates)}
    return meta


@st.cache_data
def build_date_index(_all_dots):
    index = {}
    for sid, dots in _all_dots.items():
        for dot in dots:
            d = dot["effective_start"].date()
            index.setdefault(d, set()).add(sid)
    return {d: list(sids) for d, sids in sorted(index.items())}


df, stories, all_dots, embs_all, story_names, dot_summaries = load_data()
story_meta = build_story_meta(all_dots)
date_index = build_date_index(all_dots)
sorted_dates = sorted(date_index.keys())
last_date = sorted_dates[-1]

if "selected_date" not in st.session_state:
    st.session_state.selected_date = last_date

# date navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    _, prev_btn = st.columns([1, 1])
    with prev_btn:
        if st.button("← Previous"):
            idx = sorted_dates.index(st.session_state.selected_date)
            if idx > 0:
                st.session_state.selected_date = sorted_dates[idx - 1]
with col2:
    st.markdown(
        f"<h2 style='text-align:center; color:#1a3460'>"
        f"{st.session_state.selected_date.strftime('%d %B %Y')}"
        f"</h2>",
        unsafe_allow_html=True,
    )

with col3:
    next_btn, _ = st.columns([1, 1])
    with next_btn:
        if st.button("Next →"):
            idx = sorted_dates.index(st.session_state.selected_date)
            if idx < len(sorted_dates) - 1:
                st.session_state.selected_date = sorted_dates[idx + 1]

selected_date = st.session_state.selected_date

# filter stories active on selected date
active_sids = [
    sid
    for sid in date_index.get(selected_date, [])
    if len(stories.get(sid, [])) >= MIN_ARTICLES
]


def latest_dot_time(sid):
    dots = all_dots.get(sid, [])
    today = [
        d["effective_start"]
        for d in dots
        if d["effective_start"].date() == selected_date
    ]
    return max(today) if today else pd.Timestamp.min.tz_localize("UTC")


def story_score(sid):
    return (len(all_dots.get(sid, [])), latest_dot_time(sid))


all_active_sids = sorted(active_sids, key=story_score, reverse=True)
total_active = len(all_active_sids)

if not all_active_sids:
    st.info("No stories found for this date.")
    st.stop()

if "story_page" not in st.session_state:
    st.session_state.story_page = 0

# reset pagination when date changes
if st.session_state.get("last_date") != selected_date:
    st.session_state.story_page = 0
    st.session_state.last_date = selected_date
    st.session_state.pop("story_select", None)
    st.session_state.pop("dot_select", None)
    st.session_state.pop("last_event_key", None)

total_pages = max(1, -(-total_active // STORIES_PER_PAGE))  # ceil division
st.session_state.story_page = min(st.session_state.story_page, total_pages - 1)

page_start = st.session_state.story_page * STORIES_PER_PAGE
active_sids = all_active_sids[page_start : page_start + STORIES_PER_PAGE]

# collect visible dots per story — prev: last MAX_PREV_DOTS before today, today: all on selected date
story_visible = {}
for sid in active_sids:
    dots = all_dots.get(sid, [])
    today_dots = [d for d in dots if d["effective_start"].date() == selected_date]
    prev_dots = sorted(
        [d for d in dots if d["effective_start"].date() < selected_date],
        key=lambda d: d["effective_start"],
    )[-MAX_PREV_DOTS:]
    story_visible[sid] = {"today": today_dots, "prev": prev_dots}

sid_labels = {
    sid: (
        story_names.get((sid, selected_date))
        or day_name(vis["today"], embs_all, df)
        or str(sid)
    )
    for sid, vis in story_visible.items()
}


# compute x range
max_today = max((len(v["today"]) for v in story_visible.values()), default=1)
x_left = -(MAX_PREV_DOTS + 0.5)
x_right = max_today + 0.5


def hover_text(d, sid, dot_idx_map):
    text = (
        f"<b>{d['effective_start'].strftime('%d %b %H:%M')}</b><br>"
        f"Articles: {d['size']}<br>"
        f"Sources: {', '.join(sorted(set(d['sources'])))}"
    )
    summary = dot_summaries.get((sid, dot_idx_map[id(d)]))
    if summary:
        wrapped = "<br>".join(textwrap.wrap(summary, width=45))
        text += f"<br><br>{wrapped}"
    return text


# story labels, shared between the chart and the selectboxes below
story_options = {
    f"{sid_labels[sid]}  |  {story_meta[sid]['start'].strftime('%d %b')} – {story_meta[sid]['end'].strftime('%d %b')}": sid
    for sid in active_sids
}

# resolve a click on the chart from the previous render (if any), before
# building the figure, so the clicked dot can be highlighted immediately
chart_event = st.session_state.get("timeline")
event_key = (
    str(chart_event["selection"]["points"][0])
    if chart_event and chart_event["selection"]["points"]
    else None
)
if event_key and event_key != st.session_state.get("last_event_key"):
    st.session_state["last_event_key"] = event_key
    point = chart_event["selection"]["points"][0]
    curve_idx = point["curve_number"]
    point_idx = point["point_number"]

    # trace map: per story → line, prev (optional), today (optional)
    trace_map = []
    for s in active_sids:
        vis = story_visible[s]
        trace_map.append(None)  # line trace
        if vis["prev"]:
            trace_map.append((s, vis["prev"]))
        if vis["today"]:
            trace_map.append((s, vis["today"]))

    if curve_idx < len(trace_map) and trace_map[curve_idx] is not None:
        clicked_sid, dot_group = trace_map[curve_idx]
        if point_idx < len(dot_group):
            clicked_dot = dot_group[point_idx]
            clicked_dots_all = sorted(
                all_dots.get(clicked_sid, []), key=lambda d: d["effective_start"]
            )
            clicked_dot_idx = clicked_dots_all.index(clicked_dot)
            st.session_state["story_select"] = (
                f"{sid_labels[clicked_sid]}  |  "
                f"{story_meta[clicked_sid]['start'].strftime('%d %b')} – "
                f"{story_meta[clicked_sid]['end'].strftime('%d %b')}"
            )
            st.session_state["dot_select"] = (
                f"{'★ ' if clicked_dot['effective_start'].date() == selected_date else ''}"
                f"{clicked_dot['effective_start'].strftime('%d %b %H:%M')}  "
                f"({clicked_dot['size']} articles)"
            )
            st.session_state["last_story_select"] = clicked_sid

# resolve selected story (fallback to the top story if unset or off-page)
if st.session_state.get("story_select") not in story_options:
    st.session_state["story_select"] = next(iter(story_options))
selected_sid = story_options[st.session_state["story_select"]]

# show ALL dots for the selected story (full history), not just the window
all_story_dots = sorted(
    all_dots.get(selected_sid, []), key=lambda d: d["effective_start"]
)
dot_options = {
    f"{'★ ' if d['effective_start'].date() == selected_date else ''}"
    f"{d['effective_start'].strftime('%d %b %H:%M')}  ({d['size']} articles)": i
    for i, d in enumerate(all_story_dots)
}
dot_labels = list(dot_options.keys())

# resolve selected dot (default to today's first dot when the story changed)
if (
    st.session_state.get("last_story_select") != selected_sid
    or st.session_state.get("dot_select") not in dot_options
):
    today_indices = [
        i
        for i, d in enumerate(all_story_dots)
        if d["effective_start"].date() == selected_date
    ]
    default_dot = today_indices[0] if today_indices else 0
    st.session_state["dot_select"] = dot_labels[default_dot]
st.session_state["last_story_select"] = selected_sid

selected_dot_idx = dot_options[st.session_state["dot_select"]]
selected_dot = all_story_dots[selected_dot_idx]


# build figure
fig = go.Figure()
highlight = None  # (x, y) of the selected dot, if visible on this page

for i, sid in enumerate(active_sids):
    meta = story_meta.get(sid, {})
    start_str = meta["start"].strftime("%d %b") if meta else ""
    end_str = meta["end"].strftime("%d %b") if meta else ""
    label = f"{sid_labels[sid]}  |  {start_str} – {end_str}"

    vis = story_visible[sid]
    today_dots = vis["today"]
    prev_dots = vis["prev"]

    dot_idx_map = {
        id(d): idx
        for idx, d in enumerate(
            sorted(all_dots.get(sid, []), key=lambda d: d["effective_start"])
        )
    }

    # positional x: prev dots at -len(prev)..-1, today dots at 1..N
    n_prev = len(prev_dots)
    prev_x = list(range(-n_prev, 0))  # e.g. [-3, -2, -1]
    today_x = list(range(1, len(today_dots) + 1))  # e.g. [1, 2, 3]

    line_x_start = prev_x[0] if prev_x else 0
    line_x_end = today_x[-1] if today_x else 0

    # connecting line
    fig.add_trace(
        go.Scatter(
            x=[line_x_start, line_x_end],
            y=[i, i],
            mode="lines",
            line=dict(color="#1a3460", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # story title — above the line
    fig.add_annotation(
        x=x_left,
        y=i,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="bottom",
        yshift=15,
        xshift=10,
        align="left",
        showarrow=False,
        font=dict(color="#111111", family="Arial, sans-serif", size=18),
        text=f"<b>{sid_labels[sid]}</b>",
    )

    # date range — below the line
    fig.add_annotation(
        x=x_left,
        y=i,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="top",
        yshift=-10,
        xshift=50,
        align="left",
        showarrow=False,
        font=dict(color="#666666", family="Arial, sans-serif", size=12),
        text=f"{start_str} – {end_str}",
    )

    # previous dots — grey (past)
    if prev_dots:
        fig.add_trace(
            go.Scatter(
                x=prev_x,
                y=[i] * n_prev,
                mode="markers",
                marker=dict(
                    size=DOT_SIZE,
                    color="#9e9e9e",
                    line=dict(width=1, color="white"),
                ),
                hovertext=[hover_text(d, sid, dot_idx_map) for d in prev_dots],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name=label,
            )
        )

    # today dots — navy
    if today_dots:
        fig.add_trace(
            go.Scatter(
                x=today_x,
                y=[i] * len(today_dots),
                mode="markers",
                marker=dict(
                    size=DOT_SIZE,
                    color="#1a3460",
                    line=dict(width=1, color="white"),
                ),
                hovertext=[hover_text(d, sid, dot_idx_map) for d in today_dots],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name=label,
            )
        )

    # track position of the selected dot, to draw a highlight ring after the loop
    if sid == selected_sid:
        for j, d in enumerate(prev_dots):
            if d is selected_dot:
                highlight = (prev_x[j], i)
        for j, d in enumerate(today_dots):
            if d is selected_dot:
                highlight = (today_x[j], i)

# ring around the currently selected dot
if highlight:
    fig.add_trace(
        go.Scatter(
            x=[highlight[0]],
            y=[highlight[1]],
            mode="markers",
            marker=dict(
                size=DOT_SIZE + 12,
                color="rgba(0,0,0,0)",
                line=dict(width=3, color="#8b1f1f"),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

for j in range(len(active_sids)):
    fig.add_hrect(
        y0=j - 0.5,
        y1=j + 0.5,
        fillcolor="rgba(255,255,255,0.20)" if j % 2 == 0 else "rgba(0,0,0,0.06)",
        layer="below",
        line_width=0,
    )


fig.update_layout(
    height=max(260, len(active_sids) * 90 + 50),
    margin=dict(l=10, r=10, t=30, b=20),
    yaxis=dict(autorange="reversed", showticklabels=False),
    plot_bgcolor="#e8e4de",
    paper_bgcolor="#ede9e3",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        range=[x_left, x_right],
        zeroline=False,
    ),
    clickmode="event+select",
    hoverlabel=dict(
        bgcolor="#eeedec",
        font_color="#1a3460",
        font_size=13,
        font_family="Arial, sans-serif",
        bordercolor="#8b1f1f",
    ),
)

st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="timeline")

_, prev_col, next_col, _ = st.columns([5, 1, 1, 5])
with prev_col:
    if st.button("‹", disabled=st.session_state.story_page == 0):
        st.session_state.story_page -= 1
        st.rerun()
with next_col:
    if st.button("›", disabled=st.session_state.story_page >= total_pages - 1):
        st.session_state.story_page += 1
        st.rerun()

# dot detail panel via selectboxes
st.markdown("---")
col_s, col_d = st.columns([2, 2])

with col_s:
    st.selectbox("Select story", list(story_options.keys()), key="story_select")

with col_d:
    st.selectbox("Select dot", dot_labels, key="dot_select")

summary = dot_summaries.get((selected_sid, selected_dot_idx))
if summary:
    st.info(summary)

st.markdown(f"**Sources:** {', '.join(sorted(set(selected_dot['sources'])))}")
st.markdown("**Articles:**")
for idx in sorted(selected_dot["indices"], key=lambda i: df.iloc[i]["published_at_dt"]):
    row = df.iloc[idx]
    pub = row["published_at_dt"].strftime("%d %b %H:%M")
    st.markdown(f"- **{pub}** [{row['source']}] — [{row['title']}]({row['url']})")
