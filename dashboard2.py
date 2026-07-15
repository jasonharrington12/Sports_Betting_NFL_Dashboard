"""
dashboard2.py  —  NFL Prop Betting Dashboard (ESPN data edition)
================================================================
Run:  streamlit run dashboard2.py

Tabs
────
1. Prop Analyzer   – pick player / category / line → recommendation + bar chart
2. Player Profile  – full game-log table + rolling-average trend line
3. Team Overview   – fantasy-point bar chart per team + top players per team
4. League Leaders  – sortable per-stat leaderboard
5. Data Refresh    – scrape fresh data from the ESPN API without leaving the browser
"""

import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Dashboard",
    page_icon="🏈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CAT_MAP = {
    "pass yards": ("passing_yards",   "Passing Yards"),
    "rush yards": ("rush_yards",      "Rush Yards"),
    "rec yards":  ("receiving_yards", "Receiving Yards"),
    "receptions": ("receptions",      "Receptions"),
    "pass tds":   ("passing_tds",     "Passing TDs"),
    "fantasy":    ("fantasy_points",  "Fantasy Points"),
}

LEADER_COLS = {
    "Passing Yards":   "passing_yards",
    "Rush Yards":      "rush_yards",
    "Receiving Yards": "receiving_yards",
    "Receptions":      "receptions",
    "Passing TDs":     "passing_tds",
    "Fantasy Points":  "fantasy_points",
    "Completion %":    "completion_percentage",
    "Yards/Attempt":   "yards_per_attempt",
}

C_2024  = "#5B8DB8"
C_2025  = "#E07B54"
C_AVG   = "#2C2C2C"
C_LINE  = "#D62828"
C_OVER  = "#2DC653"
C_UNDER = "#E07B54"
C_TREND = "#7c5cd8"


# ──────────────────────────────────────────────────────────────────────────────
# ESPN API HELPERS
# ──────────────────────────────────────────────────────────────────────────────
import re as _re, time as _time, requests as _requests

_ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?seasontype=2&week={week}&dates={year}"
)
_ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    "?event={game_id}"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_FP = {"passing_yards": 0.04, "passing_tds": 4.0, "interceptions": -1.0,
       "rush_yards": 0.1, "rush_tds": 6.0, "receptions": 1.0,
       "receiving_yards": 0.1, "receiving_tds": 6.0}

def _safe_int(v):
    try: return int(v)
    except: return 0

def _parse_ca(s):
    m = _re.match(r"(\d+)/(\d+)", str(s))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def _calc_fp(r):
    return sum(r.get(k, 0) * v for k, v in _FP.items())

def _get_json(url):
    try:
        r = _requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Normalise ESPN schedule abbreviations → what's stored in the CSV game logs
# ESPN scoreboard uses LAR / WSH; our scraped data uses LA / WAS
_TEAM_NORM = {
    "LAR": "LA",
    "WSH": "WAS",
}

def _norm_team(abbr):
    """Map ESPN schedule abbreviations to our internal team codes."""
    return _TEAM_NORM.get(abbr, abbr)

# ESPN team abbreviation → numeric ID (all 32 teams)
_TEAM_IDS = {
    "ARI": 22, "ATL": 1,  "BAL": 33, "BUF": 2,  "CAR": 29, "CHI": 3,
    "CIN": 4,  "CLE": 5,  "DAL": 6,  "DEN": 7,  "DET": 8,  "GB":  9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC":  12, "LV":  13, "LAC": 24,
    "LAR": 14, "MIA": 15, "MIN": 16, "NE":  17, "NO":  18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SF":  25, "SEA": 26, "TB":  27,
    "TEN": 10, "WSH": 28,
}

# Positions relevant for prop betting
_PROP_POSITIONS = {"QB", "RB", "WR", "TE"}

@st.cache_data(ttl=21600, show_spinner=False)   # cache 6 hours
def fetch_all_depth_charts():
    """
    Pull the current depth chart for all 32 NFL teams from ESPN.
    Returns a dict:
        { "NE": { "QB": ["Drake Maye", "Tommy DeVito"],
                  "RB": ["Rhamondre Stevenson", ...],
                  "WR": ["Ja'Lynn Polk", ...],
                  "TE": ["Hunter Henry", ...] },
          "KC": { ... }, ... }
    Uses the 3WR 1TE offensive scheme (most common); falls back to any scheme.
    Players are ordered starter-first (rank 1, 2, 3…).
    """
    result = {}
    for abbr, team_id in _TEAM_IDS.items():
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl"
            f"/teams/{team_id}/depthcharts"
        )
        data = _get_json(url)
        if not data:
            result[abbr] = {}
            continue

        schemes = data.get("depthchart", [])
        # Prefer the 3WR 1TE pass-heavy scheme; fall back to first scheme
        scheme = next(
            (s for s in schemes if "WR" in s.get("name", "").upper()),
            schemes[0] if schemes else None,
        )
        if scheme is None:
            result[abbr] = {}
            continue

        positions = scheme.get("positions", {})
        team_chart = {}
        for pos_data in positions.values():
            pos_abbr = pos_data.get("position", {}).get("abbreviation", "")
            if pos_abbr not in _PROP_POSITIONS:
                continue
            # Athletes are already in depth order; sort by rank to be safe
            athletes = sorted(
                pos_data.get("athletes", []),
                key=lambda a: a.get("rank", 99),
            )
            names = [a.get("displayName", "") for a in athletes if a.get("displayName")]
            if names:
                # WR can appear multiple times (WR1/WR2/WR3 slots) — merge & dedupe
                existing = team_chart.get(pos_abbr, [])
                for n in names:
                    if n not in existing:
                        existing.append(n)
                team_chart[pos_abbr] = existing

        result[abbr] = team_chart
        _time.sleep(0.1)   # be polite to ESPN

    return result

def _scrape_game(game_id, season, week, home, away):
    data = _get_json(_ESPN_SUMMARY.format(game_id=game_id))
    if not data:
        return []
    rows = []
    for grp in data.get("boxscore", {}).get("players", []):
        team = grp.get("team", {}).get("abbreviation", "UNK")
        sbn  = {s["name"]: s for s in grp.get("statistics", [])}
        aids = {}
        for cat in sbn.values():
            for e in cat.get("athletes", []):
                a = e.get("athlete", {})
                if a.get("id") and a["id"] not in aids:
                    aids[a["id"]] = a.get("displayName", "Unknown")
        for aid, name in aids.items():
            row = dict(player_id=aid,
                       game_id=f"{season}_{week:02d}_{away}_{home}",
                       completions=0, attempts=0, passing_yards=0, passing_tds=0,
                       interceptions=0, rush_attempts=0, rush_yards=0, rush_tds=0,
                       receptions=0, targets=0, receiving_yards=0, receiving_tds=0,
                       season=season, player_name=name, team=team)
            for e in sbn.get("passing",   {}).get("athletes", []):
                if e["athlete"]["id"] == aid:
                    s = e.get("stats", [])
                    if len(s) >= 5:
                        c, a2 = _parse_ca(s[0])
                        row.update(completions=c, attempts=a2,
                                   passing_yards=_safe_int(s[1]),
                                   passing_tds=_safe_int(s[3]),
                                   interceptions=_safe_int(s[4]))
                    break
            for e in sbn.get("rushing",   {}).get("athletes", []):
                if e["athlete"]["id"] == aid:
                    s = e.get("stats", [])
                    if len(s) >= 4:
                        row.update(rush_attempts=_safe_int(s[0]),
                                   rush_yards=_safe_int(s[1]),
                                   rush_tds=_safe_int(s[3]))
                    break
            for e in sbn.get("receiving", {}).get("athletes", []):
                if e["athlete"]["id"] == aid:
                    s = e.get("stats", [])
                    if len(s) >= 4:
                        row.update(receptions=_safe_int(s[0]),
                                   receiving_yards=_safe_int(s[1]),
                                   receiving_tds=_safe_int(s[3]),
                                   targets=_safe_int(s[5]) if len(s) >= 6 else 0)
                    break
            if row["attempts"]+row["rush_attempts"]+row["receptions"]+row["targets"] == 0:
                continue
            row["fantasy_points"] = round(_calc_fp(row), 4)
            rows.append(row)
    return rows

def _scrape_season_live(year, progress_text=None):
    """Scrape a full season into a DataFrame with no disk I/O."""
    all_rows = []
    for week in range(1, 19):
        if progress_text:
            progress_text.text(f"Scraping {year} — week {week}/18…")
        data = _get_json(_ESPN_SCOREBOARD.format(week=week, year=year))
        if not data:
            continue
        events = data.get("events", [])
        if not events:
            break
        for event in events:
            completed = (event.get("competitions", [{}])[0]
                         .get("status", {}).get("type", {})
                         .get("completed", False))
            if not completed:
                continue
            gid   = event["id"]
            comps = event.get("competitions", [{}])[0].get("competitors", [])
            home = away = "UNK"
            for c in comps:
                ab = c.get("team", {}).get("abbreviation", "UNK")
                if c.get("homeAway") == "home": home = ab
                else: away = ab
            all_rows.extend(_scrape_game(gid, year, week, home, away))
            _time.sleep(0.35)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).drop_duplicates()
    return df.sort_values(["player_name", "game_id"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (ESPN API only — cached for 1 hour so data stays fresh)
# ttl=3600 means: first visitor triggers a scrape, everyone else for the next
# hour gets instant loads. After 1 hour it automatically re-scrapes, picking up
# any new games that were played.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    import datetime as _dt
    # Auto-detect current and previous NFL season years.
    # NFL seasons straddle two calendar years: the 2025 season runs Sep 2025–Jan 2026.
    # Jan–Aug = still the previous season year; Sep–Dec = new season year.
    _today = _dt.date.today()
    _cur_year  = _today.year if _today.month >= 9 else _today.year - 1
    _prev_year = _cur_year - 1

    msg  = st.empty()
    prog = st.empty()
    msg.info(f"Loading {_prev_year} + {_cur_year} data from ESPN API... "
             "this takes ~5 min on first load, then caches for 1 hour.")
    df_prev = _scrape_season_live(_prev_year, prog)
    df_cur  = _scrape_season_live(_cur_year,  prog)
    prog.empty()
    msg.empty()

    # Current season may be empty before it starts (offseason) — that's fine
    if df_prev.empty and df_cur.empty:
        raise RuntimeError(
            "ESPN API returned no data. It may be temporarily unavailable — "
            "try refreshing the page in a minute."
        )

    # Use whichever frames have data
    df_2024 = df_prev if not df_prev.empty else pd.DataFrame(columns=df_cur.columns)
    df_2025 = df_cur  if not df_cur.empty  else pd.DataFrame(columns=df_prev.columns)

    for df in [df_2024, df_2025]:
        df.columns = df.columns.str.lower().str.strip()

    df_2024 = df_2024.drop_duplicates()
    df_2025 = df_2025.drop_duplicates()

    # ── team-change detection ──────────────────────────────────────────────
    teams_2024 = (
        df_2024.sort_values("game_id").groupby("player_name")["team"]
        .last().reset_index().rename(columns={"team": "team_2024"})
    )
    teams_2025 = (
        df_2025.sort_values("game_id").groupby("player_name")["team"]
        .last().reset_index().rename(columns={"team": "team_2025"})
    )
    team_changes = teams_2024.merge(teams_2025, on="player_name", how="inner")
    team_changes["changed_team"] = team_changes["team_2024"] != team_changes["team_2025"]

    # Only flag changed_team=True for players who actually changed.
    # Use outer merge so players only in 2025 (rookies) also get the column.
    df_2024 = df_2024.merge(
        team_changes[["player_name", "changed_team"]], on="player_name", how="left"
    )
    df_2024["changed_team"] = df_2024["changed_team"].fillna(False)

    # Give 2025 rows the column too (always False — they're playing for their current team)
    df_2025["changed_team"] = False

    nfl = pd.concat([df_2024, df_2025], ignore_index=True)
    nfl["changed_team"] = nfl["changed_team"].fillna(False)
    nfl = nfl.sort_values(["player_name", "season", "game_id"]).reset_index(drop=True)

    # ── season weights ─────────────────────────────────────────────────────
    def _weight(row):
        if row["season"] == 2025:
            return 1.0
        if row["season"] == 2024 and row.get("changed_team", False):
            return 0.3
        return 0.6

    nfl["weight"] = nfl.apply(_weight, axis=1)

    # ── efficiency metrics ─────────────────────────────────────────────────
    nfl["completion_percentage"] = np.where(
        nfl["attempts"] > 0, nfl["completions"] / nfl["attempts"], 0
    )
    nfl["yards_per_attempt"] = np.where(
        nfl["attempts"] > 0, nfl["passing_yards"] / nfl["attempts"], 0
    )
    nfl["yards_per_reception"] = np.where(
        nfl["receptions"] > 0, nfl["receiving_yards"] / nfl["receptions"], 0
    )

    # ── rolling averages ───────────────────────────────────────────────────
    nfl = nfl.sort_values(["player_name", "season", "game_id"])
    for col, new_col in [
        ("passing_yards",   "last_3_pass_avg"),
        ("rush_yards",      "last_3_rush_avg"),
        ("receiving_yards", "last_3_rec_avg"),
        ("fantasy_points",  "last_3_fp_avg"),
    ]:
        nfl[new_col] = (
            nfl.groupby("player_name")[col]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

    # ── fill missing ───────────────────────────────────────────────────────
    num_cols = nfl.select_dtypes(include="number").columns
    cat_cols = nfl.select_dtypes(include="object").columns
    nfl[num_cols] = nfl[num_cols].fillna(0)
    nfl[cat_cols] = nfl[cat_cols].fillna("Unknown")
    nfl["player_name"] = nfl["player_name"].str.strip()

    return nfl, team_changes


# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_player(nfl, name):
    return nfl[nfl["player_name"].str.contains(name, case=False, na=False)]


def hit_rate(df, col, line):
    if df.empty:
        return None, None, None
    try:
        over = (df[col] > float(line)).sum()
        return (over / len(df)) * 100, over, len(df)
    except Exception:
        return None, None, None


def prop_analysis(nfl, player_name, category, line, use_weighted=True, game_window="Season"):
    cat_key = category.lower().strip()
    if cat_key not in CAT_MAP:
        return None
    col = CAT_MAP[cat_key][0]
    try:
        line = float(line)
    except (TypeError, ValueError):
        return None

    pdf = find_player(nfl, player_name)
    if pdf.empty:
        return None

    full    = pdf["player_name"].iloc[0]
    p25     = pdf[pdf["season"] == 2025]
    p24     = pdf[pdf["season"] == 2024]
    changed = pdf["changed_team"].any() if "changed_team" in pdf.columns else False

    try:
        hr25, ov25, tot25 = hit_rate(p25, col, line)
    except Exception:
        hr25, ov25, tot25 = None, None, None
    try:
        hr24, ov24, tot24 = hit_rate(p24, col, line)
    except Exception:
        hr24, ov24, tot24 = None, None, None

    # Window slice — tail of the combined sorted dataframe
    n_games = {"Last 3": 3, "Last 5": 5, "Season": None}.get(game_window, None)
    window_df = pdf.tail(n_games) if n_games is not None else pdf

    vals = window_df[col].values
    wts  = window_df["weight"].values
    if use_weighted:
        w_avg = np.average(vals, weights=wts) if len(vals) else 0.0
        w_hit = np.average((vals > line).astype(float), weights=wts) * 100 if len(vals) else 0.0
    else:
        w_avg = vals.mean() if len(vals) else 0.0
        w_hit = (vals > line).mean() * 100 if len(vals) else 0.0

    window_avg = window_df[col].mean() if not window_df.empty else 0.0
    window_hit = (window_df[col] > line).mean() * 100 if not window_df.empty else 0.0

    return {
        "full_name":     full,
        "changed":       changed,
        "team_24":       p24["team"].iloc[-1] if not p24.empty else "N/A",
        "team_25":       p25["team"].iloc[-1] if not p25.empty else "N/A",
        "hr_2025":       hr25, "over_2025": ov25, "total_2025": tot25,
        "avg_2025":      p25[col].mean() if not p25.empty else None,
        "hr_2024":       hr24, "over_2024": ov24, "total_2024": tot24,
        "avg_2024":      p24[col].mean() if not p24.empty else None,
        "w_avg":         w_avg,
        "w_hit":         w_hit,
        "weight_label":  "0.3" if changed else "0.6",
        "window_avg":    window_avg,
        "window_hit":    window_hit,
        "window_label":  game_window,
        "window_games":  len(window_df),
        "std_dev":       window_df[col].std() if not window_df.empty else 0.0,
        "recommendation": "OVER" if w_avg > line else "UNDER",
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def bar_chart(nfl, player_name, category, line=None, game_window="Season"):
    col, col_label = CAT_MAP[category.lower()]
    pdf = find_player(nfl, player_name)
    if pdf.empty:
        return None

    full    = pdf["player_name"].iloc[0]
    n_games = {"Last 3": 3, "Last 5": 5, "Season": None}[game_window]

    if n_games is not None:
        combined = pdf.tail(n_games).copy()
        p24 = combined[combined["season"] == 2024].reset_index(drop=True)
        p25 = combined[combined["season"] == 2025].reset_index(drop=True)
    else:
        p24 = pdf[pdf["season"] == 2024].reset_index(drop=True)
        p25 = pdf[pdf["season"] == 2025].reset_index(drop=True)

    p24["week"] = range(1, len(p24) + 1)
    p25["week"] = range(1, len(p25) + 1)

    has24, has25 = not p24.empty, not p25.empty
    ncols = 2 if (has24 and has25) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    window_str = f"Last {n_games} Games" if n_games else "Full Season"
    fig.suptitle(f"{full}  —  {col_label}  |  {window_str}",
                 fontsize=13, fontweight="bold", y=1.02)

    def _draw(ax, sdf, label, bar_color):
        weeks  = sdf["week"].values
        values = sdf[col].values
        avg    = values.mean()
        colors = [
            C_OVER if (line is not None and v > line) else
            (C_UNDER if line is not None else bar_color)
            for v in values
        ]
        bars = ax.bar(weeks, values, color=colors, edgecolor="white",
                      linewidth=0.6, alpha=0.87, zorder=2)
        ax.axhline(avg, color=C_AVG, linewidth=1.8, linestyle="--",
                   label=f"Season Avg: {avg:.2f}", zorder=3)
        if line is not None:
            ax.axhline(line, color=C_LINE, linewidth=1.8, linestyle="-",
                       label=f"Prop Line: {line}", zorder=3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    str(int(val)) if val == int(val) else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Week", fontsize=9)
        ax.set_ylabel(col_label, fontsize=9)
        ax.set_xticks(weeks)
        ax.set_xticklabels([str(w) for w in weeks], fontsize=7)
        ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 10)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    idx = 0
    if has24:
        _draw(axes[idx], p24, "2024 Season", C_2024); idx += 1
    if has25:
        _draw(axes[idx], p25, "2025 Season", C_2025)

    if line is not None:
        fig.legend(
            handles=[
                mpatches.Patch(color=C_OVER,  label=f"Over {line}"),
                mpatches.Patch(color=C_UNDER, label=f"Under {line}"),
            ],
            loc="lower center", ncol=2, fontsize=8,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.07),
        )
    plt.tight_layout()
    return fig


def trend_chart(nfl, player_name, category):
    col, col_label = CAT_MAP[category.lower()]
    roll_map = {
        "passing_yards":   "last_3_pass_avg",
        "rush_yards":      "last_3_rush_avg",
        "receiving_yards": "last_3_rec_avg",
        "fantasy_points":  "last_3_fp_avg",
    }
    roll_col = roll_map.get(col)

    pdf = find_player(nfl, player_name).copy()
    if pdf.empty:
        return None

    pdf = pdf.reset_index(drop=True)
    pdf["game_num"] = range(1, len(pdf) + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [C_2025 if r == 2025 else C_2024 for r in pdf["season"]]
    ax.bar(pdf["game_num"], pdf[col], color=colors, alpha=0.6,
           edgecolor="white", linewidth=0.5, zorder=2, label="_nolegend_")

    if roll_col and roll_col in pdf.columns:
        ax.plot(pdf["game_num"], pdf[roll_col], color=C_TREND,
                linewidth=2, label="3-game rolling avg", zorder=4)

    # Season boundary line
    boundary = pdf[pdf["season"] == 2025]["game_num"].min()
    if pd.notna(boundary) and boundary > 1:
        ax.axvline(boundary - 0.5, color="#888", linewidth=1.2,
                   linestyle=":", label="2024 → 2025")

    ax.set_title(f"{pdf['player_name'].iloc[0]}  —  {col_label}  |  All Games",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Game #", fontsize=9)
    ax.set_ylabel(col_label, fontsize=9)
    ax.legend(
        handles=[
            mpatches.Patch(color=C_2024, label="2024"),
            mpatches.Patch(color=C_2025, label="2025"),
            plt.Line2D([0], [0], color=C_TREND, linewidth=2, label="3-game rolling avg"),
        ],
        fontsize=8, framealpha=0.85,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    plt.tight_layout()
    return fig


def team_bar_chart(nfl, season, stat_col, stat_label):
    tdf = (
        nfl[nfl["season"] == season]
        .groupby("team")[stat_col].mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(tdf) * 0.32)))
    colors = [C_2025 if season == 2025 else C_2024] * len(tdf)
    bars = ax.barh(tdf["team"], tdf[stat_col], color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, tdf[stat_col]):
        ax.text(bar.get_width() + tdf[stat_col].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7.5)
    ax.set_title(f"{season}  —  Avg {stat_label} per Game by Team",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Avg {stat_label}", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏈 NFL Prop Betting Dashboard")
st.caption("Live data via ESPN API  ·  2024 + 2025 regular seasons  ·  PPR scoring")

with st.spinner("Loading data…"):
    try:
        nfl_df, team_changes = load_data()
        data_ok = True
    except Exception as e:
        st.error(
            f"**Data load failed:** {e}\n\n"
            "If running locally, use the **Data Refresh** tab to scrape data first. "
            "If this is a fresh cloud deploy, the ESPN API may be temporarily unavailable — "
            "try refreshing the page in a minute."
        )
        data_ok = False

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
main_bet, main_players, main_teams, main_tracker, main_settings = st.tabs([
    "🎯 Betting Tools",
    "👤 Players",
    "🏟️ Teams & League",
    "📒 Bet Tracker",
    "⚙️ Settings & Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TAB 1 — BETTING TOOLS
# Sub-tabs: Prop Analyzer · Matchup Edge · Matchup Finder · Parlay Builder · Streak Finder
# ══════════════════════════════════════════════════════════════════════════════
with main_bet:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        tab1, tab6, tab8, tab7, tab_streak, tab_vegas = st.tabs([
            "📊 Prop Analyzer",
            "🆚 Matchup Edge",
            "🎯 Matchup Finder",
            "🎰 Parlay Builder",
            "🔥 Streak Finder",
            "📈 Vegas Lines",
        ])

        # ── PROP ANALYZER ────────────────────────────────────────────────────
        with tab1:
            all_players  = sorted(nfl_df["player_name"].unique())
            all_teams_pa = ["None (skip matchup)"] + sorted(nfl_df["team"].dropna().unique().tolist())

            # Build opponent defense lookup once (reuse Matchup Edge logic)
            @st.cache_data(show_spinner=False)
            def build_opp_pa(nfl):
                df = nfl.copy()
                def _opp(row):
                    parts = str(row["game_id"]).split("_")
                    if len(parts) < 4: return "UNK"
                    away, home = parts[2], parts[3]
                    return home if row["team"] == away else away
                df["opponent"] = df.apply(_opp, axis=1)
                return df
            nfl_opp_pa = build_opp_pa(nfl_df)

            c_left, c_right = st.columns([1, 3])
            with c_left:
                st.subheader("Controls")
                player_sel = st.selectbox(
                    "Player", all_players,
                    index=all_players.index("Drake Maye") if "Drake Maye" in all_players else 0,
                    key="pa_player",
                )
                cat_sel = st.selectbox(
                    "Stat Category", list(CAT_MAP.keys()),
                    format_func=str.title, key="pa_cat",
                )
                line_val = st.number_input(
                    "Prop Line", min_value=0.0, value=200.5, step=0.5,
                    format="%.1f", key="pa_line",
                )
                opp_sel = st.selectbox(
                    "Opponent this week",
                    all_teams_pa,
                    index=0,
                    key="pa_opp",
                    help="Select the opposing defense to factor matchup difficulty into the recommendation.",
                )
                weighted = st.toggle("Season Weighting", value=True,
                                     help="2025 × 1.0 · 2024 × 0.6 · team-changer × 0.3")
                game_window = st.radio(
                    "Game Window",
                    options=["Last 3", "Last 5", "Season"],
                    index=2,
                    horizontal=True,
                    help="Limit analysis and chart to the most recent N games.",
                    key="pa_window",
                )
                go = st.button("Analyze", type="primary", use_container_width=True, key="pa_go")

                st.divider()
                st.caption("**Dataset**")
                st.metric("Rows",    f"{len(nfl_df):,}")
                st.metric("Players", f"{nfl_df['player_name'].nunique():,}")
                st.metric("Team Changes", int(team_changes["changed_team"].sum()))

            with c_right:
                if go:
                    res = prop_analysis(nfl_df, player_sel, cat_sel, line_val, weighted, game_window)
                    if res is None:
                        st.error("Player not found.")
                    else:
                        pa_col = CAT_MAP[cat_sel.lower()][0]

                        # ── Matchup analysis ──────────────────────────────────
                        matchup_info = None
                        if opp_sel != "None (skip matchup)":
                            vs_opp   = nfl_opp_pa[nfl_opp_pa["opponent"] == opp_sel]
                            lg_avg   = nfl_opp_pa.groupby("opponent")[pa_col].mean().mean()
                            opp_avg  = vs_opp[pa_col].mean() if not vs_opp.empty else lg_avg
                            def_rank_series = (
                                nfl_opp_pa.groupby("opponent")[pa_col]
                                .mean().sort_values(ascending=False)
                            )
                            rank = list(def_rank_series.index).index(opp_sel) + 1 if opp_sel in def_rank_series.index else None
                            n_teams = len(def_rank_series)
                            factor  = opp_avg / lg_avg if lg_avg > 0 else 1.0
                            if factor > 1.10:
                                grade = "🟢 Soft"
                                grade_color = "#2DC653"
                            elif factor < 0.90:
                                grade = "🔴 Tough"
                                grade_color = "#D62828"
                            else:
                                grade = "🟡 Average"
                                grade_color = "#f59e0b"
                            matchup_info = {
                                "opp": opp_sel, "opp_avg": opp_avg, "lg_avg": lg_avg,
                                "factor": factor, "grade": grade,
                                "grade_color": grade_color,
                                "rank": rank, "n_teams": n_teams,
                            }

                        # ── Team change warning ───────────────────────────────
                        if res["changed"]:
                            st.warning(
                                f"⚠️ Team change: **{res['team_24']}** (2024) → "
                                f"**{res['team_25']}** (2025) — 2024 weight = {res['weight_label']}"
                            )

                        # ── Matchup banner ────────────────────────────────────
                        if matchup_info:
                            st.markdown(
                                f'<div style="background:{matchup_info["grade_color"]}22;'
                                f'border-left:5px solid {matchup_info["grade_color"]};'
                                f'padding:10px 14px;border-radius:6px;margin-bottom:12px;font-size:14px;">'
                                f'<b>Matchup vs {matchup_info["opp"]}:</b> {matchup_info["grade"]} &nbsp;·&nbsp; '
                                f'Allows <b>{matchup_info["opp_avg"]:.1f}</b> {CAT_MAP[cat_sel.lower()][1]}/gm '
                                f'(league avg {matchup_info["lg_avg"]:.1f}) &nbsp;·&nbsp; '
                                f'Def rank <b>#{matchup_info["rank"]}</b> of {matchup_info["n_teams"]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        # ── Recommendation banner (matchup-adjusted if applicable) ──
                        rec = res["recommendation"]
                        if matchup_info:
                            # Adjust weighted avg by matchup factor for final rec
                            adj_avg = res["w_avg"] * matchup_info["factor"]
                            rec = "OVER" if adj_avg > line_val else "UNDER"
                            rec_label = f"Suggested Bet: {rec} &nbsp;{line_val} &nbsp;<span style='font-size:16px;font-weight:400;opacity:0.9;'>({matchup_info['grade']} matchup)</span>"
                        else:
                            adj_avg = res["w_avg"]
                            rec_label = f"Suggested Bet: {rec} &nbsp;{line_val}"

                        color = "#2DC653" if rec == "OVER" else "#D62828"
                        st.markdown(
                            f'<div style="background:{color};color:#fff;padding:14px 20px;'
                            f'border-radius:8px;font-size:22px;font-weight:700;'
                            f'text-align:center;margin-bottom:16px;">'
                            f'{rec_label}</div>',
                            unsafe_allow_html=True,
                        )

                        # ── Key metrics ───────────────────────────────────────
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Weighted Avg",      f"{res['w_avg']:.1f}",
                                  help=f"Weighted avg over {res['window_label']} ({res['window_games']} games)")
                        m2.metric("Weighted Hit Rate", f"{res['w_hit']:.1f}%",
                                  help=f"Hit rate over {res['window_label']} ({res['window_games']} games)")
                        m3.metric(f"{res['window_label']} Avg", f"{res['window_avg']:.1f}",
                                  help=f"Simple avg over {res['window_label']} ({res['window_games']} games)")
                        m4.metric("Std Deviation",     f"{res['std_dev']:.1f}",
                                  help=f"Std dev over {res['window_label']} ({res['window_games']} games)")

                        # ── Matchup-adjusted avg metric ───────────────────────
                        if matchup_info:
                            ma1, ma2, ma3 = st.columns(3)
                            ma1.metric(
                                "Matchup-Adj Avg",
                                f"{adj_avg:.1f}",
                                delta=f"{adj_avg - res['w_avg']:+.1f} vs base avg",
                                help="Weighted avg × opponent's defensive factor",
                            )
                            ma2.metric(
                                f"{opp_sel} Allows",
                                f"{matchup_info['opp_avg']:.1f}",
                                delta=f"{matchup_info['opp_avg'] - matchup_info['lg_avg']:+.1f} vs league",
                                delta_color="inverse",
                            )
                            ma3.metric(
                                "Def Rank",
                                f"#{matchup_info['rank']}" if matchup_info['rank'] else "N/A",
                                help=f"#{matchup_info['rank']} of {matchup_info['n_teams']} teams (higher = softer defense)",
                            )

                        # ── Season split table ────────────────────────────────
                        st.subheader("Season Split")
                        split_rows = []
                        if res["hr_2025"] is not None:
                            split_rows.append({
                                "Season": "2025",
                                "Hit Rate": f"{res['hr_2025']:.1f}%",
                                "Over / Total": f"{int(res['over_2025'])} / {res['total_2025']}",
                                "Average": f"{res['avg_2025']:.1f}",
                                "Weight": "1.0",
                            })
                        if res["hr_2024"] is not None:
                            split_rows.append({
                                "Season": "2024",
                                "Hit Rate": f"{res['hr_2024']:.1f}%",
                                "Over / Total": f"{int(res['over_2024'])} / {res['total_2024']}",
                                "Average": f"{res['avg_2024']:.1f}",
                                "Weight": res["weight_label"],
                            })
                        if split_rows:
                            st.dataframe(pd.DataFrame(split_rows),
                                         use_container_width=True, hide_index=True)

                        st.subheader("Week-by-Week Chart")
                        fig = bar_chart(nfl_df, player_sel, cat_sel, line=line_val, game_window=game_window)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                else:
                    st.info("👈 Set your controls and click **Analyze**.")

        # ── MATCHUP EDGE (formerly tab6) — content block follows below ───────
        with tab6:
            pass  # filled in below

        # ── MATCHUP FINDER (formerly tab8) — content block follows below ─────
        with tab8:
            pass  # filled in below

        # ── PARLAY BUILDER (formerly tab7) — content block follows below ─────
        with tab7:
            pass  # filled in below

        # ── STREAK FINDER ────────────────────────────────────────────────────
        with tab_streak:
            st.subheader("🔥 Streak Finder")
            st.caption("Find players on the longest active over/under streak for any prop line.")

            sf_c1, sf_c2 = st.columns([1, 3])
            with sf_c1:
                sf_cat    = st.selectbox("Stat Category", list(CAT_MAP.keys()),
                                          format_func=str.title, key="sf_cat")
                sf_line   = st.number_input("Prop Line", min_value=0.0, value=65.5,
                                             step=0.5, format="%.1f", key="sf_line")
                sf_dir    = st.radio("Streak Direction", ["Over", "Under"], horizontal=True,
                                      key="sf_dir")
                sf_season = st.radio("Season", [2025, 2024, "Both"], key="sf_season")
                sf_min    = st.number_input("Min streak length", min_value=1, value=2,
                                             step=1, key="sf_min")
                sf_top    = st.slider("Show top N players", 5, 30, 15, key="sf_top")

            with sf_c2:
                sf_col = CAT_MAP[sf_cat.lower()][0]

                if sf_season == "Both":
                    sf_df = nfl_df.copy()
                else:
                    sf_df = nfl_df[nfl_df["season"] == int(sf_season)].copy()

                def calc_streak(series, line, direction):
                    """Return current active streak length (+ = ongoing, 0 = broken last game)."""
                    vals = series.values
                    streak = 0
                    for v in reversed(vals):
                        hit = (v > line) if direction == "Over" else (v < line)
                        if hit:
                            streak += 1
                        else:
                            break
                    return streak

                streak_rows = []
                for player, grp in sf_df.sort_values(["player_name", "game_id"]).groupby("player_name"):
                    s = calc_streak(grp[sf_col], sf_line, sf_dir)
                    if s >= sf_min:
                        recent_avg = grp[sf_col].tail(s).mean()
                        season_avg = grp[sf_col].mean()
                        streak_rows.append({
                            "Player":      player,
                            "Team":        grp["team"].iloc[-1],
                            "Streak":      s,
                            "Avg During Streak": round(recent_avg, 1),
                            "Season Avg":  round(season_avg, 1),
                            "Prop Line":   sf_line,
                            "Direction":   sf_dir,
                        })

                if not streak_rows:
                    st.info(f"No players found with a {sf_min}+ game {sf_dir} streak on {sf_cat.title()} {sf_line}.")
                else:
                    streak_df = (
                        pd.DataFrame(streak_rows)
                        .sort_values("Streak", ascending=False)
                        .head(sf_top)
                        .reset_index(drop=True)
                    )
                    streak_df.index = range(1, len(streak_df) + 1)

                    st.markdown(
                        f"**{len(streak_rows)} players** with an active {sf_dir} streak "
                        f"≥ {sf_min} games on **{sf_cat.title()} {sf_line}** — showing top {sf_top}."
                    )
                    st.dataframe(streak_df, use_container_width=True)

                    # Bar chart of streak lengths
                    fig_s, ax_s = plt.subplots(figsize=(9, max(3, len(streak_df) * 0.45)))
                    bar_c_s = C_OVER if sf_dir == "Over" else C_LINE
                    bars_s = ax_s.barh(
                        streak_df["Player"][::-1], streak_df["Streak"][::-1],
                        color=bar_c_s, alpha=0.85, edgecolor="white", linewidth=0.4,
                    )
                    for bar, row in zip(bars_s, streak_df.iloc[::-1].itertuples()):
                        ax_s.text(
                            bar.get_width() + 0.1,
                            bar.get_y() + bar.get_height() / 2,
                            f"{row.Streak}g · avg {row._4:.1f}",
                            va="center", fontsize=8,
                        )
                    ax_s.set_xlabel("Active Streak (games)", fontsize=9)
                    ax_s.set_title(
                        f"Active {sf_dir} Streaks — {sf_cat.title()} {sf_line}",
                        fontsize=11, fontweight="bold",
                    )
                    ax_s.spines["top"].set_visible(False)
                    ax_s.spines["right"].set_visible(False)
                    ax_s.grid(axis="x", linestyle="--", alpha=0.35)
                    plt.tight_layout()
                    st.pyplot(fig_s, use_container_width=True)
                    plt.close(fig_s)

        # ── VEGAS LINES (tab_vegas) — content block follows below ────────────
        with tab_vegas:
            pass  # filled in below


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TAB 2 — PLAYERS
# Sub-tabs: Player Profile · League Leaders · Home/Away Splits · Start/Sit Advisor
# ══════════════════════════════════════════════════════════════════════════════
with main_players:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        tab2, tab4, tab10, tab11 = st.tabs([
            "👤 Player Profile",
            "🏆 League Leaders",
            "🏠 Home/Away Splits",
            "🏆 Start/Sit Advisor",
        ])

        # ── PLAYER PROFILE ────────────────────────────────────────────────────
        with tab2:
            all_players2 = sorted(nfl_df["player_name"].unique())
            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.subheader("Player")
                pp_player = st.selectbox(
                    "Select player", all_players2,
                    index=all_players2.index("Drake Maye") if "Drake Maye" in all_players2 else 0,
                    key="pp_player",
                )
                pp_cat = st.selectbox(
                    "Trend stat", list(CAT_MAP.keys()),
                    format_func=str.title, key="pp_cat",
                )
                pp_season = st.radio("Season filter", ["Both", "2024", "2025"], key="pp_season")

            with col_b:
                pdf = find_player(nfl_df, pp_player)
                if pdf.empty:
                    st.error("Player not found.")
                else:
                    full = pdf["player_name"].iloc[0]
                    team = pdf.sort_values("game_id")["team"].iloc[-1]
                    p24  = pdf[pdf["season"] == 2024]
                    p25  = pdf[pdf["season"] == 2025]

                    st.subheader(f"{full}  ·  {team}")
                    h1, h2, h3, h4, h5 = st.columns(5)
                    h1.metric("Games (2025)", len(p25))
                    h2.metric("Games (2024)", len(p24))
                    h3.metric("2025 Avg Fantasy", f"{p25['fantasy_points'].mean():.1f}" if not p25.empty else "—")
                    h4.metric("2024 Avg Fantasy", f"{p24['fantasy_points'].mean():.1f}" if not p24.empty else "—")
                    changed = pdf["changed_team"].any() if "changed_team" in pdf.columns else False
                    h5.metric("Team Change", "Yes ⚠️" if changed else "No")

                    st.subheader(f"{CAT_MAP[pp_cat.lower()][1]} — All Games Trend")
                    fig2 = trend_chart(nfl_df, pp_player, pp_cat)
                    if fig2:
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)

                    st.subheader("Game Log")
                    if pp_season == "2024":
                        log_df = p24.copy()
                    elif pp_season == "2025":
                        log_df = p25.copy()
                    else:
                        log_df = pdf.copy()

                    display_cols = [
                        "season", "game_id", "team",
                        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                        "rush_attempts", "rush_yards", "rush_tds",
                        "receptions", "targets", "receiving_yards", "receiving_tds",
                        "fantasy_points",
                    ]
                    display_cols = [c for c in display_cols if c in log_df.columns]
                    st.dataframe(log_df[display_cols].reset_index(drop=True),
                                 use_container_width=True, hide_index=True)

        # ── LEAGUE LEADERS ────────────────────────────────────────────────────
        with tab4:
            ll_col1, ll_col2 = st.columns([1, 3])
            with ll_col1:
                st.subheader("Filters")
                ll_season = st.radio("Season", ["2025", "2024", "Both"], key="ll_season")
                ll_stat   = st.selectbox("Stat", list(LEADER_COLS.keys()), key="ll_stat")
                ll_agg    = st.radio("Aggregate by", ["Average", "Total"], key="ll_agg")
                ll_min    = st.number_input("Min games played", min_value=1, value=4, step=1, key="ll_min")
                ll_top    = st.slider("Show top N", 10, 50, 25, key="ll_top")

            with ll_col2:
                stat_col = LEADER_COLS[ll_stat]
                if ll_season == "Both":
                    ll_df = nfl_df.copy()
                else:
                    ll_df = nfl_df[nfl_df["season"] == int(ll_season)].copy()

                games_per_player = ll_df.groupby("player_name")["game_id"].count()
                eligible = games_per_player[games_per_player >= ll_min].index
                ll_df = ll_df[ll_df["player_name"].isin(eligible)]

                if ll_agg == "Average":
                    leaders = (ll_df.groupby("player_name")[stat_col].mean()
                               .sort_values(ascending=False).head(ll_top).reset_index())
                    val_label = f"Avg {ll_stat}"
                else:
                    leaders = (ll_df.groupby("player_name")[stat_col].sum()
                               .sort_values(ascending=False).head(ll_top).reset_index())
                    val_label = f"Total {ll_stat}"

                leaders.columns = ["Player", val_label]
                leaders[val_label] = leaders[val_label].round(1)
                leaders.index = range(1, len(leaders) + 1)

                st.subheader(f"Top {ll_top} — {val_label}  ({ll_season})")

                fig4, ax4 = plt.subplots(figsize=(9, max(4, len(leaders) * 0.35)))
                bar_color = C_2025 if ll_season == "2025" else (C_2024 if ll_season == "2024" else C_TREND)
                bars4 = ax4.barh(leaders["Player"][::-1], leaders[val_label][::-1],
                                 color=bar_color, alpha=0.85, edgecolor="white", linewidth=0.4)
                for bar, val in zip(bars4, leaders[val_label][::-1]):
                    ax4.text(bar.get_width() + leaders[val_label].max() * 0.01,
                             bar.get_y() + bar.get_height() / 2,
                             f"{val:.1f}", va="center", fontsize=7.5)
                ax4.set_xlabel(val_label, fontsize=9)
                ax4.spines["top"].set_visible(False)
                ax4.spines["right"].set_visible(False)
                ax4.grid(axis="x", linestyle="--", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig4, use_container_width=True)
                plt.close(fig4)
                st.dataframe(leaders, use_container_width=True)

        # ── HOME/AWAY SPLITS (formerly tab10) — content block follows below ──
        with tab10:
            pass  # filled in below

        # ── START/SIT ADVISOR (formerly tab11) — content block follows below ─
        with tab11:
            pass  # filled in below


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TAB 3 — TEAMS & LEAGUE
# Sub-tabs: Team Overview · Depth Charts · Injury Report
# ══════════════════════════════════════════════════════════════════════════════
with main_teams:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        tab3, tab_depth, tab9 = st.tabs([
            "🏟️ Team Overview",
            "📋 Depth Charts",
            "🚑 Injury Report",
        ])

        # ── TEAM OVERVIEW ─────────────────────────────────────────────────────
        with tab3:
            t_col1, t_col2 = st.columns([1, 3])
            with t_col1:
                st.subheader("Filters")
                to_season = st.radio("Season", [2025, 2024], key="to_season")
                to_stat   = st.selectbox("Stat to chart", list(LEADER_COLS.keys()), key="to_stat")
                to_team   = st.selectbox("Team spotlight",
                                          ["All"] + sorted(nfl_df["team"].unique().tolist()),
                                          key="to_team")

            with t_col2:
                stat_col   = LEADER_COLS[to_stat]
                stat_label = to_stat

                fig3 = team_bar_chart(nfl_df, to_season, stat_col, stat_label)
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

                season_df = nfl_df[nfl_df["season"] == to_season]
                if to_team != "All":
                    season_df = season_df[season_df["team"] == to_team]

                st.subheader(
                    f"Top 15 Players — {to_stat} ({to_season}"
                    + (f" · {to_team}" if to_team != "All" else "") + ")"
                )
                top_players = (
                    season_df.groupby("player_name")[stat_col]
                    .mean().sort_values(ascending=False).head(15).reset_index()
                    .rename(columns={"player_name": "Player", stat_col: f"Avg {to_stat}"})
                )
                top_players[f"Avg {to_stat}"] = top_players[f"Avg {to_stat}"].round(1)
                st.dataframe(top_players, use_container_width=True, hide_index=True)

                st.subheader(f"Team Summary — {to_season}")
                team_summary = (
                    nfl_df[nfl_df["season"] == to_season]
                    .groupby("team")
                    .agg(
                        Games=("game_id", "nunique"),
                        Players=("player_name", "nunique"),
                        Avg_Fantasy=("fantasy_points", "mean"),
                        Avg_Pass_Yds=("passing_yards", "mean"),
                        Avg_Rush_Yds=("rush_yards", "mean"),
                        Avg_Rec_Yds=("receiving_yards", "mean"),
                    )
                    .round(1)
                    .sort_values("Avg_Fantasy", ascending=False)
                    .reset_index()
                    .rename(columns={"team": "Team"})
                )
                st.dataframe(team_summary, use_container_width=True, hide_index=True)

        # ── DEPTH CHARTS ─────────────────────────────────────────────────────
        with tab_depth:
            st.subheader("📋 Current NFL Depth Charts")
            st.caption("Live from ESPN · QB / RB / WR / TE starters & backups · cached 6 hours")

            with st.spinner("Loading depth charts for all 32 teams…"):
                dc_data = fetch_all_depth_charts()

            if not dc_data:
                st.warning("Could not load depth charts from ESPN. Try again in a moment.")
            else:
                dc_c1, dc_c2 = st.columns([1, 3])
                with dc_c1:
                    dc_team = st.selectbox(
                        "Select Team",
                        sorted(dc_data.keys()),
                        key="dc_team",
                    )
                    dc_pos_filter = st.multiselect(
                        "Positions",
                        ["QB", "RB", "WR", "TE"],
                        default=["QB", "RB", "WR", "TE"],
                        key="dc_pos",
                    )

                with dc_c2:
                    chart = dc_data.get(dc_team, {})
                    if not chart:
                        st.info(f"No depth chart data available for {dc_team}.")
                    else:
                        for pos in ["QB", "RB", "WR", "TE"]:
                            if pos not in dc_pos_filter:
                                continue
                            players = chart.get(pos, [])
                            if not players:
                                continue
                            st.markdown(f"**{pos}**")
                            rows = []
                            for i, name in enumerate(players, 1):
                                # Cross-reference with our game log data
                                p_data = nfl_df[nfl_df["player_name"].str.contains(
                                    name.split(" ")[-1], case=False, na=False
                                )]
                                p_data = p_data[p_data["player_name"].str.contains(
                                    name.split(" ")[0], case=False, na=False
                                )]
                                if not p_data.empty:
                                    p25 = p_data[p_data["season"] == 2025]
                                    p24 = p_data[p_data["season"] == 2024]
                                    stat_col_map = {"QB": "passing_yards", "RB": "rush_yards",
                                                    "WR": "receiving_yards", "TE": "receiving_yards"}
                                    sc = stat_col_map[pos]
                                    avg_25 = f"{p25[sc].mean():.1f}" if not p25.empty else "—"
                                    avg_24 = f"{p24[sc].mean():.1f}" if not p24.empty else "—"
                                    games  = len(p_data)
                                else:
                                    avg_25 = avg_24 = "New/No data"
                                    games  = 0
                                rows.append({
                                    "Depth": f"#{i}",
                                    "Player": name,
                                    "2025 Avg": avg_25,
                                    "2024 Avg": avg_24,
                                    "Games in DB": games,
                                })
                            st.dataframe(pd.DataFrame(rows),
                                         use_container_width=True, hide_index=True)

        # ── INJURY REPORT (formerly tab9) — content block follows below ───────
        with tab9:
            pass  # filled in below


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TAB 4 — SETTINGS & DATA
# Sub-tabs: Data Refresh
# ══════════════════════════════════════════════════════════════════════════════
with main_settings:
    tab5, = st.tabs(["🔄 Data Refresh"])

    with tab5:
        import datetime as _dt

        st.subheader("🔄 Data Refresh")
        st.markdown(
            "Data is pulled **live from the ESPN API** and cached for **1 hour**. "
            "After 1 hour the cache expires and the next page load automatically "
            "fetches the latest games — no action needed week-to-week."
        )
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ℹ️ How it works")
            st.markdown(
                """
- **First load of the day** → scrapes 2024 + 2025 from ESPN (~5 min)
- **Everyone else within that hour** → instant load from cache
- **After 1 hour** → cache expires, next visitor triggers a fresh scrape
- **New games** appear automatically the next time the cache refreshes
                """
            )
        with c2:
            st.markdown("### ⚡ Manual Refresh")
            st.caption("Force a fresh scrape right now — useful after a big game or if data looks out of date.")
            if data_ok:
                season_counts = nfl_df.groupby("season")["game_id"].nunique()
                for season, games in season_counts.items():
                    latest_wk = (
                        nfl_df[nfl_df["season"] == season]["game_id"]
                        .str.split("_", expand=True)[1]
                        .dropna().astype(int).max()
                    )
                    st.success(f"✅ **{season}** — {games} games loaded  ·  latest week: **{latest_wk}**")
                st.metric("Total Players", f"{nfl_df['player_name'].nunique():,}")
            st.divider()
            if st.button("🔄 Refresh Data Now", type="primary",
                         use_container_width=True, key="api_refresh"):
                st.cache_data.clear()
                time.sleep(0.5)
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT BLOCKS FOR SUB-TABS DECLARED INSIDE main_bet / main_players / main_teams
# These must be filled AFTER the with-blocks that declared the tab objects.
# ══════════════════════════════════════════════════════════════════════════════

# ── MATCHUP EDGE (tab6 inside main_bet) ──────────────────────────────────────
with tab6:
    if not data_ok:
        st.info("Load data first using the **Data Refresh** tab.")
    else:
        # ── How it works ──────────────────────────────────────────────────────
        # "Defensive average" = how many yards / TDs that stat category's
        # position group has put up AGAINST each team on average.
        # e.g. passing_yards allowed by defense = avg passing_yards opponents QBs
        # recorded when facing that team.
        #
        # We approximate this from our game log data:
        # For each game_id we know both teams.  We parse the game_id format
        # "{season}_{week}_{away}_{home}" to get opponent per row, then group
        # by opponent to get "avg yards allowed".

        # ── build opponent column ─────────────────────────────────────────────
        @st.cache_data(show_spinner=False)
        def build_opponent_col(nfl):
            df = nfl.copy()
            # game_id format: "2024_01_NYJ_SF"  → away=NYJ, home=SF
            def parse_opponent(row):
                parts = str(row["game_id"]).split("_")
                if len(parts) < 4:
                    return "UNK"
                away, home = parts[2], parts[3]
                return home if row["team"] == away else away
            df["opponent"] = df.apply(parse_opponent, axis=1)
            return df

        nfl_opp = build_opponent_col(nfl_df)

        # ── controls ─────────────────────────────────────────────────────────
        me_col1, me_col2 = st.columns([1, 3])

        with me_col1:
            st.subheader("Matchup Setup")

            me_player = st.selectbox(
                "Player",
                sorted(nfl_df["player_name"].unique()),
                index=sorted(nfl_df["player_name"].unique()).index("Drake Maye")
                if "Drake Maye" in nfl_df["player_name"].values else 0,
                key="me_player",
            )
            me_cat = st.selectbox(
                "Stat Category",
                list(CAT_MAP.keys()),
                format_func=str.title,
                key="me_cat",
            )
            me_line = st.number_input(
                "Prop Line", min_value=0.0, value=200.5, step=0.5,
                format="%.1f", key="me_line",
            )
            me_opp = st.selectbox(
                "Opposing Team (next game)",
                sorted(nfl_opp["team"].unique()),
                key="me_opp",
            )
            me_season = st.radio(
                "Defensive sample season", ["2025", "2024", "Both"],
                key="me_season",
            )
            me_go = st.button("Run Matchup Analysis", type="primary",
                              use_container_width=True, key="me_go")

        with me_col2:
            if me_go:
                col, col_label = CAT_MAP[me_cat.lower()]

                # ── player career stats ───────────────────────────────────────
                pdf = find_player(nfl_df, me_player)
                if pdf.empty:
                    st.error("Player not found.")
                    st.stop()

                full_name = pdf["player_name"].iloc[0]
                p25 = pdf[pdf["season"] == 2025]
                p24 = pdf[pdf["season"] == 2024]

                player_avg_25  = p25[col].mean()  if not p25.empty else None
                player_avg_24  = p24[col].mean()  if not p24.empty else None
                player_last3   = p25[col].tail(3).mean() if not p25.empty else pdf[col].tail(3).mean()
                player_all_avg = pdf[col].mean()

                # ── defensive averages allowed vs this stat ───────────────────
                # filter to games where the opposing team = me_opp
                if me_season == "2025":
                    def_df = nfl_opp[nfl_opp["season"] == 2025]
                elif me_season == "2024":
                    def_df = nfl_opp[nfl_opp["season"] == 2024]
                else:
                    def_df = nfl_opp.copy()

                # rows where players faced me_opp
                vs_opp = def_df[def_df["opponent"] == me_opp]

                # league-wide average allowed per game for this stat
                league_def_avg = (
                    def_df.groupby("opponent")[col].mean().mean()
                    if not def_df.empty else 0
                )

                opp_allowed_avg  = vs_opp[col].mean()  if not vs_opp.empty else 0
                opp_allowed_std  = vs_opp[col].std()   if not vs_opp.empty else 0
                opp_games        = len(vs_opp)

                # defensive rank: lower allowed = tougher defense
                def_rank_df = def_df.groupby("opponent")[col].mean().sort_values(ascending=False)
                opp_rank    = (
                    def_rank_df.index.tolist().index(me_opp) + 1
                    if me_opp in def_rank_df.index else None
                )
                total_teams = len(def_rank_df)

                # ── matchup edge score ─────────────────────────────────────────
                # Edge = player_last3 - opp_allowed_avg
                # Positive → player is likely to exceed what the defense allows
                edge = player_last3 - opp_allowed_avg if opp_allowed_avg else None
                edge_vs_line = player_last3 - me_line

                # ── recommendation ────────────────────────────────────────────
                # weighted: 60% last-3 vs line, 40% edge vs defense
                if edge is not None:
                    score = 0.6 * edge_vs_line + 0.4 * edge
                else:
                    score = edge_vs_line
                recommendation = "OVER" if score > 0 else "UNDER"
                conf_color = "#2DC653" if recommendation == "OVER" else "#D62828"

                # ── banner ────────────────────────────────────────────────────
                st.markdown(
                    f'<div style="background:{conf_color};color:#fff;padding:14px 20px;'
                    f'border-radius:8px;font-size:22px;font-weight:700;'
                    f'text-align:center;margin-bottom:16px;">'
                    f'Matchup Suggestion: {recommendation} &nbsp;{me_line}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── key metrics ───────────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Player Last 3 Avg",        f"{player_last3:.1f}")
                k2.metric(f"{me_opp} Allows (Avg)",   f"{opp_allowed_avg:.1f}",
                          delta=f"{opp_allowed_avg - league_def_avg:+.1f} vs league",
                          delta_color="inverse")
                k3.metric("Edge vs Defense",
                          f"{edge:+.1f}" if edge is not None else "N/A")
                k4.metric("Prop Line Gap",             f"{edge_vs_line:+.1f}")

                st.divider()

                # ── two column detail ─────────────────────────────────────────
                d1, d2 = st.columns(2)

                with d1:
                    st.subheader(f"📌 {full_name}")
                    player_rows = [
                        {"Metric": "2025 Season Avg",    "Value": f"{player_avg_25:.1f}"  if player_avg_25  is not None else "—"},
                        {"Metric": "2024 Season Avg",    "Value": f"{player_avg_24:.1f}"  if player_avg_24  is not None else "—"},
                        {"Metric": "Last 3 Games Avg",   "Value": f"{player_last3:.1f}"},
                        {"Metric": "Career Avg (both)",  "Value": f"{player_all_avg:.1f}"},
                        {"Metric": "Prop Line",          "Value": str(me_line)},
                        {"Metric": "Last-3 vs Line",     "Value": f"{edge_vs_line:+.1f}"},
                    ]
                    st.dataframe(pd.DataFrame(player_rows),
                                 use_container_width=True, hide_index=True)

                with d2:
                    st.subheader(f"🛡️ {me_opp} Defense")
                    def_rows = [
                        {"Metric": f"Avg {col_label} Allowed",    "Value": f"{opp_allowed_avg:.1f}"},
                        {"Metric": "Std Dev (allowed)",            "Value": f"{opp_allowed_std:.1f}"},
                        {"Metric": "Sample Games",                 "Value": str(opp_games)},
                        {"Metric": "League Avg Allowed",           "Value": f"{league_def_avg:.1f}"},
                        {"Metric": f"Defensive Rank (of {total_teams})",
                                                                   "Value": f"#{opp_rank}" if opp_rank else "N/A"},
                        {"Metric": "Edge vs Defense",              "Value": f"{edge:+.1f}" if edge is not None else "N/A"},
                    ]
                    st.dataframe(pd.DataFrame(def_rows),
                                 use_container_width=True, hide_index=True)

                st.divider()

                # ── chart: player game-by-game vs opp defensive avg line ──────
                st.subheader(f"{full_name}  —  {col_label}  vs  {me_opp} Defensive Avg")

                chart_df = pdf.copy().reset_index(drop=True)
                chart_df["game_num"] = range(1, len(chart_df) + 1)

                fig6, ax6 = plt.subplots(figsize=(12, 4))

                bar_colors = [C_2025 if s == 2025 else C_2024 for s in chart_df["season"]]
                ax6.bar(chart_df["game_num"], chart_df[col],
                        color=bar_colors, alpha=0.65, edgecolor="white",
                        linewidth=0.5, zorder=2, label="_nolegend_")

                # prop line
                ax6.axhline(me_line, color=C_LINE, linewidth=1.8, linestyle="-",
                            label=f"Prop Line: {me_line}", zorder=4)

                # opp defensive avg
                if opp_allowed_avg:
                    ax6.axhline(opp_allowed_avg, color="#f59e0b", linewidth=1.8,
                                linestyle="--",
                                label=f"{me_opp} Avg Allowed: {opp_allowed_avg:.1f}",
                                zorder=4)

                # player season avg
                ax6.axhline(player_all_avg, color=C_AVG, linewidth=1.4,
                            linestyle=":", label=f"Player Avg: {player_all_avg:.1f}",
                            zorder=3)

                # season boundary
                boundary = chart_df[chart_df["season"] == 2025]["game_num"].min()
                if pd.notna(boundary) and boundary > 1:
                    ax6.axvline(boundary - 0.5, color="#888", linewidth=1,
                                linestyle=":", label="2024 → 2025")

                ax6.set_xlabel("Game #", fontsize=9)
                ax6.set_ylabel(col_label, fontsize=9)
                ax6.legend(fontsize=8, framealpha=0.85)
                ax6.spines["top"].set_visible(False)
                ax6.spines["right"].set_visible(False)
                ax6.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
                plt.tight_layout()
                st.pyplot(fig6, use_container_width=True)
                plt.close(fig6)

                # ── games vs this opponent (if any in history) ────────────────
                hist = nfl_opp[
                    (nfl_opp["player_name"] == full_name) &
                    (nfl_opp["opponent"] == me_opp)
                ]
                if not hist.empty:
                    st.subheader(f"📋 {full_name}  past games vs  {me_opp}")
                    hist_cols = ["season", "game_id", col, "fantasy_points"]
                    hist_cols = [c for c in hist_cols if c in hist.columns]
                    st.dataframe(hist[hist_cols].reset_index(drop=True),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info(f"No historical games found for {full_name} vs {me_opp} in this dataset.")

            else:
                st.info("👈 Set the player, stat, prop line, and opposing team — then click **Run Matchup Analysis**.")


# ══════════════════════════════════════════════════════════════════════════════
# PARLAY BUILDER (tab7 inside main_bet)
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        # ── session-state parlay list ─────────────────────────────────────────
        if "parlay_legs" not in st.session_state:
            st.session_state["parlay_legs"] = []   # list of dicts

        all_players_pb = sorted(nfl_df["player_name"].unique())

        # ── PARLAY MATH HELPERS ───────────────────────────────────────────────
        def american_to_prob(odds: int) -> float:
            """Convert American odds to implied probability (0-1)."""
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)

        def prob_to_american(p: float) -> int:
            """Convert probability (0-1) back to American odds.
            Clamps output so favourites never exceed -350 and underdogs never
            exceed +350, keeping lines within a realistic sportsbook range.
            """
            if p <= 0 or p >= 1:
                return 0
            if p >= 0.5:
                return max(-350, -round((p / (1 - p)) * 100))
            else:
                return min(350, round(((1 - p) / p) * 100))

        def parlay_payout(leg_odds: list[int], stake: float) -> float:
            """
            Calculate parlay payout from a list of American odds and a stake.
            Converts each leg to a decimal multiplier, multiplies them all, then
            applies to the stake.
            """
            multiplier = 1.0
            for o in leg_odds:
                if o > 0:
                    multiplier *= (o / 100 + 1)
                else:
                    multiplier *= (100 / abs(o) + 1)
            return round(stake * multiplier, 2)

        def confidence_label(prob: float) -> tuple[str, str]:
            """Return (label, hex-color) for a combined win probability."""
            if prob >= 0.55:
                return "Strong", "#2DC653"
            if prob >= 0.42:
                return "Moderate", "#f59e0b"
            return "Risky", "#D62828"

        def score_leg(nfl, player_name, category, line, use_weighted=True):
            """
            Returns a dict with hit_rate (%), weighted_avg, recommendation,
            implied_prob for this leg based on historical data.
            """
            col = CAT_MAP[category.lower()][0]
            pdf = find_player(nfl, player_name)
            if pdf.empty:
                return None

            vals = pdf[col].values
            wts  = pdf["weight"].values

            if use_weighted:
                w_avg = np.average(vals, weights=wts)
                w_hit = np.average((vals > line).astype(float), weights=wts)
            else:
                w_avg = vals.mean()
                w_hit = (vals > line).mean()

            rec = "OVER" if w_avg > line else "UNDER"
            # implied prob: if bet OVER, use hit rate; if UNDER use (1 - hit rate)
            implied = w_hit if rec == "OVER" else 1 - w_hit
            # cap between 22% and 78% → American odds stay within ±350
            implied = max(0.222, min(0.778, implied))

            return {
                "player":      pdf["player_name"].iloc[0],
                "category":    category,
                "line":        line,
                "col":         col,
                "w_avg":       round(float(w_avg), 1),
                "hit_rate_pct": round(float(w_hit) * 100, 1),
                "recommendation": rec,
                "implied_prob": round(implied, 4),
                "american_odds": prob_to_american(implied),
            }

        # ─────────────────────────────────────────────────────────────────────
        # LAYOUT: add-leg panel (left) | parlay slip (right)
        # ─────────────────────────────────────────────────────────────────────
        pb_left, pb_right = st.columns([1, 2])

        # ── LEFT: Add a leg ───────────────────────────────────────────────────
        with pb_left:
            st.subheader("➕ Add a Leg")

            pb_player = st.selectbox(
                "Player",
                all_players_pb,
                index=all_players_pb.index("Drake Maye")
                if "Drake Maye" in all_players_pb else 0,
                key="pb_player",
            )
            pb_cat = st.selectbox(
                "Stat",
                list(CAT_MAP.keys()),
                format_func=str.title,
                key="pb_cat",
            )
            pb_line = st.number_input(
                "Prop Line", min_value=0.0, value=200.5, step=0.5,
                format="%.1f", key="pb_line",
            )
            pb_weighted = st.toggle(
                "Season weighting", value=True, key="pb_weighted"
            )

            add_leg = st.button(
                "➕ Add to Parlay", type="primary",
                use_container_width=True, key="pb_add",
            )

            if add_leg:
                if len(st.session_state["parlay_legs"]) >= 8:
                    st.warning("Maximum 8 legs reached.")
                else:
                    result = score_leg(
                        nfl_df, pb_player, pb_cat, pb_line, pb_weighted
                    )
                    if result is None:
                        st.error("Player not found.")
                    else:
                        # prevent duplicate legs
                        exists = any(
                            l["player"] == result["player"]
                            and l["category"] == result["category"]
                            and l["line"] == result["line"]
                            for l in st.session_state["parlay_legs"]
                        )
                        if exists:
                            st.warning("This exact leg is already in your parlay.")
                        else:
                            st.session_state["parlay_legs"].append(result)
                            st.rerun()

            # clear button
            if st.session_state["parlay_legs"]:
                st.divider()
                if st.button("🗑️ Clear All Legs", use_container_width=True, key="pb_clear"):
                    st.session_state["parlay_legs"] = []
                    st.rerun()

        # ── RIGHT: Parlay slip ────────────────────────────────────────────────
        with pb_right:
            legs = st.session_state["parlay_legs"]

            if not legs:
                st.info("👈 Add at least 2 legs from the left panel to build your parlay.")
            else:
                st.subheader(f"🎰 Parlay Slip  —  {len(legs)} leg{'s' if len(legs) > 1 else ''}")

                # ── Per-leg table with remove buttons ─────────────────────────
                remove_idx = None
                for i, leg in enumerate(legs):
                    rec_color = "#2DC653" if leg["recommendation"] == "OVER" else "#D62828"
                    c1, c2, c3, c4, c5, c6 = st.columns([2, 1.2, 1, 1, 1, 0.5])
                    c1.markdown(f"**{leg['player']}**")
                    c2.markdown(f"{leg['category'].title()} {leg['recommendation']} **{leg['line']}**")
                    c3.metric("Wtd Avg", leg["w_avg"])
                    c4.metric("Hit Rate", f"{leg['hit_rate_pct']}%")
                    c5.markdown(
                        f"<span style='color:{rec_color};font-weight:700;font-size:15px'>"
                        f"{leg['recommendation']}</span>",
                        unsafe_allow_html=True,
                    )
                    if c6.button("✕", key=f"rm_{i}"):
                        remove_idx = i

                if remove_idx is not None:
                    st.session_state["parlay_legs"].pop(remove_idx)
                    st.rerun()

                st.divider()

                # ── Parlay math ───────────────────────────────────────────────
                if len(legs) >= 2:
                    combined_prob = 1.0
                    for leg in legs:
                        combined_prob *= leg["implied_prob"]

                    combined_american = prob_to_american(combined_prob)
                    conf_label, conf_color = confidence_label(combined_prob)

                    # stake input
                    stake = st.number_input(
                        "Stake ($)", min_value=1.0, value=10.0, step=5.0,
                        format="%.2f", key="pb_stake",
                    )
                    payout = parlay_payout([l["american_odds"] for l in legs], stake)
                    profit = round(payout - stake, 2)

                    # ── summary metrics ───────────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Combined Win Prob", f"{combined_prob*100:.1f}%")
                    m2.metric("Parlay Odds",       f"+{combined_american}" if combined_american > 0 else str(combined_american))
                    m3.metric("Potential Payout",  f"${payout:,.2f}")
                    m4.metric("Potential Profit",  f"${profit:,.2f}")

                    # ── confidence banner ─────────────────────────────────────
                    st.markdown(
                        f'<div style="background:{conf_color};color:#fff;'
                        f'padding:12px 20px;border-radius:8px;'
                        f'font-size:20px;font-weight:700;text-align:center;'
                        f'margin:12px 0;">'
                        f'Parlay Confidence: {conf_label} &nbsp;·&nbsp; '
                        f'{combined_prob*100:.1f}% est. probability'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── leg breakdown chart ───────────────────────────────────
                    st.subheader("Leg Breakdown")
                    fig7, ax7 = plt.subplots(figsize=(9, max(3, len(legs) * 0.55)))
                    labels  = [
                        f"{l['player']}\n{l['category'].title()} {l['recommendation']} {l['line']}"
                        for l in legs
                    ]
                    probs   = [l["implied_prob"] * 100 for l in legs]
                    colors  = [C_OVER if l["recommendation"] == "OVER" else C_LINE for l in legs]

                    bars = ax7.barh(labels[::-1], probs[::-1], color=colors[::-1],
                                   alpha=0.85, edgecolor="white", linewidth=0.5)
                    ax7.axvline(50, color="#888", linewidth=1, linestyle="--", label="50% line")
                    for bar, val in zip(bars, probs[::-1]):
                        ax7.text(
                            bar.get_width() + 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%",
                            va="center", fontsize=8,
                        )
                    ax7.set_xlabel("Estimated Win Probability (%)", fontsize=9)
                    ax7.set_xlim(0, 105)
                    ax7.spines["top"].set_visible(False)
                    ax7.spines["right"].set_visible(False)
                    ax7.grid(axis="x", linestyle="--", alpha=0.35)
                    ax7.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig7, use_container_width=True)
                    plt.close(fig7)

                    # ── full leg detail table ─────────────────────────────────
                    st.subheader("Full Leg Details")
                    detail_rows = []
                    for leg in legs:
                        detail_rows.append({
                            "Player":        leg["player"],
                            "Stat":          leg["category"].title(),
                            "Line":          leg["line"],
                            "Pick":          leg["recommendation"],
                            "Wtd Avg":       leg["w_avg"],
                            "Hit Rate":      f"{leg['hit_rate_pct']}%",
                            "Leg Odds":      f"+{leg['american_odds']}" if leg["american_odds"] > 0 else str(leg["american_odds"]),
                            "Leg Prob":      f"{leg['implied_prob']*100:.1f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ── risk note ─────────────────────────────────────────────
                    st.caption(
                        "⚠️ Probabilities are estimated from historical hit rates using "
                        "2024/2025 weighted game logs. They are not guaranteed outcomes. "
                        "Bet responsibly."
                    )
                else:
                    st.info("Add at least **2 legs** to calculate parlay odds.")


# ══════════════════════════════════════════════════════════════════════════════
# MATCHUP FINDER (tab8 inside main_bet)
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        # ── helpers ───────────────────────────────────────────────────────────
        @st.cache_data(show_spinner=False)
        def build_defense_table(nfl):
            df = nfl.copy()
            def get_opp(row):
                parts = str(row["game_id"]).split("_")
                if len(parts) < 4: return "UNK"
                away, home = parts[2], parts[3]
                return home if row["team"] == away else away
            df["opponent"] = df.apply(get_opp, axis=1)
            return df

        @st.cache_data(ttl=3600, show_spinner=False)
        def fetch_this_weeks_games():
            """
            Finds the next/current NFL week and returns upcoming (unplayed) games.
            Checks the next calendar year first so the upcoming season's schedule
            shows during the offseason (e.g. 2025 schedule visible in May 2025).
            Falls back to the most recent completed week if nothing is found.
            """
            import datetime as _dt
            today     = _dt.date.today()
            cur_year  = today.year if today.month >= 9 else today.year - 1
            next_year = cur_year + 1

            def _scrape_year(year):
                """
                Walk weeks 1-18 for the given year.
                Returns (upcoming_this_week, last_completed_list).
                Stops as soon as it finds the first week that has ANY upcoming
                (not-yet-played) game — so we never accumulate the whole season.
                """
                last_completed = []
                for week in range(1, 19):
                    url = (
                        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                        f"?seasontype=2&week={week}&dates={year}"
                    )
                    data = _get_json(url)
                    if not data:
                        continue
                    events = data.get("events", [])
                    if not events:
                        break
                    week_upcoming = []
                    for e in events:
                        comp  = e["competitions"][0]
                        done  = comp["status"]["type"]["completed"]
                        teams = {c["homeAway"]: c["team"]["abbreviation"]
                                 for c in comp["competitors"]}
                        entry = {
                            "home":      teams.get("home", "UNK"),
                            "away":      teams.get("away", "UNK"),
                            "date":      e["date"][:10],
                            "week":      week,
                            "season":    year,
                            "completed": done,
                            "name":      e.get("shortName", e.get("name", "")),
                            "espn_id":   e.get("id", ""),
                        }
                        if not done:
                            week_upcoming.append(entry)
                        else:
                            last_completed.append(entry)
                    # As soon as we find a week with upcoming games, return just those
                    if week_upcoming:
                        return week_upcoming, last_completed
                return [], last_completed

            # Try current season first
            upcoming, last_completed = _scrape_year(cur_year)

            # If no upcoming games in current year, check next year's schedule
            # (handles offseason when ESPN has already posted the upcoming season)
            if not upcoming:
                upcoming_next, _ = _scrape_year(next_year)
                if upcoming_next:
                    upcoming = upcoming_next

            return upcoming if upcoming else last_completed[-16:]  # final fallback

        @st.cache_data(ttl=3600, show_spinner=False)
        def fetch_game_odds(espn_id: str) -> dict:
            """
            Pull game-level odds (spread + over/under) from ESPN's free
            odds endpoint for a single game event ID.
            Returns a dict with keys: over_under, home_spread, away_spread,
            book_name.  All values are None if not available.
            """
            url = (
                f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
                f"/events/{espn_id}/competitions/{espn_id}/odds"
            )
            data = _get_json(url)
            result = {"over_under": None, "home_spread": None,
                      "away_spread": None, "book_name": None}
            if not data:
                return result
            # ESPN returns a list of odds providers; take the first (consensus)
            items = data.get("items", [])
            if not items:
                return result
            provider = items[0]
            result["book_name"]   = provider.get("provider", {}).get("name", "ESPN")
            result["over_under"]  = provider.get("overUnder")
            result["home_spread"] = provider.get("homeTeamOdds", {}).get("spreadOdds")
            result["away_spread"] = provider.get("awayTeamOdds", {}).get("spreadOdds")
            # spreadOdds is a signed float, e.g. -3.5; try "spread" key as fallback
            if result["home_spread"] is None:
                result["home_spread"] = provider.get("spread")
            return result

        # ── build defensive stats table ───────────────────────────────────────
        nfl_def = build_defense_table(nfl_df)

        st.subheader("🎯 Matchup Finder — This Week's Best Props")
        st.caption(
            "Pulls this week's schedule from ESPN, finds the softest defensive matchup "
            "for each game, and suggests the best player + prop line to target. "
            "Add your Odds API key to show **real book lines** instead of model projections."
        )

        # ── controls ──────────────────────────────────────────────────────────
        mf_c1, mf_c2 = st.columns([1, 4])
        with mf_c1:
            mf_stat      = st.selectbox("Stat category", list(CAT_MAP.keys()),
                                         format_func=str.title, key="mf_stat")
            mf_season    = st.radio("Defensive sample", [2025, 2024, "Both"],
                                     key="mf_season")
            mf_min_games = st.number_input("Min games sample", 1, 18, 4,
                                            key="mf_min")
            mf_min_player = st.number_input("Min player games", 1, 18, 3,
                                             key="mf_min_player")
            st.divider()
            st.caption("**Odds API key** — optional. When set, uses real DraftKings prop lines instead of model projections.")
            _mf_secret_key = st.secrets.get("ODDS_API_KEY", "") if hasattr(st, "secrets") else ""
            mf_api_key = st.text_input(
                "The Odds API key",
                value=_mf_secret_key,
                type="password",
                key="mf_api_key",
                placeholder="Leave blank to use model-projected lines",
            )

        with mf_c2:
            col, col_label = CAT_MAP[mf_stat.lower()]

            # Build defense ranks
            if mf_season == "Both":
                def_sample = nfl_def.copy()
            else:
                def_sample = nfl_def[nfl_def["season"] == int(mf_season)]

            def_agg = (
                def_sample.groupby("opponent")
                .agg(avg_allowed=(col, "mean"), games=(col, "count"))
                .reset_index()
            )
            def_agg = def_agg[def_agg["games"] >= mf_min_games]
            league_avg = def_agg["avg_allowed"].mean()
            def_agg = def_agg.sort_values("avg_allowed", ascending=False).reset_index(drop=True)
            def_agg["rank"] = def_agg.index + 1
            def_ranks = dict(zip(def_agg["opponent"], def_agg["rank"]))
            def_avgs  = dict(zip(def_agg["opponent"], def_agg["avg_allowed"]))
            total_teams = len(def_agg)

            # Fetch schedule, odds, depth charts, and (optionally) real prop lines
            with st.spinner("Fetching schedule, odds, and depth charts from ESPN..."):
                games = fetch_this_weeks_games()
                for g in games:
                    if g.get("espn_id"):
                        g["odds"] = fetch_game_odds(g["espn_id"])
                    else:
                        g["odds"] = {"over_under": None, "home_spread": None,
                                     "away_spread": None, "book_name": None}
                depth_charts = fetch_all_depth_charts()

            # Build a lookup: player name (lower) → {cat → real book line}
            # Only populated when an Odds API key is provided.
            mf_real_lines: dict = {}   # {"patrick mahomes": {"pass yards": 287.5, ...}, ...}
            if mf_api_key.strip():
                with st.spinner("Fetching real prop lines from The Odds API…"):
                    raw_prop_rows = fetch_odds_api_props(mf_api_key.strip())
                for rr in raw_prop_rows:
                    key = rr["player_raw"].lower().strip()
                    if key not in mf_real_lines:
                        mf_real_lines[key] = {}
                    mf_real_lines[key][rr["cat"]] = rr["line"]
                if mf_real_lines:
                    st.success(
                        f"✅ Real prop lines loaded for **{len(mf_real_lines)}** players "
                        f"from {raw_prop_rows[0]['bookmaker'] if raw_prop_rows else 'book'}."
                    )
                else:
                    st.warning("Odds API returned no prop lines — using model-projected lines.")

            # Helper: map stat col → depth chart position(s)
            _COL_TO_POS = {
                "passing_yards":   ["QB"],
                "passing_tds":     ["QB"],
                "rush_yards":      ["RB"],
                "receiving_yards": ["WR", "TE", "RB"],
                "receptions":      ["WR", "TE", "RB"],
                "fantasy_points":  ["QB", "RB", "WR", "TE"],
            }

            def depth_chart_players(team, stat_col, max_rank=3):
                """Return the top-N depth chart players for a team/stat combo."""
                chart = depth_charts.get(team, {})
                positions = _COL_TO_POS.get(stat_col, [])
                players = []
                for pos in positions:
                    players.extend(chart.get(pos, [])[:max_rank])
                return players   # ordered starter-first; empty = no depth data

            if not games:
                st.warning("No schedule data available. ESPN API may be temporarily unavailable.")
            else:
                is_upcoming = any(not g["completed"] for g in games)
                week_num    = games[0]["week"]
                label       = f"Week {week_num} Upcoming Games" if is_upcoming else f"Week {week_num} (Most Recent — Offseason)"

                # ── Game odds summary cards ───────────────────────────────────
                ou_games = [g for g in games if g["odds"].get("over_under")]
                if ou_games:
                    st.markdown(f"**{label}** — {len(games)} games  ·  "
                                f"O/U lines from ESPN")
                    ou_cols = st.columns(min(len(ou_games), 4))
                    for i, g in enumerate(ou_games[:8]):
                        ou  = g["odds"]["over_under"]
                        spd = g["odds"]["home_spread"]
                        spd_str = f"  ·  Spread: {spd:+.1f}" if spd is not None else ""
                        col_idx = i % min(len(ou_games), 4)
                        ou_cols[col_idx].markdown(
                            f'<div style="background:#f7f8fa;border:1px solid #e5e7eb;'
                            f'border-radius:6px;padding:8px 10px;margin-bottom:6px;font-size:12px;">'
                            f'<b>{g["away"]} @ {g["home"]}</b><br>'
                            f'<span style="color:#3b82d4;font-weight:700;">O/U {ou}</span>'
                            f'{spd_str}</div>',
                            unsafe_allow_html=True,
                        )
                    st.divider()
                else:
                    st.markdown(f"**{label}** — {len(games)} games")

                # ── Per-game prop suggestions ─────────────────────────────────
                prop_rows = []
                for game in games:
                    home_disp, away_disp = game["home"], game["away"]   # original for display
                    home, away = _norm_team(home_disp), _norm_team(away_disp)  # normalised for data lookup

                    for offense_team, defense_team in [(away, home), (home, away)]:
                        def_avg = def_avgs.get(defense_team, league_avg)
                        def_rank = def_ranks.get(defense_team, total_teams)
                        matchup_grade = (
                            "🟢 Soft" if def_avg > league_avg * 1.10
                            else ("🔴 Tough" if def_avg < league_avg * 0.90 else "🟡 Average")
                        )

                        # Find best player on offense_team for this stat
                        # Prefer depth-chart starters; fall back to historical leaders
                        dc_names = depth_chart_players(offense_team, col, max_rank=3)

                        # ── Step 1: players who played for this team in 2025/2024 ──
                        team_players = nfl_def[
                            (nfl_def["team"] == offense_team) &
                            (nfl_def["season"] == 2025) &
                            (nfl_def[col] > 0)
                        ]
                        if team_players.empty:
                            team_players = nfl_def[
                                (nfl_def["team"] == offense_team) &
                                (nfl_def["season"] == 2024) &
                                (nfl_def[col] > 0)
                            ]

                        # ── Step 2: for depth chart players not found on this team,
                        #    look them up by name across ALL teams (team changers / new signings)
                        new_signings = []   # rows for players found on other teams
                        if dc_names:
                            found_names = set(team_players["player_name"].unique()) if not team_players.empty else set()
                            for dc_name in dc_names:
                                if dc_name not in found_names:
                                    # fuzzy-ish: try exact match first, then last-name match
                                    anywhere = nfl_def[
                                        (nfl_def["player_name"] == dc_name) &
                                        (nfl_def[col] > 0)
                                    ]
                                    if anywhere.empty:
                                        last_nm = dc_name.split(" ")[-1]
                                        anywhere = nfl_def[
                                            nfl_def["player_name"].str.endswith(last_nm, na=False) &
                                            (nfl_def[col] > 0)
                                        ]
                                    if not anywhere.empty:
                                        new_signings.append(anywhere)

                        if new_signings:
                            signing_df = pd.concat(new_signings)
                            # Down-weight new signings: multiply their mean by 0.75
                            # (they're on a new team/system — less predictable)
                            signing_agg = (
                                signing_df.groupby("player_name")[col]
                                .agg(["mean", "count", "std"])
                                .reset_index()
                            )
                            signing_agg["mean"] = signing_agg["mean"] * 0.75
                            signing_agg["new_signing"] = True
                            if not team_players.empty:
                                team_agg = (
                                    team_players.groupby("player_name")[col]
                                    .agg(["mean", "count", "std"])
                                    .reset_index()
                                )
                                team_agg["new_signing"] = False
                                team_players_combined = pd.concat([team_agg, signing_agg])
                            else:
                                team_players_combined = signing_agg
                        else:
                            if team_players.empty:
                                continue
                            team_players_combined = (
                                team_players.groupby("player_name")[col]
                                .agg(["mean", "count", "std"])
                                .reset_index()
                            )
                            team_players_combined["new_signing"] = False

                        p_agg = team_players_combined[team_players_combined["count"] >= mf_min_player]
                        if p_agg.empty:
                            # relax min games for new signings
                            p_agg = team_players_combined[team_players_combined["count"] >= 1]
                        if p_agg.empty:
                            continue

                        # ── Step 3: sort by depth chart order if available ──
                        if dc_names:
                            dc_filtered = p_agg[p_agg["player_name"].isin(dc_names)]
                            if not dc_filtered.empty:
                                dc_filtered = dc_filtered.copy()
                                dc_filtered["dc_rank"] = dc_filtered["player_name"].apply(
                                    lambda n: dc_names.index(n) if n in dc_names else 99
                                )
                                p_agg = dc_filtered.sort_values("dc_rank")
                            else:
                                p_agg = p_agg.sort_values("mean", ascending=False)
                        else:
                            p_agg = p_agg.sort_values("mean", ascending=False)

                        best = p_agg.iloc[0]
                        is_new_signing = bool(best.get("new_signing", False))

                        # Prop line = player's weighted avg adjusted for matchup
                        player_avg = float(best["mean"])
                        # Undo the 0.75 down-weight for display purposes
                        if is_new_signing:
                            player_avg_display = player_avg / 0.75
                        else:
                            player_avg_display = player_avg
                        matchup_factor = def_avg / league_avg if league_avg > 0 else 1.0

                        # Get last-3 avg for this player (from any team)
                        p_df   = nfl_def[nfl_def["player_name"] == best["player_name"]]
                        p_2025 = p_df[p_df["season"] == 2025]
                        last3  = float(p_2025[col].tail(3).mean()) if not p_2025.empty else player_avg_display

                        # ── Line: real book line if available, else model projection ──
                        proj = player_avg_display * matchup_factor
                        player_key = best["player_name"].lower().strip()
                        real_line = mf_real_lines.get(player_key, {}).get(
                            CAT_MAP[mf_stat.lower()][1].lower()  # try label first
                            if False else mf_stat.lower(),        # use cat key
                        )
                        # Also try partial name match (handles "Patrick Mahomes" vs "P. Mahomes")
                        if real_line is None:
                            last_name = player_key.split()[-1]
                            for k, v in mf_real_lines.items():
                                if k.endswith(last_name) and mf_stat.lower() in v:
                                    real_line = v[mf_stat.lower()]
                                    break

                        if real_line is not None:
                            suggested_line = real_line
                            line_source    = "📖 Book"
                        else:
                            if col == "passing_yards":
                                increments = [i + 0.5 for i in range(50, 500, 25)]
                            elif col in ("rush_yards", "receiving_yards"):
                                increments = [i + 0.5 for i in range(0, 250, 10)]
                            elif col == "receptions":
                                increments = [i + 0.5 for i in range(0, 20, 1)]
                            elif col == "passing_tds":
                                increments = [0.5, 1.5, 2.5, 3.5]
                            else:
                                increments = [i + 0.5 for i in range(0, 60, 5)]
                            suggested_line = min(increments, key=lambda x: abs(x - proj))
                            line_source    = "📐 Model"

                        rec = "OVER" if proj > suggested_line else "UNDER"
                        confidence = abs(proj - suggested_line)

                        # Implied odds: hit-rate of player vs this line weighted
                        p_vals = p_df[col].values
                        hit_rate = float((p_vals > suggested_line).mean()) if len(p_vals) else 0.5
                        implied = hit_rate if rec == "OVER" else 1 - hit_rate
                        implied = max(0.222, min(0.778, implied))
                        raw_odds = -round((implied / (1 - implied)) * 100) if implied >= 0.5 \
                                   else round(((1 - implied) / implied) * 100)
                        # clamp to ±350
                        american = max(-350, raw_odds) if raw_odds < 0 else min(350, raw_odds)
                        odds_str = f"{american:+d}"

                        ou = game["odds"].get("over_under")
                        prop_rows.append({
                            "Game":         f"{away_disp} @ {home_disp}",
                            "Date":         game["date"],
                            "Game O/U":     f"{ou}" if ou else "—",
                            "Offense":      offense_team,
                            "Defense":      defense_team,
                            "Player":       best["player_name"] + (" 🆕" if is_new_signing else ""),
                            "Stat":         col_label,
                            "Player Avg":   round(player_avg_display, 1),
                            "Last 3 Avg":   round(last3, 1),
                            "Def Allows":   round(def_avg, 1),
                            "Def Rank":     f"#{def_rank}",
                            "Matchup":      matchup_grade,
                            "Suggested Line": suggested_line,
                            "Line Source":  line_source,
                            "Pick":         rec,
                            "Odds":         odds_str,
                            "_conf":        confidence,
                        })

                if prop_rows:
                    prop_df = pd.DataFrame(prop_rows).sort_values("_conf", ascending=False)
                    prop_df = prop_df.drop(columns=["_conf"])

                    # ── Top 5 best bets banner ────────────────────────────────
                    st.subheader("Top Prop Suggestions This Week")
                    top5 = prop_df[prop_df["Matchup"] == "🟢 Soft"].head(5)
                    if top5.empty:
                        top5 = prop_df.head(5)

                    for _, r in top5.iterrows():
                        pick_color = "#2DC653" if r["Pick"] == "OVER" else "#D62828"
                        ou_str = f'&nbsp;&nbsp;·&nbsp;&nbsp;Game O/U: <b>{r["Game O/U"]}</b>' if r["Game O/U"] != "—" else ""
                        st.markdown(
                            f'<div style="border-left:5px solid {pick_color};'
                            f'padding:10px 14px;background:#f7f8fa;border-radius:6px;'
                            f'margin-bottom:8px;">'
                            f'<b>{r["Player"]}</b> ({r["Offense"]}) &nbsp;|&nbsp; '
                            f'<b>{col_label} {r["Pick"]} {r["Suggested Line"]}</b> '
                            f'<span style="color:#3b82d4;font-weight:600;">({r["Odds"]})</span>'
                            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                            f'vs {r["Defense"]} {r["Matchup"]} (allows {r["Def Allows"]:.1f}/gm)'
                            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                            f'Player avg: {r["Player Avg"]:.1f} &nbsp;|&nbsp; Last 3: {r["Last 3 Avg"]:.1f}'
                            f'{ou_str}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Auto-Parlay from top suggestions ──────────────────────
                    st.divider()
                    st.subheader("🎰 Auto-Generated Parlay")
                    st.caption("Picks the best legs from soft matchups. Use **Mixed** mode to combine passing, rushing, receiving and more.")

                    ap_c1, ap_c2 = st.columns([1, 3])
                    with ap_c1:
                        ap_mode = st.radio(
                            "Stat mode",
                            ["Single stat", "Mixed stats"],
                            horizontal=True,
                            key="ap_mode",
                            help="Single = current stat only. Mixed = best leg from each selected stat type.",
                        )
                        ap_stat_options = list(CAT_MAP.keys())
                        if ap_mode == "Mixed stats":
                            ap_stats = st.multiselect(
                                "Stats to include",
                                ap_stat_options,
                                default=["pass yards", "rush yards", "rec yards"],
                                format_func=str.title,
                                key="ap_stats",
                            )
                        else:
                            ap_stats = [mf_stat]

                        ap_legs     = st.slider("Max legs", 2, 8, 3, key="ap_legs")
                        ap_stake    = st.number_input("Stake ($)", min_value=1.0, value=10.0,
                                                      step=5.0, format="%.2f", key="ap_stake")
                        ap_weighted = st.toggle("Season weighting", value=True, key="ap_weighted")
                        ap_build    = st.button("⚡ Build Auto-Parlay", type="primary",
                                                use_container_width=True, key="ap_build")
                        ap_send     = st.button("➕ Send to Parlay Builder", use_container_width=True,
                                                key="ap_send",
                                                help="Loads these legs into the Parlay Builder tab")

                    with ap_c2:
                        if not ap_stats:
                            st.info("Select at least one stat to include.")
                        else:
                            # ── Build candidate pool across all selected stats ──
                            # For each stat, run the same defense-rank logic and
                            # collect the single best unique-player soft matchup.
                            candidate_legs = []
                            seen_players   = set()

                            for ap_stat in ap_stats:
                                ap_col = CAT_MAP[ap_stat.lower()][0]

                                # Build defense ranks for this stat
                                if mf_season == "Both":
                                    ap_def_sample = nfl_def.copy()
                                else:
                                    ap_def_sample = nfl_def[nfl_def["season"] == int(mf_season)]

                                ap_def_agg = (
                                    ap_def_sample.groupby("opponent")
                                    .agg(avg_allowed=(ap_col, "mean"), games=(ap_col, "count"))
                                    .reset_index()
                                )
                                ap_def_agg = ap_def_agg[ap_def_agg["games"] >= mf_min_games]
                                if ap_def_agg.empty:
                                    continue
                                ap_league_avg = ap_def_agg["avg_allowed"].mean()
                                ap_def_avgs   = dict(zip(ap_def_agg["opponent"], ap_def_agg["avg_allowed"]))

                                # Find best player for this stat across all this week's games
                                for game in games:
                                    for offense_team, defense_team in [
                                        (_norm_team(game["away"]), _norm_team(game["home"])),
                                        (_norm_team(game["home"]), _norm_team(game["away"])),
                                    ]:
                                        ap_def_avg = ap_def_avgs.get(defense_team, ap_league_avg)
                                        # Only soft matchups in mixed mode
                                        if ap_mode == "Mixed stats" and ap_def_avg <= ap_league_avg * 1.05:
                                            continue

                                        team_p = nfl_def[
                                            (nfl_def["team"] == offense_team) &
                                            (nfl_def["season"] == 2025) &
                                            (nfl_def[ap_col] > 0)
                                        ]
                                        if team_p.empty:
                                            team_p = nfl_def[
                                                (nfl_def["team"] == offense_team) &
                                                (nfl_def["season"] == 2024) &
                                                (nfl_def[ap_col] > 0)
                                            ]
                                        if team_p.empty:
                                            continue

                                        p_agg2 = (
                                            team_p.groupby("player_name")[ap_col]
                                            .agg(["mean", "count"])
                                            .reset_index()
                                        )
                                        p_agg2 = p_agg2[p_agg2["count"] >= mf_min_player]
                                        if p_agg2.empty:
                                            continue

                                        # Apply depth chart filter for auto-parlay too
                                        ap_dc_names = depth_chart_players(offense_team, ap_col, max_rank=3)
                                        if ap_dc_names:
                                            ap_dc_filtered = p_agg2[p_agg2["player_name"].isin(ap_dc_names)]
                                            if not ap_dc_filtered.empty:
                                                ap_dc_filtered = ap_dc_filtered.copy()
                                                ap_dc_filtered["dc_rank"] = ap_dc_filtered["player_name"].apply(
                                                    lambda n: ap_dc_names.index(n) if n in ap_dc_names else 99
                                                )
                                                p_agg2 = ap_dc_filtered.sort_values("dc_rank")

                                        best2 = p_agg2.iloc[0]
                                        if best2["player_name"] in seen_players:
                                            continue

                                        # Calculate suggested line for this stat
                                        matchup_f = ap_def_avg / ap_league_avg if ap_league_avg > 0 else 1.0
                                        proj2 = float(best2["mean"]) * matchup_f
                                        if ap_col == "passing_yards":
                                            incs = [i + 0.5 for i in range(50, 500, 25)]
                                        elif ap_col in ("rush_yards", "receiving_yards"):
                                            incs = [i + 0.5 for i in range(0, 250, 10)]
                                        elif ap_col == "receptions":
                                            incs = [i + 0.5 for i in range(0, 20, 1)]
                                        elif ap_col == "passing_tds":
                                            incs = [0.5, 1.5, 2.5, 3.5]
                                        else:
                                            incs = [i + 0.5 for i in range(0, 60, 5)]
                                        sug_line = min(incs, key=lambda x: abs(x - proj2))

                                        leg = score_leg(
                                            nfl_df,
                                            best2["player_name"],
                                            ap_stat,
                                            sug_line,
                                            ap_weighted,
                                        )
                                        if leg:
                                            leg["defense"]  = defense_team
                                            leg["def_avg"]  = round(ap_def_avg, 1)
                                            leg["matchup"]  = "🟢 Soft" if ap_def_avg > ap_league_avg * 1.10 else "🟡 Average"
                                            candidate_legs.append(leg)
                                            seen_players.add(best2["player_name"])
                                            break  # one best leg per stat per pass
                                    else:
                                        continue
                                    break

                            # Sort by implied prob descending, take top ap_legs
                            candidate_legs.sort(key=lambda x: x["implied_prob"], reverse=True)
                            auto_legs = candidate_legs[:ap_legs]

                            if len(auto_legs) < 2:
                                st.info("Not enough soft matchups found. Try **Single stat** mode, lower the leg count, or switch to **Both** seasons in the Defensive sample filter.")
                            else:
                                # Combined probability & payout
                                combined_prob = 1.0
                                for lg in auto_legs:
                                    combined_prob *= lg["implied_prob"]

                                combined_american = prob_to_american(combined_prob)
                                payout  = parlay_payout([lg["american_odds"] for lg in auto_legs], ap_stake)
                                profit  = round(payout - ap_stake, 2)
                                conf_label_ap, conf_color_ap = confidence_label(combined_prob)

                                # Confidence banner
                                st.markdown(
                                    f'<div style="background:{conf_color_ap};color:#fff;'
                                    f'padding:10px 18px;border-radius:8px;font-size:18px;'
                                    f'font-weight:700;text-align:center;margin-bottom:12px;">'
                                    f'Auto-Parlay Confidence: {conf_label_ap} &nbsp;·&nbsp; '
                                    f'{combined_prob*100:.1f}% est. probability'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                                # Summary metrics
                                pm1, pm2, pm3, pm4 = st.columns(4)
                                pm1.metric("Legs",             len(auto_legs))
                                pm2.metric("Combined Odds",    f"+{combined_american}" if combined_american > 0 else str(combined_american))
                                pm3.metric("Potential Payout", f"${payout:,.2f}")
                                pm4.metric("Potential Profit", f"${profit:,.2f}")

                                # Leg detail table
                                leg_rows = []
                                for lg in auto_legs:
                                    leg_rows.append({
                                        "Player":    lg["player"],
                                        "Stat":      lg["category"].title(),
                                        "Line":      lg["line"],
                                        "Pick":      lg["recommendation"],
                                        "vs Defense": lg.get("defense", "—"),
                                        "Def Allows": lg.get("def_avg", "—"),
                                        "Matchup":   lg.get("matchup", "—"),
                                        "Wtd Avg":   lg["w_avg"],
                                        "Hit Rate":  f"{lg['hit_rate_pct']}%",
                                        "Leg Odds":  f"+{lg['american_odds']}" if lg["american_odds"] > 0 else str(lg["american_odds"]),
                                        "Leg Prob":  f"{lg['implied_prob']*100:.1f}%",
                                    })
                                st.dataframe(pd.DataFrame(leg_rows),
                                             use_container_width=True, hide_index=True)

                                # Send to Parlay Builder
                                if ap_send:
                                    existing = st.session_state.get("parlay_legs", [])
                                    added = 0
                                    for lg in auto_legs:
                                        dup = any(
                                            e["player"] == lg["player"] and
                                            e["category"] == lg["category"] and
                                            e["line"] == lg["line"]
                                            for e in existing
                                        )
                                        if not dup and len(existing) < 8:
                                            existing.append(lg)
                                            added += 1
                                    st.session_state["parlay_legs"] = existing
                                    if added:
                                        st.success(f"✅ {added} leg{'s' if added > 1 else ''} added to Parlay Builder — switch to the 🎰 Parlay Builder tab.")
                                    else:
                                        st.info("All legs already in Parlay Builder (or builder is full at 8 legs).")

                    # ── Full table ────────────────────────────────────────────
                    st.divider()
                    st.subheader(f"All {col_label} Props — {label}")
                    st.dataframe(prop_df, use_container_width=True, hide_index=True)

                    # ── Defense rankings chart ────────────────────────────────
                    st.divider()
                    st.subheader(f"Full Defense Rankings — {col_label}")
                    show_top = st.slider("Show top N defenses", 5, 32, 16, key="mf_chart_n")
                    chart_df8 = def_agg.head(show_top)
                    fig8, ax8 = plt.subplots(figsize=(9, max(3, len(chart_df8) * 0.45)))
                    bar_colors8 = [
                        C_OVER  if v > league_avg * 1.10 else
                        (C_LINE if v < league_avg * 0.90 else "#f59e0b")
                        for v in chart_df8["avg_allowed"]
                    ]
                    bars8 = ax8.barh(
                        chart_df8["opponent"][::-1], chart_df8["avg_allowed"][::-1],
                        color=bar_colors8[::-1], alpha=0.85, edgecolor="white", linewidth=0.4
                    )
                    ax8.axvline(league_avg, color=C_AVG, linewidth=1.5, linestyle="--",
                                label=f"League avg: {league_avg:.1f}")
                    for bar, val in zip(bars8, chart_df8["avg_allowed"][::-1]):
                        ax8.text(bar.get_width() + chart_df8["avg_allowed"].max() * 0.01,
                                 bar.get_y() + bar.get_height() / 2,
                                 f"{val:.1f}", va="center", fontsize=8)
                    ax8.set_title(
                        f"Avg {col_label} Allowed per Game — Softest to Toughest",
                        fontsize=11, fontweight="bold"
                    )
                    ax8.set_xlabel(f"Avg {col_label} Allowed", fontsize=9)
                    ax8.legend(fontsize=8)
                    ax8.spines["top"].set_visible(False)
                    ax8.spines["right"].set_visible(False)
                    ax8.grid(axis="x", linestyle="--", alpha=0.35)
                    plt.tight_layout()
                    st.pyplot(fig8, use_container_width=True)
                    plt.close(fig8)


# ══════════════════════════════════════════════════════════════════════════════
# INJURY REPORT (tab9 inside main_teams)
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        @st.cache_data(ttl=1800, show_spinner=False)  # refresh every 30 min
        def fetch_injuries():
            """Fetch current NFL injury report from ESPN API."""
            url = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries")
            data = _get_json(url)
            if not data:
                return pd.DataFrame()
            rows = []
            for team_entry in data.get("injuries", []):
                team_abbr = team_entry.get("team", {}).get("abbreviation", "UNK")
                team_name = team_entry.get("team", {}).get("displayName", "Unknown")
                for inj in team_entry.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    rows.append({
                        "team":     team_abbr,
                        "team_name": team_name,
                        "player":   athlete.get("displayName", "Unknown"),
                        "position": athlete.get("position", {}).get("abbreviation", ""),
                        "status":   inj.get("status", ""),
                        "detail":   inj.get("details", {}).get("detail", ""),
                        "side":     inj.get("details", {}).get("side", ""),
                        "return_date": inj.get("details", {}).get("returnDate", ""),
                        "fantasy_status": inj.get("fantasyStatus", {}).get("description", ""),
                    })
            return pd.DataFrame(rows)

        st.subheader("🚑 NFL Injury Report")
        st.caption("Live data from ESPN · refreshes every 30 minutes")

        with st.spinner("Fetching injury report…"):
            inj_df = fetch_injuries()

        if inj_df.empty:
            st.warning("No injury data returned from ESPN. Try again in a moment.")
        else:
            # Filters
            ir_c1, ir_c2, ir_c3 = st.columns(3)
            inj_teams = ["All"] + sorted(inj_df["team"].unique().tolist())
            inj_pos   = ["All"] + sorted(inj_df["position"].dropna().unique().tolist())
            inj_stat  = ["All"] + sorted(inj_df["status"].dropna().unique().tolist())

            sel_team = ir_c1.selectbox("Filter by team",     inj_teams, key="ir_team")
            sel_pos  = ir_c2.selectbox("Filter by position", inj_pos,   key="ir_pos")
            sel_stat = ir_c3.selectbox("Filter by status",   inj_stat,  key="ir_stat")

            view_inj = inj_df.copy()
            if sel_team != "All": view_inj = view_inj[view_inj["team"] == sel_team]
            if sel_pos  != "All": view_inj = view_inj[view_inj["position"] == sel_pos]
            if sel_stat != "All": view_inj = view_inj[view_inj["status"] == sel_stat]

            # Colour-code by status
            def status_icon(s):
                s = str(s).lower()
                if "out" in s:           return "🔴 Out"
                if "doubtful" in s:      return "🟠 Doubtful"
                if "questionable" in s:  return "🟡 Questionable"
                if "probable" in s:      return "🟢 Probable"
                return s.title()

            view_inj = view_inj.copy()
            view_inj["status"] = view_inj["status"].apply(status_icon)

            st.metric("Injured players shown", len(view_inj))
            st.dataframe(
                view_inj[["team","player","position","status","detail",
                           "side","fantasy_status","return_date"]]
                .rename(columns={
                    "team": "Team", "player": "Player", "position": "Pos",
                    "status": "Status", "detail": "Injury",
                    "side": "Side", "fantasy_status": "Fantasy Status",
                    "return_date": "Est. Return",
                })
                .sort_values(["Team","Player"]),
                use_container_width=True, hide_index=True,
            )

            # Quick lookup — cross reference with your player data
            st.divider()
            st.subheader("Player Injury Lookup")
            ir_search = st.text_input("Search injured player", key="ir_search",
                                       placeholder="e.g. Justin Jefferson")
            if ir_search:
                found = inj_df[inj_df["player"].str.contains(ir_search,
                               case=False, na=False)]
                if not found.empty:
                    for _, row in found.iterrows():
                        icon = status_icon(row["status"])
                        st.warning(
                            f"**{row['player']}** ({row['position']} · {row['team']})  "
                            f"— {icon}  |  {row['detail']} {row['side']}  "
                            f"|  Fantasy: {row['fantasy_status']}"
                        )
                else:
                    st.success(f"No injury listing found for '{ir_search}' — likely active.")


# ══════════════════════════════════════════════════════════════════════════════
# HOME / AWAY SPLITS (tab10 inside main_players)
# ══════════════════════════════════════════════════════════════════════════════
with tab10:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        @st.cache_data(show_spinner=False)
        def build_home_away(nfl):
            """Add is_home column: True if player's team is the home team in game_id."""
            df = nfl.copy()
            def _is_home(row):
                parts = str(row["game_id"]).split("_")
                if len(parts) < 4:
                    return None
                home_team = parts[3]
                return row["team"] == home_team
            df["is_home"] = df.apply(_is_home, axis=1)
            return df

        nfl_ha = build_home_away(nfl_df)
        all_players_ha = sorted(nfl_df["player_name"].unique())

        st.subheader("🏠 Home / Away Splits")
        ha_c1, ha_c2 = st.columns([1, 3])

        with ha_c1:
            ha_player = st.selectbox(
                "Player", all_players_ha,
                index=all_players_ha.index("Drake Maye")
                if "Drake Maye" in all_players_ha else 0,
                key="ha_player",
            )
            ha_cat    = st.selectbox("Stat", list(CAT_MAP.keys()),
                                      format_func=str.title, key="ha_cat")
            ha_season = st.radio("Season", [2025, 2024, "Both"], key="ha_season")
            ha_line   = st.number_input("Prop line (optional)",
                                         min_value=0.0, value=0.0, step=0.5,
                                         format="%.1f", key="ha_line")

        with ha_c2:
            col, col_label = CAT_MAP[ha_cat.lower()]
            pdf_ha = nfl_ha[nfl_ha["player_name"].str.contains(
                ha_player, case=False, na=False)].copy()

            if pdf_ha.empty:
                st.error("Player not found.")
            else:
                full_ha = pdf_ha["player_name"].iloc[0]
                if ha_season != "Both":
                    pdf_ha = pdf_ha[pdf_ha["season"] == int(ha_season)]

                home_games = pdf_ha[pdf_ha["is_home"] == True]
                away_games = pdf_ha[pdf_ha["is_home"] == False]

                # Summary metrics
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Home Avg",    f"{home_games[col].mean():.1f}" if not home_games.empty else "—")
                m2.metric("Away Avg",    f"{away_games[col].mean():.1f}" if not away_games.empty else "—")
                m3.metric("Home Games",  len(home_games))
                m4.metric("Away Games",  len(away_games))
                diff = (home_games[col].mean() - away_games[col].mean()) if (not home_games.empty and not away_games.empty) else 0
                m5.metric("Home vs Away", f"{diff:+.2f}")
                if ha_line > 0:
                    hr_home = (home_games[col] > ha_line).mean() * 100 if not home_games.empty else 0
                    hr_away = (away_games[col] > ha_line).mean() * 100 if not away_games.empty else 0
                    m6.metric("Hit Rate H/A", f"{hr_home:.0f}% / {hr_away:.0f}%")

                # Side-by-side bar charts
                fig10, axes10 = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
                for ax, gdf, label, color in [
                    (axes10[0], home_games, "Home Games", C_2025),
                    (axes10[1], away_games, "Away Games", C_2024),
                ]:
                    if gdf.empty:
                        ax.set_title(f"{label} — No data", fontsize=11)
                        continue
                    gdf = gdf.reset_index(drop=True)
                    gdf["gn"] = range(1, len(gdf) + 1)
                    vals = gdf[col].values
                    bar_c = [C_OVER if (ha_line > 0 and v > ha_line) else
                             (C_LINE if ha_line > 0 else color)
                             for v in vals]
                    bars10 = ax.bar(gdf["gn"], vals, color=bar_c, alpha=0.85,
                                    edgecolor="white", linewidth=0.5)
                    avg10 = vals.mean()
                    ax.axhline(avg10, color=C_AVG, linewidth=1.6, linestyle="--",
                               label=f"Avg: {avg10:.1f}")
                    if ha_line > 0:
                        ax.axhline(ha_line, color=C_LINE, linewidth=1.6,
                                   linestyle="-", label=f"Line: {ha_line}")
                    for bar, val in zip(bars10, vals):
                        if val > 0:
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + max(vals) * 0.02,
                                    f"{int(val)}" if val == int(val) else f"{val:.2f}",
                                    ha="center", va="bottom", fontsize=7.5)
                    ax.set_title(label, fontsize=11, fontweight="bold")
                    ax.set_xlabel("Game #", fontsize=9)
                    ax.set_ylabel(col_label, fontsize=9)
                    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 10)
                    ax.legend(fontsize=8)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(axis="y", linestyle="--", alpha=0.35)

                fig10.suptitle(f"{full_ha}  —  {col_label}  |  Home vs Away",
                               fontsize=13, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig10, use_container_width=True)
                plt.close(fig10)

                # Game log table
                st.subheader("Game Log with Home/Away Flag")
                log_ha = pdf_ha[["season","game_id","is_home","team",col,"fantasy_points"]].copy()
                log_ha["is_home"] = log_ha["is_home"].map({True: "🏠 Home", False: "✈️ Away"})
                log_ha = log_ha.rename(columns={
                    "season":"Season","game_id":"Game","is_home":"Location",
                    "team":"Team", col: col_label,"fantasy_points":"Fantasy Pts"
                })
                st.dataframe(log_ha, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# START / SIT ADVISOR (tab11 inside main_players)
# ══════════════════════════════════════════════════════════════════════════════
with tab11:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        @st.cache_data(show_spinner=False)
        def build_opp_defense(nfl):
            df = nfl.copy()
            def get_opp(row):
                parts = str(row["game_id"]).split("_")
                if len(parts) < 4: return "UNK"
                away, home = parts[2], parts[3]
                return home if row["team"] == away else away
            df["opponent"] = df.apply(get_opp, axis=1)
            return df

        nfl_ss = build_opp_defense(nfl_df)
        all_players_ss = sorted(nfl_df["player_name"].unique())
        all_teams_ss   = sorted(nfl_df["team"].unique())

        st.subheader("🏆 Start / Sit Advisor")
        st.caption("Compare two players and get a fantasy start recommendation based on stats + matchup.")

        ss_c1, ss_c2 = st.columns(2)
        with ss_c1:
            st.markdown("#### Player A")
            ss_p1     = st.selectbox("Player A", all_players_ss,
                                      index=all_players_ss.index("Drake Maye")
                                      if "Drake Maye" in all_players_ss else 0,
                                      key="ss_p1")
            ss_opp1   = st.selectbox("Player A opponent this week",
                                      all_teams_ss, key="ss_opp1")

        with ss_c2:
            st.markdown("#### Player B")
            ss_p2     = st.selectbox("Player B", all_players_ss,
                                      index=min(1, len(all_players_ss)-1),
                                      key="ss_p2")
            ss_opp2   = st.selectbox("Player B opponent this week",
                                      all_teams_ss, key="ss_opp2")

        ss_cat = st.selectbox("Scoring stat", list(CAT_MAP.keys()),
                               format_func=str.title, key="ss_cat")
        ss_go  = st.button("🏆 Get Recommendation", type="primary",
                            use_container_width=True, key="ss_go")

        if ss_go:
            col, col_label = CAT_MAP[ss_cat.lower()]

            def player_score(player_name, opp_team, col):
                """Return a score dict for one player vs one opponent."""
                pdf = find_player(nfl_ss, player_name)
                if pdf.empty:
                    return None
                full   = pdf["player_name"].iloc[0]
                p25    = pdf[pdf["season"] == 2025]
                p24    = pdf[pdf["season"] == 2024]

                vals   = pdf[col].values
                wts    = pdf["weight"].values
                w_avg  = np.average(vals, weights=wts)
                last3  = p25[col].tail(3).mean() if not p25.empty else pdf[col].tail(3).mean()
                season_avg = p25[col].mean() if not p25.empty else pdf[col].mean()

                # Opponent defensive strength for this stat
                opp_data = nfl_ss[nfl_ss["opponent"] == opp_team]
                opp_avg_allowed = opp_data[col].mean() if not opp_data.empty else w_avg
                league_avg = nfl_ss.groupby("opponent")[col].mean().mean()
                matchup_factor = opp_avg_allowed / league_avg if league_avg > 0 else 1.0

                # Score = weighted avg * matchup factor, boosted by recent form
                form_boost = (last3 / season_avg) if season_avg > 0 else 1.0
                form_boost = max(0.5, min(2.0, form_boost))  # cap between 0.5x and 2x
                score = w_avg * matchup_factor * form_boost

                return {
                    "name":           full,
                    "opponent":       opp_team,
                    "w_avg":          round(float(w_avg), 1),
                    "last3":          round(float(last3), 1),
                    "season_avg":     round(float(season_avg), 1),
                    "opp_allowed":    round(float(opp_avg_allowed), 1),
                    "league_avg":     round(float(league_avg), 1),
                    "matchup_factor": round(float(matchup_factor), 2),
                    "form_boost":     round(float(form_boost), 2),
                    "final_score":    round(float(score), 2),
                    "matchup_grade":  "🟢 Favorable" if matchup_factor > 1.08
                                      else ("🔴 Tough" if matchup_factor < 0.92
                                            else "🟡 Neutral"),
                }

            s1 = player_score(ss_p1, ss_opp1, col)
            s2 = player_score(ss_p2, ss_opp2, col)

            if s1 is None or s2 is None:
                st.error("One or both players not found.")
            else:
                # Recommendation banner
                winner = s1 if s1["final_score"] >= s2["final_score"] else s2
                loser  = s2 if winner == s1 else s1
                margin = abs(s1["final_score"] - s2["final_score"])
                confidence = "Strong" if margin > s1["w_avg"] * 0.15 else "Lean"
                banner_color = "#2DC653"

                st.markdown(
                    f'<div style="background:{banner_color};color:#fff;'
                    f'padding:14px 20px;border-radius:8px;font-size:22px;'
                    f'font-weight:700;text-align:center;margin-bottom:16px;">'
                    f'START: {winner["name"]}  ({confidence} recommendation)'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Side-by-side comparison
                d1, d2 = st.columns(2)
                for col_disp, s, is_winner in [(d1, s1, s1==winner), (d2, s2, s2==winner)]:
                    border = "#2DC653" if is_winner else "#e5e7eb"
                    col_disp.markdown(
                        f'<div style="border:3px solid {border};border-radius:8px;'
                        f'padding:14px;margin-bottom:8px;">'
                        f'<b style="font-size:16px">{s["name"]}</b><br>'
                        f'vs <b>{s["opponent"]}</b> &nbsp;·&nbsp; {s["matchup_grade"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    col_disp.metric("Weighted Avg",     f"{s['w_avg']:.1f}")
                    col_disp.metric("Last 3 Avg",       f"{s['last3']:.1f}")
                    col_disp.metric("Season Avg",       f"{s['season_avg']:.1f}")
                    col_disp.metric(f"{s['opponent']} Allows", f"{s['opp_allowed']:.1f}",
                                     delta=f"{s['opp_allowed']-s['league_avg']:+.1f} vs league",
                                     delta_color="inverse")
                    col_disp.metric("Matchup Factor",   f"{s['matchup_factor']:.2f}x")
                    col_disp.metric("Form Boost",       f"{s['form_boost']:.2f}x")
                    col_disp.metric("Final Score",      f"{s['final_score']:.1f}",
                                     delta="START" if is_winner else "SIT",
                                     delta_color="normal" if is_winner else "inverse")

                # Comparison bar chart
                fig11, ax11 = plt.subplots(figsize=(7, 4))
                categories  = ["Weighted Avg", "Last 3 Avg", "Opp Allowed", "Final Score"]
                vals_p1     = [s1["w_avg"], s1["last3"], s1["opp_allowed"], s1["final_score"]]
                vals_p2     = [s2["w_avg"], s2["last3"], s2["opp_allowed"], s2["final_score"]]
                x           = np.arange(len(categories))
                width       = 0.35
                ax11.bar(x - width/2, vals_p1, width, label=s1["name"],
                          color=C_2025, alpha=0.85, edgecolor="white")
                ax11.bar(x + width/2, vals_p2, width, label=s2["name"],
                          color=C_2024, alpha=0.85, edgecolor="white")
                ax11.set_xticks(x)
                ax11.set_xticklabels(categories, fontsize=9)
                ax11.set_ylabel(col_label, fontsize=9)
                ax11.set_title(f"Start/Sit Comparison — {col_label}", fontsize=11, fontweight="bold")
                ax11.legend(fontsize=9)
                ax11.spines["top"].set_visible(False)
                ax11.spines["right"].set_visible(False)
                ax11.grid(axis="y", linestyle="--", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig11, use_container_width=True)
                plt.close(fig11)

                st.caption(
                    "Final Score = Weighted Avg × Matchup Factor × Form Boost. "
                    "Matchup Factor > 1.0 = defense allows more than league average. "
                    "Form Boost = last-3 avg / season avg (capped 0.5×–2.0×)."
                )
        else:
            st.info("👈 Select two players, their opponents this week, and click **Get Recommendation**.")


# ══════════════════════════════════════════════════════════════════════════════
# VEGAS LINE IMPORTER (tab_vegas inside main_bet)
# Auto-fetches live NFL player props from The Odds API (free tier).
# Falls back to manual paste if no API key is set.
# ══════════════════════════════════════════════════════════════════════════════

# ── Odds API helpers ──────────────────────────────────────────────────────────
# Map Odds API market keys → our internal CAT_MAP keys
_ODDS_MARKET_MAP = {
    "player_pass_yds":     "pass yards",
    "player_rush_yds":     "rush yards",
    "player_reception_yds":"rec yards",
    "player_receptions":   "receptions",
    "player_pass_tds":     "pass tds",
}
# Markets to request (comma-joined for the API call)
_ODDS_MARKETS = ",".join(_ODDS_MARKET_MAP.keys())

@st.cache_data(ttl=900, show_spinner=False)   # cache 15 min — free tier has 500 req/month
def fetch_odds_api_props(api_key: str) -> list:
    """
    Pull live NFL player prop lines from The Odds API.
    Returns a list of dicts: {player, cat, line, bookmaker, opp}
    opp = the opposing team abbreviation derived from the game matchup.
    """
    # Step 1: get event IDs for upcoming NFL games
    events_url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
        f"?apiKey={api_key}&dateFormat=iso"
    )
    events_data = _get_json(events_url)
    if not events_data or not isinstance(events_data, list):
        return []

    # Step 2: for each event fetch player props
    rows = []
    for event in events_data[:16]:   # max 16 games/week; respect quota
        event_id   = event.get("id", "")
        home_team  = event.get("home_team", "")
        away_team  = event.get("away_team", "")

        props_url = (
            f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
            f"/events/{event_id}/odds"
            f"?apiKey={api_key}&regions=us&markets={_ODDS_MARKETS}"
            f"&oddsFormat=american&dateFormat=iso"
        )
        props_data = _get_json(props_url)
        if not props_data:
            continue

        bookmakers = props_data.get("bookmakers", [])
        # Prefer DraftKings, fall back to first available
        bm = next((b for b in bookmakers if b["key"] == "draftkings"), None)
        if bm is None and bookmakers:
            bm = bookmakers[0]
        if bm is None:
            continue

        bm_name = bm.get("title", "Book")
        for market in bm.get("markets", []):
            market_key = market.get("key", "")
            cat = _ODDS_MARKET_MAP.get(market_key)
            if cat is None:
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name", "").lower() != "over":
                    continue   # only grab the Over line (we store one line per player/stat)
                player_name = outcome.get("description", outcome.get("player", ""))
                point       = outcome.get("point")
                if not player_name or point is None:
                    continue
                # Determine opponent: if player is on home team, opp = away, and vice versa.
                # We don't know the player's team from this endpoint, so store both teams.
                rows.append({
                    "player_raw": player_name,
                    "cat":        cat,
                    "line":       float(point),
                    "opp":        None,          # filled below where we can
                    "home":       home_team,
                    "away":       away_team,
                    "bookmaker":  bm_name,
                })
        _time.sleep(0.2)   # be polite

    return rows


def _full_team_name_to_abbr(full: str) -> str:
    """Best-effort map of ESPN full team name → our 2-3 letter abbreviation."""
    _MAP = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
        "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE",
        "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    }
    return _MAP.get(full, full[:3].upper())


with tab_vegas:
    if not data_ok:
        st.info("Load data first using the **⚙️ Settings & Data** tab.")
    else:
        st.subheader("📈 Vegas Lines — Live Prop Odds")
        st.caption(
            "Pulls live NFL player prop lines from The Odds API (DraftKings / consensus) "
            "and scores each one against the historical model. "
            "Free tier: 500 requests/month · lines refresh every 15 min."
        )

        # ── Build opponent lookup once (shared) ───────────────────────────────
        @st.cache_data(show_spinner=False)
        def _build_opp_vl(nfl):
            df = nfl.copy()
            def _opp(row):
                parts = str(row["game_id"]).split("_")
                if len(parts) < 4: return "UNK"
                away, home = parts[2], parts[3]
                return home if row["team"] == away else away
            df["opponent"] = df.apply(_opp, axis=1)
            return df
        nfl_vl_opp = _build_opp_vl(nfl_df)

        # ── Shared scoring helper ─────────────────────────────────────────────
        def _score_parsed_rows(parsed_rows, vl_weighted, vl_window, vl_opp_mode, vl_min_edge):
            """Score a list of {player_raw, cat, line, opp} dicts. Returns result_rows list."""
            result_rows = []
            for pr in parsed_rows:
                res = prop_analysis(
                    nfl_df,
                    pr["player_raw"], pr["cat"],
                    pr["line"], vl_weighted, vl_window,
                )
                if res is None:
                    result_rows.append({
                        "Player":     pr["player_raw"],
                        "Stat":       pr["cat"].title(),
                        "Book Line":  pr["line"],
                        "Bookmaker":  pr.get("bookmaker", "—"),
                        "Model Avg":  "—",
                        "Edge":       "—",
                        "Hit Rate":   "—",
                        "Std Dev":    "—",
                        "Pick":       "NOT FOUND",
                        "Confidence": "—",
                        "_edge_val":  0.0,
                        "_found":     False,
                    })
                    continue

                col_name  = CAT_MAP[pr["cat"]][0]
                model_avg = res["w_avg"]
                edge      = model_avg - pr["line"]
                hr_val    = res["w_hit"]
                std_dev   = res["std_dev"]
                pick      = "OVER" if edge > 0 else "UNDER"

                matchup_grade = ""
                if vl_opp_mode and pr.get("opp"):
                    opp = str(pr["opp"]).strip().upper()
                    vs_opp   = nfl_vl_opp[nfl_vl_opp["opponent"] == opp]
                    lg_avg_d = nfl_vl_opp.groupby("opponent")[col_name].mean().mean()
                    opp_avg  = vs_opp[col_name].mean() if not vs_opp.empty else lg_avg_d
                    factor   = opp_avg / lg_avg_d if lg_avg_d > 0 else 1.0
                    edge     = (model_avg * factor) - pr["line"]
                    pick     = "OVER" if edge > 0 else "UNDER"
                    if factor > 1.10:   matchup_grade = "🟢 Soft"
                    elif factor < 0.90: matchup_grade = "🔴 Tough"
                    else:               matchup_grade = "🟡 Avg"

                conf_dist = abs(hr_val - 50.0)
                if conf_dist >= 20:   conf = "★★★ High"
                elif conf_dist >= 10: conf = "★★  Med"
                else:                 conf = "★    Low"

                row = {
                    "Player":     res["full_name"],
                    "Stat":       pr["cat"].title(),
                    "Book Line":  pr["line"],
                    "Bookmaker":  pr.get("bookmaker", "—"),
                    "Model Avg":  round(model_avg, 1),
                    "Edge":       round(edge, 1),
                    "Hit Rate":   f"{hr_val:.1f}%",
                    "Std Dev":    round(std_dev, 1),
                    "Pick":       pick,
                    "Confidence": conf,
                    "_edge_val":  abs(edge),
                    "_found":     True,
                }
                if vl_opp_mode and pr.get("opp"):
                    row["Opponent"] = str(pr["opp"]).strip().upper()
                    row["Matchup"]  = matchup_grade
                result_rows.append(row)
            return result_rows

        def _render_results(result_rows, vl_min_edge, vl_opp_mode, vl_weighted):
            """Render KPIs, top-edge banners, full table, and send-to-parlay for a result set."""
            result_rows.sort(key=lambda r: r["_edge_val"], reverse=True)
            found = [r for r in result_rows if r["_found"]]
            not_found = [r for r in result_rows if not r["_found"]]

            if not found and not_found:
                st.warning(
                    f"{len(not_found)} player(s) not found in the dataset. "
                    "They may be rookies or have no game log data yet."
                )
                return

            # KPIs
            n_over   = sum(1 for r in found if r["Pick"] == "OVER")
            n_under  = sum(1 for r in found if r["Pick"] == "UNDER")
            n_high   = sum(1 for r in found if "High" in r["Confidence"])
            avg_edge = (sum(r["_edge_val"] for r in found) / len(found)) if found else 0

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Lines analysed",  len(found))
            k2.metric("OVER picks",      n_over)
            k3.metric("UNDER picks",     n_under)
            k4.metric("High confidence", n_high)
            k5.metric("Avg edge",        f"{avg_edge:.1f}")

            if not_found:
                with st.expander(f"⚠️ {len(not_found)} player(s) not found in dataset"):
                    st.write(", ".join(r["Player"] for r in not_found))

            # Top-edge banners
            top_edges = [r for r in found if r["_edge_val"] >= vl_min_edge]
            if top_edges:
                st.markdown(f"**{len(top_edges)} bet(s) with edge ≥ {vl_min_edge:.0f} units:**")
                for r in top_edges[:8]:
                    pick_color = "#2DC653" if r["Pick"] == "OVER" else "#D62828"
                    edge_str   = f"+{r['Edge']:.1f}" if r["Edge"] > 0 else f"{r['Edge']:.1f}"
                    opp_str    = f" &nbsp;·&nbsp; vs {r.get('Opponent','')} {r.get('Matchup','')}" if vl_opp_mode and r.get("Opponent") else ""
                    bk_str     = f" &nbsp;·&nbsp; {r['Bookmaker']}" if r.get("Bookmaker") and r["Bookmaker"] != "—" else ""
                    st.markdown(
                        f'<div style="border-left:5px solid {pick_color};padding:8px 14px;'
                        f'background:#f7f8fa;border-radius:6px;margin-bottom:6px;font-size:14px;">'
                        f'<b>{r["Player"]}</b> &nbsp;|&nbsp; '
                        f'{r["Stat"]} <span style="color:{pick_color};font-weight:700;">'
                        f'{r["Pick"]} {r["Book Line"]}</span>'
                        f'&nbsp;&nbsp;·&nbsp;&nbsp;Model avg: <b>{r["Model Avg"]}</b>'
                        f'&nbsp;&nbsp;·&nbsp;&nbsp;Edge: <b>{edge_str}</b>'
                        f'&nbsp;&nbsp;·&nbsp;&nbsp;Hit rate: {r["Hit Rate"]}'
                        f'&nbsp;&nbsp;·&nbsp;&nbsp;{r["Confidence"]}'
                        f'{opp_str}{bk_str}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"No lines with edge ≥ {vl_min_edge:.0f} units. Try lowering the threshold.")

            st.divider()
            st.subheader("Full Slate Results")

            disp_df = pd.DataFrame(result_rows).drop(columns=["_edge_val", "_found"])

            def _color_edge(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return ""
                if v >= vl_min_edge:  return "background-color:#d4f5dc"
                if v <= -vl_min_edge: return "background-color:#fde8e8"
                return ""

            def _color_pick(val):
                if val == "OVER":      return "color:#2DC653;font-weight:700"
                if val == "UNDER":     return "color:#D62828;font-weight:700"
                if val == "NOT FOUND": return "color:#888"
                return ""

            styled = disp_df.style.map(_color_edge, subset=["Edge"]).map(_color_pick, subset=["Pick"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Send to Parlay Builder
            st.divider()
            vl_send_n = st.slider("Send top N edges to Parlay Builder", 2, 8, 3, key="vl_send_n")
            if st.button("➕ Send to Parlay Builder", key="vl_send",
                         help="Loads the top-N edge legs into the Parlay Builder tab"):
                sendable = sorted([r for r in found if r["Pick"] in ("OVER","UNDER")],
                                  key=lambda r: r["_edge_val"], reverse=True)
                existing = st.session_state.get("parlay_legs", [])
                added = 0
                for r in sendable[:vl_send_n]:
                    leg = score_leg(nfl_df, r["Player"], r["Stat"].lower(),
                                    r["Book Line"], vl_weighted)
                    if leg is None:
                        continue
                    dup = any(
                        e["player"] == leg["player"] and
                        e["category"] == leg["category"] and
                        e["line"] == leg["line"]
                        for e in existing
                    )
                    if not dup and len(existing) < 8:
                        existing.append(leg)
                        added += 1
                st.session_state["parlay_legs"] = existing
                if added:
                    st.success(f"✅ {added} leg(s) added — switch to the 🎰 Parlay Builder tab.")
                else:
                    st.info("All legs already in Parlay Builder (or it's full at 8 legs).")

        # ═════════════════════════════════════════════════════════════════════
        # CONTROLS PANEL
        # ═════════════════════════════════════════════════════════════════════
        vl_c1, vl_c2 = st.columns([1, 3])

        with vl_c1:
            st.markdown("#### Settings")
            vl_weighted = st.toggle("Season weighting", value=True, key="vl_weighted")
            vl_window   = st.radio("Game window", ["Last 3", "Last 5", "Season"],
                                   index=2, horizontal=True, key="vl_window")
            vl_opp_mode = st.toggle(
                "Matchup adjustment",
                value=True, key="vl_opp_mode",
                help="Factor in opponent defensive strength.",
            )
            vl_min_edge = st.number_input(
                "Min edge to highlight",
                min_value=0.0, value=5.0, step=1.0, format="%.1f",
                key="vl_min_edge",
                help="Rows where |model avg – book line| ≥ this value are highlighted.",
            )

            st.divider()
            st.markdown("#### API Key")
            st.caption(
                "Get a free key at [the-odds-api.com](https://the-odds-api.com). "
                "Add it to Streamlit secrets as `ODDS_API_KEY`, or paste it below."
            )
            # Read from st.secrets first, allow manual override
            _secret_key = st.secrets.get("ODDS_API_KEY", "") if hasattr(st, "secrets") else ""
            vl_api_key  = st.text_input(
                "The Odds API key",
                value=_secret_key,
                type="password",
                key="vl_api_key",
                placeholder="Paste key here or set ODDS_API_KEY in secrets",
            )

            vl_fetch_btn = st.button("🔄 Fetch Live Lines", type="primary",
                                     use_container_width=True, key="vl_fetch")
            vl_manual_mode = st.toggle("Manual paste mode", value=False, key="vl_manual",
                                       help="Skip the API and paste lines yourself instead.")

        with vl_c2:
            # ── AUTO FETCH MODE ───────────────────────────────────────────────
            if not vl_manual_mode:
                if vl_fetch_btn or st.session_state.get("vl_auto_rows"):
                    if vl_fetch_btn:
                        if not vl_api_key.strip():
                            st.error(
                                "No API key set. Add `ODDS_API_KEY` to your Streamlit secrets "
                                "or paste a key in the field on the left."
                            )
                            st.stop()
                        with st.spinner("Fetching live prop lines from The Odds API…"):
                            raw_rows = fetch_odds_api_props(vl_api_key.strip())
                        if not raw_rows:
                            st.warning(
                                "No prop lines returned. The API key may be invalid, "
                                "quota may be exhausted, or there are no NFL games this week."
                            )
                            st.stop()
                        # Convert full team names → abbreviations for opp lookup
                        for r in raw_rows:
                            r["opp"] = None   # we don't know player team from API; skip opp adjust
                        st.session_state["vl_auto_rows"] = raw_rows
                        st.session_state["vl_auto_key"]  = vl_api_key.strip()

                    cached_rows = st.session_state.get("vl_auto_rows", [])
                    if not cached_rows:
                        st.info("👈 Enter your API key and click **Fetch Live Lines**.")
                    else:
                        # Filter controls
                        all_stats_vl = sorted({r["cat"] for r in cached_rows})
                        sel_stats_vl = st.multiselect(
                            "Filter stat categories",
                            options=all_stats_vl,
                            default=all_stats_vl,
                            format_func=str.title,
                            key="vl_stat_filter",
                        )
                        filtered_rows = [r for r in cached_rows if r["cat"] in sel_stats_vl]

                        st.caption(
                            f"**{len(filtered_rows)} prop lines** fetched from "
                            f"{cached_rows[0].get('bookmaker','book') if cached_rows else '—'} · "
                            "cached 15 min · toggle **Manual paste mode** to add custom lines"
                        )

                        if filtered_rows:
                            result_rows = _score_parsed_rows(
                                filtered_rows, vl_weighted, vl_window,
                                vl_opp_mode, vl_min_edge,
                            )
                            _render_results(result_rows, vl_min_edge, vl_opp_mode, vl_weighted)
                else:
                    st.info(
                        "👈 Enter your [The Odds API](https://the-odds-api.com) key and click "
                        "**Fetch Live Lines** to automatically pull this week's NFL prop lines.\n\n"
                        "Or enable **Manual paste mode** to enter lines yourself."
                    )

            # ── MANUAL PASTE MODE ─────────────────────────────────────────────
            else:
                with st.expander("📋 Input format", expanded=False):
                    st.markdown(
                        "One prop per row: `Player Name, stat, line` — optional 4th column for opponent.\n\n"
                        "**Stats:** `pass yards` · `rush yards` · `rec yards` · `receptions` · `pass tds` · `fantasy`"
                    )
                vl_paste = st.text_area(
                    "Paste prop lines",
                    height=200,
                    key="vl_paste",
                    placeholder=(
                        "Patrick Mahomes, pass yards, 287.5\n"
                        "Saquon Barkley, rush yards, 84.5, PHI\n"
                        "Ja'Marr Chase, rec yards, 74.5\n"
                        "Travis Kelce, receptions, 5.5"
                    ),
                )
                vl_go = st.button("⚡ Run Analysis", type="primary",
                                  use_container_width=True, key="vl_go")

                if vl_go and vl_paste.strip():
                    parse_errors = []
                    parsed_rows  = []
                    for raw_line in vl_paste.strip().splitlines():
                        raw_line = raw_line.strip()
                        if not raw_line or raw_line.startswith("#"):
                            continue
                        parts = [p.strip() for p in raw_line.split(",")]
                        if len(parts) < 3:
                            parse_errors.append(f"⚠️ Skipped: `{raw_line}`")
                            continue
                        cat_raw = parts[1].lower().strip()
                        try:
                            line_raw = float(parts[2])
                        except ValueError:
                            parse_errors.append(f"⚠️ Bad line value: `{raw_line}`")
                            continue
                        if cat_raw not in CAT_MAP:
                            cat_raw = next((k for k in CAT_MAP if k.startswith(cat_raw[:4])), None)
                        if cat_raw is None:
                            parse_errors.append(f"⚠️ Unknown category: `{raw_line}`")
                            continue
                        parsed_rows.append({
                            "player_raw": parts[0],
                            "cat":        cat_raw,
                            "line":       line_raw,
                            "opp":        parts[3] if len(parts) >= 4 else None,
                            "bookmaker":  "Manual",
                        })
                    for e in parse_errors:
                        st.warning(e)
                    if parsed_rows:
                        result_rows = _score_parsed_rows(
                            parsed_rows, vl_weighted, vl_window,
                            vl_opp_mode, vl_min_edge,
                        )
                        _render_results(result_rows, vl_min_edge, vl_opp_mode, vl_weighted)
                    else:
                        st.error("No valid rows. Check the format above.")
                elif vl_go:
                    st.warning("Paste some lines above first.")
                else:
                    st.info("👈 Paste lines above and click **Run Analysis**.")


# ══════════════════════════════════════════════════════════════════════════════
# BET TRACKER  (main_tracker — 4th main tab)
# ══════════════════════════════════════════════════════════════════════════════
import json as _json, os as _os

_BET_FILE = _os.path.join(_os.path.dirname(__file__), "bets.json")


def _load_bets() -> list:
    """Load bets from the JSON file; return empty list if missing or corrupt."""
    if not _os.path.exists(_BET_FILE):
        return []
    try:
        with open(_BET_FILE, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_bets(bets: list):
    """Persist bet list to disk."""
    with open(_BET_FILE, "w", encoding="utf-8") as f:
        _json.dump(bets, f, indent=2)


# Keep a single copy in session state so edits are instant without re-reading disk
if "bt_bets" not in st.session_state:
    st.session_state["bt_bets"] = _load_bets()


with main_tracker:
    st.title("📒 Bet Tracker")
    st.caption(
        "Log your prop bets, mark results, and track P&L / ROI over time. "
        "Data is saved to **bets.json** in the project folder."
    )

    import datetime as _dtbt

    # ── helper: rebuild DataFrame from session state ──────────────────────────
    def _bets_df() -> pd.DataFrame:
        bets = st.session_state["bt_bets"]
        if not bets:
            return pd.DataFrame(columns=[
                "id","date","player","stat","line","pick","stake",
                "odds","result","actual","pnl","notes",
            ])
        return pd.DataFrame(bets)

    # ══════════════════════════════════════════════════════════════════════════
    # Sub-tabs
    # ══════════════════════════════════════════════════════════════════════════
    bt_log_tab, bt_results_tab, bt_stats_tab = st.tabs([
        "➕ Log Bet",
        "✅ Mark Results",
        "📊 P&L Dashboard",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # LOG BET
    # ─────────────────────────────────────────────────────────────────────────
    with bt_log_tab:
        st.subheader("➕ Log a New Bet")

        if data_ok:
            all_players_bt = ["(type manually)"] + sorted(nfl_df["player_name"].unique())
        else:
            all_players_bt = ["(type manually)"]

        bl_c1, bl_c2 = st.columns([1, 1])

        with bl_c1:
            bt_date    = st.date_input("Date", value=_dtbt.date.today(), key="bt_date")
            bt_player_sel = st.selectbox("Player (from data)", all_players_bt, key="bt_player_sel")
            bt_player_txt = st.text_input(
                "Player name (manual override)",
                value="" if bt_player_sel == "(type manually)" else bt_player_sel,
                key="bt_player_txt",
            )
            bt_stat    = st.selectbox("Stat category", list(CAT_MAP.keys()),
                                       format_func=str.title, key="bt_stat")
            bt_line    = st.number_input("Prop Line", min_value=0.0, value=65.5,
                                          step=0.5, format="%.1f", key="bt_line")

        with bl_c2:
            bt_pick    = st.radio("Pick", ["OVER", "UNDER"], horizontal=True, key="bt_pick")
            bt_odds    = st.number_input(
                "American Odds (e.g. -110 or +130)",
                min_value=-10000, max_value=10000, value=-110, step=5,
                key="bt_odds",
            )
            bt_stake   = st.number_input("Stake ($)", min_value=0.01, value=10.0,
                                          step=5.0, format="%.2f", key="bt_stake")
            bt_notes   = st.text_input("Notes (optional)", key="bt_notes",
                                        placeholder="e.g. Soft matchup vs DET run D")

            # Model suggestion quick-look
            if data_ok and bt_player_txt and bt_player_txt != "(type manually)":
                res_bt = prop_analysis(nfl_df, bt_player_txt, bt_stat, bt_line,
                                       use_weighted=True, game_window="Season")
                if res_bt:
                    rec_color = "#2DC653" if res_bt["recommendation"] == "OVER" else "#D62828"
                    st.markdown(
                        f'<div style="background:{rec_color}22;border-left:4px solid {rec_color};'
                        f'padding:8px 12px;border-radius:5px;font-size:13px;">'
                        f'Model says: <b style="color:{rec_color}">{res_bt["recommendation"]}</b>'
                        f' &nbsp;·&nbsp; Wtd avg: <b>{res_bt["w_avg"]:.1f}</b>'
                        f' &nbsp;·&nbsp; Hit rate: <b>{res_bt["w_hit"]:.1f}%</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Potential payout preview
        if bt_odds > 0:
            potential_win = round(bt_stake * bt_odds / 100, 2)
        else:
            potential_win = round(bt_stake * 100 / abs(bt_odds), 2)
        potential_payout = round(bt_stake + potential_win, 2)

        st.caption(
            f"Potential payout: **${potential_payout:.2f}** "
            f"(win ${potential_win:.2f} on ${bt_stake:.2f} stake)"
        )

        if st.button("➕ Log Bet", type="primary", use_container_width=True, key="bt_log_btn"):
            player_final = bt_player_txt.strip() if bt_player_txt.strip() else bt_player_sel
            if not player_final or player_final == "(type manually)":
                st.error("Enter a player name.")
            else:
                new_bet = {
                    "id":     len(st.session_state["bt_bets"]) + 1,
                    "date":   str(bt_date),
                    "player": player_final,
                    "stat":   bt_stat,
                    "line":   bt_line,
                    "pick":   bt_pick,
                    "stake":  round(bt_stake, 2),
                    "odds":   bt_odds,
                    "result": "Pending",
                    "actual": None,
                    "pnl":    None,
                    "notes":  bt_notes.strip(),
                }
                st.session_state["bt_bets"].append(new_bet)
                _save_bets(st.session_state["bt_bets"])
                st.success(
                    f"✅ Logged: **{player_final}** {bt_stat.title()} "
                    f"{bt_pick} {bt_line} @ {bt_odds:+d} for ${bt_stake:.2f}"
                )
                st.rerun()

        # ── Pending bets quick view ───────────────────────────────────────────
        df_bt = _bets_df()
        pending = df_bt[df_bt["result"] == "Pending"] if not df_bt.empty else pd.DataFrame()
        if not pending.empty:
            st.divider()
            st.markdown(f"**{len(pending)} pending bet(s):**")
            st.dataframe(
                pending[["date","player","stat","line","pick","stake","odds","notes"]]
                .rename(columns={"date":"Date","player":"Player","stat":"Stat",
                                  "line":"Line","pick":"Pick","stake":"Stake ($)",
                                  "odds":"Odds","notes":"Notes"}),
                use_container_width=True, hide_index=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # MARK RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    with bt_results_tab:
        st.subheader("✅ Mark Results")
        df_bt = _bets_df()
        pending_r = df_bt[df_bt["result"] == "Pending"] if not df_bt.empty else pd.DataFrame()

        if pending_r.empty:
            st.info("No pending bets. Log some bets first using the ➕ Log Bet tab.")
        else:
            st.caption(
                f"{len(pending_r)} pending bet(s). Enter the actual stat value and the result "
                "will be calculated automatically."
            )

            for idx, row_r in pending_r.iterrows():
                with st.expander(
                    f"#{row_r['id']} · {row_r['date']} · {row_r['player']} "
                    f"— {row_r['stat'].title()} {row_r['pick']} {row_r['line']} "
                    f"@ {int(row_r['odds']):+d}  (${row_r['stake']:.2f})",
                    expanded=False,
                ):
                    rc1, rc2 = st.columns(2)
                    actual_val = rc1.number_input(
                        "Actual stat value",
                        min_value=0.0, value=0.0, step=0.5,
                        format="%.1f",
                        key=f"bt_actual_{row_r['id']}",
                    )
                    void_check = rc2.checkbox("Void / No action", key=f"bt_void_{row_r['id']}")

                    if st.button("Save Result", key=f"bt_save_{row_r['id']}"):
                        bets = st.session_state["bt_bets"]
                        bet_to_update = next((b for b in bets if b["id"] == row_r["id"]), None)
                        if bet_to_update is not None:
                            if void_check:
                                bet_to_update["result"] = "Void"
                                bet_to_update["actual"] = None
                                bet_to_update["pnl"]    = 0.0
                            else:
                                # Determine win/loss
                                hit = (
                                    (actual_val > row_r["line"] and row_r["pick"] == "OVER") or
                                    (actual_val < row_r["line"] and row_r["pick"] == "UNDER")
                                )
                                if actual_val == row_r["line"]:
                                    bet_to_update["result"] = "Push"
                                    bet_to_update["pnl"]    = 0.0
                                elif hit:
                                    odds_v = int(row_r["odds"])
                                    win    = (row_r["stake"] * odds_v / 100) if odds_v > 0 \
                                             else (row_r["stake"] * 100 / abs(odds_v))
                                    bet_to_update["result"] = "Win"
                                    bet_to_update["pnl"]    = round(win, 2)
                                else:
                                    bet_to_update["result"] = "Loss"
                                    bet_to_update["pnl"]    = -round(row_r["stake"], 2)
                                bet_to_update["actual"] = round(actual_val, 1)
                            _save_bets(bets)
                            st.session_state["bt_bets"] = bets
                            st.success(
                                f"Saved: {bet_to_update['result']} "
                                f"(P&L: {'+' if (bet_to_update['pnl'] or 0) >= 0 else ''}"
                                f"${bet_to_update['pnl']:.2f})"
                            )
                            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # P&L DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    with bt_stats_tab:
        st.subheader("📊 P&L Dashboard")
        df_bt = _bets_df()

        if df_bt.empty or df_bt[df_bt["result"] != "Pending"].empty:
            st.info("No settled bets yet. Log and settle some bets first.")
        else:
            settled = df_bt[df_bt["result"].isin(["Win","Loss","Push","Void"])].copy()
            settled["pnl"]   = pd.to_numeric(settled["pnl"],   errors="coerce").fillna(0)
            settled["stake"] = pd.to_numeric(settled["stake"],  errors="coerce").fillna(0)
            settled["date"]  = pd.to_datetime(settled["date"],  errors="coerce")

            total_bets   = len(settled)
            wins         = (settled["result"] == "Win").sum()
            losses       = (settled["result"] == "Loss").sum()
            pushes       = (settled["result"] == "Push").sum()
            total_staked = settled["stake"].sum()
            total_pnl    = settled["pnl"].sum()
            roi          = (total_pnl / total_staked * 100) if total_staked > 0 else 0
            win_rate     = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            avg_odds     = settled[settled["result"].isin(["Win","Loss"])]["odds"].mean()
            best_win     = settled[settled["result"] == "Win"]["pnl"].max() if wins else 0
            worst_loss   = settled[settled["result"] == "Loss"]["pnl"].min() if losses else 0

            # ── Top-line KPIs ─────────────────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total P&L",    f"{'+'if total_pnl>=0 else ''}${total_pnl:.2f}",
                      delta=f"ROI {roi:+.1f}%",
                      delta_color="normal" if total_pnl >= 0 else "inverse")
            k2.metric("Win Rate",     f"{win_rate:.1f}%",
                      delta=f"{wins}W / {losses}L / {pushes}P")
            k3.metric("Total Staked", f"${total_staked:.2f}",
                      delta=f"{total_bets} bets")
            k4.metric("Avg Odds",     f"{avg_odds:+.0f}" if not pd.isna(avg_odds) else "—")

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Best Win",    f"+${best_win:.2f}"  if wins   else "—")
            k6.metric("Worst Loss",  f"${worst_loss:.2f}" if losses  else "—")
            k7.metric("Avg Stake",   f"${settled['stake'].mean():.2f}")
            k8.metric("Avg P&L/bet", f"{'+'if total_pnl/total_bets>=0 else ''}${total_pnl/total_bets:.2f}")

            st.divider()

            # ── Cumulative P&L chart ──────────────────────────────────────────
            st.subheader("Cumulative P&L")
            cum = settled.sort_values("date").copy()
            cum["cum_pnl"] = cum["pnl"].cumsum()
            cum["bet_num"] = range(1, len(cum) + 1)

            fig_pnl, ax_pnl = plt.subplots(figsize=(11, 3.5))
            line_color = C_OVER if total_pnl >= 0 else C_LINE
            ax_pnl.plot(cum["bet_num"], cum["cum_pnl"], color=line_color,
                        linewidth=2.2, zorder=3)
            ax_pnl.fill_between(cum["bet_num"], cum["cum_pnl"], 0,
                                 alpha=0.12, color=line_color)
            ax_pnl.axhline(0, color="#888", linewidth=1, linestyle="--")
            ax_pnl.scatter(cum["bet_num"], cum["cum_pnl"],
                           c=[C_OVER if p > 0 else (C_LINE if p < 0 else "#888")
                              for p in cum["pnl"]],
                           s=28, zorder=4, edgecolors="white", linewidths=0.4)
            ax_pnl.set_xlabel("Bet #", fontsize=9)
            ax_pnl.set_ylabel("Cumulative P&L ($)", fontsize=9)
            ax_pnl.set_title("Cumulative P&L Over Time", fontsize=11, fontweight="bold")
            ax_pnl.spines["top"].set_visible(False)
            ax_pnl.spines["right"].set_visible(False)
            ax_pnl.grid(axis="y", linestyle="--", alpha=0.35)
            plt.tight_layout()
            st.pyplot(fig_pnl, use_container_width=True)
            plt.close(fig_pnl)

            # ── Per-stat breakdown ────────────────────────────────────────────
            st.divider()
            stat_cols_bt, chart_col_bt = st.columns([1, 2])

            with stat_cols_bt:
                st.subheader("By Stat Category")
                stat_summary = (
                    settled.groupby("stat")
                    .agg(
                        Bets   =("pnl", "count"),
                        Wins   =("result", lambda x: (x == "Win").sum()),
                        Losses =("result", lambda x: (x == "Loss").sum()),
                        PnL    =("pnl", "sum"),
                        Staked =("stake", "sum"),
                    )
                    .reset_index()
                )
                stat_summary["ROI %"] = (
                    stat_summary["PnL"] / stat_summary["Staked"] * 100
                ).round(1)
                stat_summary["Win %"] = (
                    stat_summary["Wins"] / (stat_summary["Wins"] + stat_summary["Losses"])
                    .replace(0, pd.NA) * 100
                ).round(1)
                stat_summary["PnL"]   = stat_summary["PnL"].round(2)
                stat_summary["stat"]  = stat_summary["stat"].str.title()
                stat_summary = stat_summary.rename(columns={"stat": "Stat"})
                st.dataframe(stat_summary[["Stat","Bets","Wins","Losses","Win %","PnL","ROI %"]],
                             use_container_width=True, hide_index=True)

            with chart_col_bt:
                st.subheader("P&L by Stat Category")
                fig_sc, ax_sc = plt.subplots(figsize=(7, max(3, len(stat_summary) * 0.6)))
                bar_cols_sc = [C_OVER if v >= 0 else C_LINE
                               for v in stat_summary["PnL"]]
                ax_sc.barh(stat_summary["Stat"][::-1], stat_summary["PnL"][::-1],
                           color=bar_cols_sc[::-1], alpha=0.85,
                           edgecolor="white", linewidth=0.4)
                ax_sc.axvline(0, color="#888", linewidth=1)
                for i, (val, label) in enumerate(zip(
                    stat_summary["PnL"][::-1], stat_summary["Stat"][::-1]
                )):
                    ax_sc.text(
                        val + (stat_summary["PnL"].abs().max() * 0.02 if val >= 0
                               else -stat_summary["PnL"].abs().max() * 0.02),
                        i, f"${val:+.2f}", va="center", fontsize=8,
                        ha="left" if val >= 0 else "right",
                    )
                ax_sc.set_xlabel("P&L ($)", fontsize=9)
                ax_sc.spines["top"].set_visible(False)
                ax_sc.spines["right"].set_visible(False)
                ax_sc.grid(axis="x", linestyle="--", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig_sc, use_container_width=True)
                plt.close(fig_sc)

            # ── Model accuracy: did the model agree with result? ───────────────
            st.divider()
            st.subheader("Model Accuracy Check")
            st.caption(
                "For each settled bet, was the model's recommendation (from the Prop Analyzer) "
                "on the same side as the winning outcome?"
            )
            win_loss = settled[settled["result"].isin(["Win","Loss"])].copy()
            if not win_loss.empty and data_ok:
                model_correct = 0
                model_checked = 0
                for _, bet_row in win_loss.iterrows():
                    res_chk = prop_analysis(
                        nfl_df, bet_row["player"], bet_row["stat"],
                        bet_row["line"], use_weighted=True, game_window="Season",
                    )
                    if res_chk is None:
                        continue
                    model_pick = res_chk["recommendation"]
                    actual_win = bet_row["result"] == "Win"
                    # Bet won means our pick was correct
                    model_matched = (model_pick == bet_row["pick"])
                    outcome_correct = (model_matched and actual_win) or (not model_matched and not actual_win)
                    model_correct += int(outcome_correct)
                    model_checked += 1

                if model_checked > 0:
                    model_acc = model_correct / model_checked * 100
                    acc_color = "#2DC653" if model_acc >= 55 else ("#f59e0b" if model_acc >= 45 else "#D62828")
                    st.markdown(
                        f'<div style="background:{acc_color}22;border-left:5px solid {acc_color};'
                        f'padding:12px 16px;border-radius:6px;font-size:15px;">'
                        f'Model agreed with winning outcome in '
                        f'<b>{model_correct} / {model_checked}</b> bets '
                        f'= <b style="color:{acc_color}">{model_acc:.1f}% accuracy</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Not enough data to compute model accuracy yet.")
            else:
                st.info("No settled Win/Loss bets to check yet." if not data_ok else
                        "Stats data needed to compute model accuracy.")

            # ── Full bet history table ────────────────────────────────────────
            st.divider()
            st.subheader("Full Bet History")
            show_all_bt = st.checkbox("Show all bets (including pending)", key="bt_show_all")
            display_bt  = df_bt if show_all_bt else settled.sort_values("date", ascending=False)

            def _pnl_color(val):
                try:
                    v = float(val)
                    if v > 0:  return "color:#2DC653;font-weight:700"
                    if v < 0:  return "color:#D62828;font-weight:700"
                except Exception:
                    pass
                return ""

            disp_bt_cols = ["id","date","player","stat","line","pick",
                            "stake","odds","result","actual","pnl","notes"]
            disp_bt_cols = [c for c in disp_bt_cols if c in display_bt.columns]
            styled_bt = (
                display_bt[disp_bt_cols]
                .rename(columns={
                    "id":"#","date":"Date","player":"Player","stat":"Stat",
                    "line":"Line","pick":"Pick","stake":"Stake","odds":"Odds",
                    "result":"Result","actual":"Actual","pnl":"P&L","notes":"Notes",
                })
                .style.map(_pnl_color, subset=["P&L"])
            )
            st.dataframe(styled_bt, use_container_width=True, hide_index=True)

            # ── Export / Danger zone ─────────────────────────────────────────
            st.divider()
            ex_c1, ex_c2 = st.columns(2)
            with ex_c1:
                csv_bt = display_bt[disp_bt_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download bet history CSV",
                    data=csv_bt,
                    file_name="bet_history.csv",
                    mime="text/csv",
                    key="bt_download",
                )
            with ex_c2:
                if st.button("🗑️ Clear ALL bets (irreversible)", key="bt_clear_all"):
                    st.session_state["bt_bets"] = []
                    _save_bets([])
                    st.success("All bets cleared.")
                    st.rerun()
