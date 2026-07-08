"""
scrape_nfl_data.py
==================
Scrapes NFL player game logs from the ESPN public API and saves them as:
    nfl_2024_player_game_logs.csv
    nfl_2025_player_game_logs.csv

These CSVs match the exact column schema expected by dashboard2.py / app.py:
    player_id, game_id, completions, attempts, passing_yards, passing_tds,
    interceptions, rush_attempts, rush_yards, rush_tds, receptions, targets,
    receiving_yards, receiving_tds, season, fantasy_points, player_name, team

Fantasy points formula (standard PPR):
    passing_yards  * 0.04
    passing_tds    * 4
    interceptions  * -1
    rush_yards     * 0.1
    rush_tds       * 6
    receptions     * 1      (PPR)
    receiving_yards* 0.1
    receiving_tds  * 6

Usage:
    python scrape_nfl_data.py                       # full scrape of 2024 + 2025
    python scrape_nfl_data.py --year 2025           # full scrape of one season
    python scrape_nfl_data.py --update              # NEW: only fetch weeks not yet in CSVs
    python scrape_nfl_data.py --update --year 2025  # incremental for one year

Season awareness (--update mode):
    - Offseason (Feb 4 - Sep 3): skips automatically, nothing to update.
    - Regular season (Sep 4 - Jan): detects the latest week already saved and
      only scrapes weeks after that. Typically finishes in ~2 minutes.
    - Automatically handles new season years (e.g. 2026) without code changes.

Requirements:  requests, pandas  (both included in Anaconda)
"""

import argparse
import datetime
import os
import time
import re
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?seasontype=2&week={week}&dates={year}"
)
BASE_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    "?event={game_id}"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DELAY   = 0.4   # seconds between requests — be polite to the API
WEEKS   = range(1, 19)   # regular season weeks 1–18

# NFL regular season runs roughly Sep 4 – Jan 6 (week 1 through week 18)
SEASON_START_MONTH = 9   # September
SEASON_END_MONTH   = 1   # January (next calendar year)

# Fantasy scoring weights (PPR)
FP = {
    "passing_yards"  : 0.04,
    "passing_tds"    : 4.0,
    "interceptions"  : -1.0,
    "rush_yards"     : 0.1,
    "rush_tds"       : 6.0,
    "receptions"     : 1.0,
    "receiving_yards": 0.1,
    "receiving_tds"  : 6.0,
}

COL_ORDER = [
    "player_id", "game_id", "completions", "attempts",
    "passing_yards", "passing_tds", "interceptions",
    "rush_attempts", "rush_yards", "rush_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "season", "fantasy_points", "player_name", "team",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_completions_attempts(ca_str):
    """'26/41' -> (26, 41)"""
    m = re.match(r"(\d+)/(\d+)", str(ca_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def calc_fantasy(row):
    return (
        row["passing_yards"]    * FP["passing_yards"]
        + row["passing_tds"]    * FP["passing_tds"]
        + row["interceptions"]  * FP["interceptions"]
        + row["rush_yards"]     * FP["rush_yards"]
        + row["rush_tds"]       * FP["rush_tds"]
        + row["receptions"]     * FP["receptions"]
        + row["receiving_yards"]* FP["receiving_yards"]
        + row["receiving_tds"]  * FP["receiving_tds"]
    )


def get_json(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} -> {url}")
        except requests.RequestException as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
        time.sleep(1.5)
    return None


# ---------------------------------------------------------------------------
# GAME SCRAPER
# ---------------------------------------------------------------------------

def scrape_game(game_id, season, week, home_team, away_team):
    """
    Fetches boxscore for one game and returns a list of player-stat dicts.
    """
    url  = BASE_SUMMARY.format(game_id=game_id)
    data = get_json(url)
    if data is None:
        return []

    boxscore      = data.get("boxscore", {})
    player_groups = boxscore.get("players", [])
    rows          = []

    for group in player_groups:
        team_abbr    = group.get("team", {}).get("abbreviation", "UNK")
        stats_list   = group.get("statistics", [])
        stat_by_name = {s["name"]: s for s in stats_list}

        # Gather all athlete IDs that appear in any stat category
        athlete_ids = {}
        for cat in stat_by_name.values():
            for entry in cat.get("athletes", []):
                ath  = entry.get("athlete", {})
                aid  = ath.get("id", "")
                name = ath.get("displayName", "Unknown")
                if aid and aid not in athlete_ids:
                    athlete_ids[aid] = name

        for athlete_id, player_name in athlete_ids.items():
            row = {
                "player_id"      : athlete_id,
                "game_id"        : f"{season}_{week:02d}_{away_team}_{home_team}",
                "completions"    : 0,
                "attempts"       : 0,
                "passing_yards"  : 0,
                "passing_tds"    : 0,
                "interceptions"  : 0,
                "rush_attempts"  : 0,
                "rush_yards"     : 0,
                "rush_tds"       : 0,
                "receptions"     : 0,
                "targets"        : 0,
                "receiving_yards": 0,
                "receiving_tds"  : 0,
                "season"         : season,
                "player_name"    : player_name,
                "team"           : team_abbr,
            }

            # PASSING  Labels: ['C/ATT', 'YDS', 'AVG', 'TD', 'INT', 'SACKS', 'QBR', 'RTG']
            for entry in stat_by_name.get("passing", {}).get("athletes", []):
                if entry["athlete"]["id"] == athlete_id:
                    s = entry.get("stats", [])
                    if len(s) >= 5:
                        comp, att = parse_completions_attempts(s[0])
                        row["completions"]   = comp
                        row["attempts"]      = att
                        row["passing_yards"] = safe_int(s[1])
                        row["passing_tds"]   = safe_int(s[3])
                        row["interceptions"] = safe_int(s[4])
                    break

            # RUSHING  Labels: ['CAR', 'YDS', 'AVG', 'TD', 'LONG']
            for entry in stat_by_name.get("rushing", {}).get("athletes", []):
                if entry["athlete"]["id"] == athlete_id:
                    s = entry.get("stats", [])
                    if len(s) >= 4:
                        row["rush_attempts"] = safe_int(s[0])
                        row["rush_yards"]    = safe_int(s[1])
                        row["rush_tds"]      = safe_int(s[3])
                    break

            # RECEIVING  Labels: ['REC', 'YDS', 'AVG', 'TD', 'LONG', 'TGTS']
            for entry in stat_by_name.get("receiving", {}).get("athletes", []):
                if entry["athlete"]["id"] == athlete_id:
                    s = entry.get("stats", [])
                    if len(s) >= 4:
                        row["receptions"]     = safe_int(s[0])
                        row["receiving_yards"]= safe_int(s[1])
                        row["receiving_tds"]  = safe_int(s[3])
                        row["targets"]        = safe_int(s[5]) if len(s) >= 6 else 0
                    break

            stat_sum = (
                row["attempts"] + row["rush_attempts"]
                + row["receptions"] + row["targets"]
            )
            if stat_sum == 0:
                continue

            row["fantasy_points"] = round(calc_fantasy(row), 4)
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# SEASON SCRAPER  (full or partial week range)
# ---------------------------------------------------------------------------

def scrape_season(year, weeks=None):
    """
    Scrape a full or partial season.
    weeks: iterable of week numbers to scrape. Defaults to all 18 weeks.
    """
    weeks = list(weeks or WEEKS)

    print(f"\n{'='*55}")
    print(f"  Scraping {year} NFL  —  weeks {weeks[0]}–{weeks[-1]}")
    print(f"{'='*55}")

    all_rows = []

    for week in weeks:
        url  = BASE_SCOREBOARD.format(week=week, year=year)
        data = get_json(url)
        if data is None:
            print(f"  Week {week}: failed to fetch scoreboard, skipping.")
            continue

        events = data.get("events", [])
        if not events:
            print(f"  Week {week}: no games found yet — stopping.")
            break   # no point checking later weeks if this one is empty

        print(f"  Week {week:2d}: {len(events)} games", end="", flush=True)

        for event in events:
            # Skip games not yet completed
            completed = (
                event.get("competitions", [{}])[0]
                .get("status", {})
                .get("type", {})
                .get("completed", False)
            )
            if not completed:
                continue

            game_id     = event["id"]
            comps       = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])

            home_team = away_team = "UNK"
            for comp in competitors:
                abbr = comp.get("team", {}).get("abbreviation", "UNK")
                if comp.get("homeAway") == "home":
                    home_team = abbr
                else:
                    away_team = abbr

            game_rows = scrape_game(game_id, year, week, home_team, away_team)
            all_rows.extend(game_rows)
            time.sleep(DELAY)

        print(f"  -> {len(all_rows)} player rows so far")

    return all_rows


# ---------------------------------------------------------------------------
# SEASON / OFFSEASON DETECTION
# ---------------------------------------------------------------------------

def current_nfl_year():
    """
    Returns the NFL season year currently active or most recently completed.
    NFL seasons straddle two calendar years (e.g. the 2025 season runs Sep 2025 - Jan 2026).
    Jan-Aug -> still the previous season's year.
    """
    today = datetime.date.today()
    return today.year if today.month >= 9 else today.year - 1


def is_offseason():
    """
    Returns True if today is in the NFL offseason (roughly Feb 4 - Sep 3).
    """
    today = datetime.date.today()
    offseason_start = datetime.date(today.year, 2, 4)
    offseason_end   = datetime.date(today.year, 9, 3)
    return offseason_start <= today <= offseason_end


def latest_week_in_csv(year):
    """
    Reads the existing CSV and returns the highest week already stored, or 0.
    Week is parsed from game_id format: "{year}_{week:02d}_{away}_{home}"
    """
    csv_path = f"nfl_{year}_player_game_logs.csv"
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path, usecols=["game_id"])
        if df.empty:
            return 0
        weeks = (
            df["game_id"]
            .str.split("_", expand=True)[1]
            .dropna()
            .astype(int)
        )
        return int(weeks.max()) if not weeks.empty else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# INCREMENTAL UPDATE
# ---------------------------------------------------------------------------

def update_season(year):
    """
    Only scrapes weeks not yet in the CSV and appends them.
    Returns number of new rows added.
    """
    csv_path  = f"nfl_{year}_player_game_logs.csv"
    last_week = latest_week_in_csv(year)
    next_week = last_week + 1

    if next_week > 18:
        print(f"  {year}: all 18 weeks already saved. Nothing to update.")
        return 0

    print(f"  {year}: CSV has weeks 1-{last_week}. Fetching week {next_week} onward...")
    new_rows = scrape_season(year, weeks=range(next_week, 19))

    if not new_rows:
        print(f"  {year}: no new completed games found.")
        return 0

    new_df = pd.DataFrame(new_rows)[COL_ORDER]

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates()
    combined = combined.sort_values(["player_name", "game_id"]).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)

    added = len(new_df)
    print(f"\n  Appended {added} new rows -> {csv_path}  (total: {len(combined)})")
    return added


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape NFL game logs from ESPN API")
    parser.add_argument(
        "--year", nargs="+", type=int, default=None,
        help="Season year(s) to scrape. Defaults to current + previous season."
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Incremental mode: only fetch weeks not yet in the CSV (~2 min)."
    )
    args = parser.parse_args()

    cur_year = current_nfl_year()

    # ── UPDATE MODE ────────────────────────────────────────────────────────
    if args.update:
        if is_offseason():
            print("Offseason detected (Feb 4 - Sep 3). No new games to fetch.")
            print(f"  Next season starts around Sep 4, {datetime.date.today().year}.")
            return

        years = args.year if args.year else [cur_year]
        print(f"Incremental update mode  —  season year(s): {years}")

        total_new = 0
        for year in years:
            total_new += update_season(year)

        if total_new == 0:
            print("\nAlready up to date — no new rows added.")
        else:
            print(f"\nUpdate complete — {total_new} new rows added across all seasons.")
        return

    # ── FULL SCRAPE MODE ───────────────────────────────────────────────────
    years = args.year if args.year else [cur_year - 1, cur_year]

    for year in years:
        rows = scrape_season(year)
        if not rows:
            print(f"No data scraped for {year}.")
            continue

        df = pd.DataFrame(rows)[COL_ORDER]
        df = df.drop_duplicates()
        df = df.sort_values(["player_name", "game_id"]).reset_index(drop=True)

        out_file = f"nfl_{year}_player_game_logs.csv"
        df.to_csv(out_file, index=False)
        print(f"\nSaved {len(df)} rows to {out_file}")
        print(f"  Players : {df['player_name'].nunique()}")
        print(f"  Teams   : {df['team'].nunique()}")
        weeks_found = df["game_id"].str.split("_", expand=True)[1].astype(int).max()
        print(f"  Weeks   : 1-{weeks_found}")


if __name__ == "__main__":
    main()
