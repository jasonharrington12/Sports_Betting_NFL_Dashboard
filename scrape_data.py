"""
scrape_data.py — Local NFL data scraper
========================================
Run this from your machine (NOT on Streamlit Cloud — ESPN blocks server IPs).

Usage:
    python scrape_data.py              # scrape 2024 + 2025, append new weeks
    python scrape_data.py --full       # force full rescrape of both seasons
    python scrape_data.py --year 2025  # scrape one season only

Output: final_nfl_2024_2025_player_game_logs.csv  (commit this to your repo)
"""

import argparse
import sys
import time
import os

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_CSV  = "final_nfl_2024_2025_player_game_logs.csv"
CORE_BASE   = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DELAY       = 0.15   # seconds between requests — polite but fast

# PPR fantasy scoring
FP_WEIGHTS  = {
    "passing_yards": 0.04, "passing_tds": 4.0, "interceptions": -1.0,
    "rush_yards": 0.1,     "rush_tds": 6.0,
    "receptions": 1.0,     "receiving_yards": 0.1, "receiving_tds": 6.0,
}

TEAM_NORM = {
    "LAR": "LA", "WSH": "WAS",
}

# ── HTTP helper ───────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update(HEADERS)

def get_json(url: str):
    url = url.replace("http://", "https://")
    try:
        r = _session.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
        print(f"  WARN {r.status_code}: {url[:90]}")
    except Exception as e:
        print(f"  ERR {e}: {url[:90]}")
    return None


# ── Scraping helpers ──────────────────────────────────────────────────────────
def parse_short_name(short_name: str):
    """'BAL @ KC' → (away='BAL', home='KC')"""
    parts = short_name.split(" @ ")
    if len(parts) == 2:
        away = TEAM_NORM.get(parts[0].strip(), parts[0].strip())
        home = TEAM_NORM.get(parts[1].strip(), parts[1].strip())
        return away, home
    return "UNK", "UNK"


def scrape_game(comp, event_short_name: str, season: int, week: int) -> list:
    """Scrape one game's player stats. Returns list of row dicts."""
    away, home = parse_short_name(event_short_name)
    rows = []
    seen_player_ids = set()   # deduplicate across categories

    for competitor in comp.get("competitors", []):
        is_home    = competitor.get("homeAway") == "home"
        team_abbr  = home if is_home else away
        stats_ref  = (competitor.get("statistics") or {}).get("$ref", "")
        if not stats_ref:
            continue

        team_stats = get_json(stats_ref)
        if not team_stats:
            continue

        categories = (team_stats.get("splits") or {}).get("categories", [])
        # Build map: athlete_ref → {cat: {statname: value}}
        athlete_data: dict = {}
        for cat in categories:
            cat_name = cat.get("name", "")
            if cat_name not in ("passing", "rushing", "receiving"):
                continue
            for ath_entry in cat.get("athletes", []):
                ath_ref      = (ath_entry.get("athlete") or {}).get("$ref", "")
                stats_entry  = (ath_entry.get("statistics") or {}).get("$ref", "")
                if not ath_ref or not stats_entry:
                    continue
                if ath_ref not in athlete_data:
                    athlete_data[ath_ref] = {"_stats_ref": stats_entry}
                athlete_data[ath_ref][cat_name + "_ref"] = stats_entry

        # For each unique athlete, fetch name + per-category stats
        for ath_ref, cat_refs in athlete_data.items():
            # Extract athlete ID from ref URL
            try:
                ath_id = ath_ref.rstrip("/").rstrip("?lang=en&region=us").split("/")[-1].split("?")[0]
            except Exception:
                ath_id = ""

            if ath_id in seen_player_ids:
                continue
            seen_player_ids.add(ath_id)

            # Fetch athlete name
            ath_data = get_json(ath_ref)
            time.sleep(DELAY)
            if not ath_data:
                continue
            player_name = ath_data.get("displayName", "Unknown")

            # Collect stats from whichever category ref is available
            # (all three point to the same player-game stats endpoint)
            p_stats  = {}
            ru_stats = {}
            rc_stats = {}

            stats_ref_url = cat_refs.get("passing_ref") or cat_refs.get("rushing_ref") or cat_refs.get("receiving_ref", "")
            if stats_ref_url:
                ps_data = get_json(stats_ref_url)
                time.sleep(DELAY)
                if ps_data:
                    for cat in (ps_data.get("splits") or {}).get("categories", []):
                        smap = {s["name"]: s.get("value", 0) for s in cat.get("stats", [])}
                        if cat["name"] == "passing":
                            p_stats = smap
                        elif cat["name"] == "rushing":
                            ru_stats = smap
                        elif cat["name"] == "receiving":
                            rc_stats = smap

            completions     = int(p_stats.get("completions",        0))
            attempts        = int(p_stats.get("passingAttempts",    0))
            passing_yards   = int(p_stats.get("passingYards",       0))
            passing_tds     = int(p_stats.get("passingTouchdowns",  0))
            interceptions   = int(p_stats.get("interceptions",      0))
            rush_attempts   = int(ru_stats.get("rushingAttempts",   0))
            rush_yards      = int(ru_stats.get("rushingYards",      0))
            rush_tds        = int(ru_stats.get("rushingTouchdowns", 0))
            receptions      = int(rc_stats.get("receptions",        0))
            targets         = int(rc_stats.get("receivingTargets",  0))
            receiving_yards = int(rc_stats.get("receivingYards",    0))
            receiving_tds   = int(rc_stats.get("receivingTouchdowns", 0))

            # Skip anyone with no meaningful involvement
            if attempts < 5 and rush_attempts < 3 and targets < 1 and receptions < 1:
                continue

            fp = sum(locals().get(k, 0) * v for k, v in FP_WEIGHTS.items())

            rows.append(dict(
                player_id       = ath_id,
                game_id         = f"{season}_{week:02d}_{away}_{home}",
                season          = season,
                player_name     = player_name,
                team            = team_abbr,
                completions     = completions,
                attempts        = attempts,
                passing_yards   = passing_yards,
                passing_tds     = passing_tds,
                interceptions   = interceptions,
                rush_attempts   = rush_attempts,
                rush_yards      = rush_yards,
                rush_tds        = rush_tds,
                receptions      = receptions,
                targets         = targets,
                receiving_yards = receiving_yards,
                receiving_tds   = receiving_tds,
                fantasy_points  = round(fp, 4),
            ))

    return rows


def scrape_week(season: int, week: int, season_type: int = 2) -> list:
    """Scrape all completed games for one week."""
    url   = f"{CORE_BASE}/seasons/{season}/types/{season_type}/weeks/{week}/events?limit=20"
    data  = get_json(url)
    if not data:
        return []

    rows = []
    items = data.get("items", [])
    for idx, item in enumerate(items):
        event = get_json(item.get("$ref", ""))
        time.sleep(DELAY)
        if not event:
            continue

        short_name = event.get("shortName", "")
        comp_ref   = (event.get("competitions") or [{}])[0].get("$ref", "")
        if not comp_ref:
            continue

        comp = get_json(comp_ref)
        time.sleep(DELAY)
        if not comp:
            continue

        # Check completed via status $ref
        status_ref = (comp.get("status") or {}).get("$ref", "")
        status     = get_json(status_ref) if status_ref else {}
        time.sleep(DELAY)
        if not (status or {}).get("type", {}).get("completed", False):
            continue

        away, home = parse_short_name(short_name)
        print(f"    Game {idx+1}/{len(items)}: {away} @ {home}", end=" ... ", flush=True)
        game_rows = scrape_game(comp, short_name, season, week)
        print(f"{len(game_rows)} players")
        rows.extend(game_rows)
        time.sleep(DELAY)

    return rows


def latest_week_in_df(df: pd.DataFrame, year: int) -> int:
    if df.empty or "game_id" not in df.columns:
        return 0
    sub = df[df["season"] == year]
    if sub.empty:
        return 0
    try:
        weeks = sub["game_id"].str.split("_", expand=True)[1].dropna().astype(int)
        return int(weeks.max()) if not weeks.empty else 0
    except Exception:
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",  action="store_true", help="Force full rescrape")
    parser.add_argument("--year",  type=int, default=0, help="Scrape one year only")
    args = parser.parse_args()

    import datetime
    today    = datetime.date.today()
    cur_year = today.year if today.month >= 9 else today.year if today.month >= 2 else today.year - 1
    prev_year = cur_year - 1
    years    = [args.year] if args.year else [prev_year, cur_year]

    # Load existing CSV
    if os.path.exists(OUTPUT_CSV) and not args.full:
        print(f"Loading existing data from {OUTPUT_CSV}…")
        base = pd.read_csv(OUTPUT_CSV, low_memory=False)
        base.columns = base.columns.str.lower().str.strip()
    else:
        base = pd.DataFrame()

    all_new_rows = []

    for year in years:
        done = latest_week_in_df(base, year)
        print(f"\n{'='*50}")
        print(f"Season {year}  — already have up to week {done}")

        # Regular season
        for week in range(done + 1, 19):
            print(f"\n  Week {week}/18:")
            rows = scrape_week(year, week, season_type=2)
            if not rows and week > 1:
                print("  No data returned — season complete or not yet played.")
                break
            all_new_rows.extend(rows)
            # Save after every week so progress isn't lost
            if all_new_rows:
                _save(base, all_new_rows, OUTPUT_CSV)

        # Playoffs (season_type=3, weeks 1-4 → re-encode as 19-22)
        playoff_done = 0
        if not base.empty and "game_id" in base.columns and "season" in base.columns:
            sub = base[base["season"] == year]
            try:
                wks = sub["game_id"].str.split("_", expand=True)[1].dropna().astype(int)
                playoff_done = int(wks[wks >= 19].max()) - 18 if not wks[wks >= 19].empty else 0
            except Exception:
                playoff_done = 0

        for week in range(playoff_done + 1, 5):
            print(f"\n  Playoffs week {week}/4:")
            rows = scrape_week(year, week, season_type=3)
            if not rows and week > 1:
                print("  No playoff data — playoffs not yet played.")
                break
            # Re-encode playoff weeks as 19-22
            for r in rows:
                parts = r["game_id"].split("_")
                r["game_id"] = f"{parts[0]}_{int(parts[1])+18:02d}_{'_'.join(parts[2:])}"
            all_new_rows.extend(rows)
            if all_new_rows:
                _save(base, all_new_rows, OUTPUT_CSV)

    if all_new_rows:
        _save(base, all_new_rows, OUTPUT_CSV)
        print(f"\nDone. Saved {len(all_new_rows)} new rows to {OUTPUT_CSV}")
    else:
        print("\nNo new data found — everything is already up to date.")


def _save(base: pd.DataFrame, new_rows: list, path: str):
    new_df   = pd.DataFrame(new_rows).drop_duplicates()
    combined = pd.concat([base, new_df], ignore_index=True).drop_duplicates()
    combined.to_csv(path, index=False)


if __name__ == "__main__":
    main()
