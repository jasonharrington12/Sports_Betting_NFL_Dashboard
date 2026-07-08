# 🏈 NFL Prop Betting Dashboard

A Streamlit dashboard for analyzing NFL player props using 2024 + 2025 season game logs scraped live from the ESPN public API.

## Features

| Tab | Description |
|---|---|
| 📊 Prop Analyzer | Pick a player, stat, and line → weighted OVER/UNDER recommendation + bar chart |
| 👤 Player Profile | Full game log + rolling average trend across both seasons |
| 🏟️ Team Overview | Fantasy points by team, top players, team summary stats |
| 🏆 League Leaders | Sortable leaderboard for any stat category |
| 🔄 Data Refresh | Scrape or incrementally update data from inside the dashboard |
| 🆚 Matchup Edge | Evaluate a prop against the opposing team's defensive averages |
| 🎰 Parlay Builder | Build multi-leg parlays with combined probability and payout estimates |

## Data Source

All data is pulled from the **ESPN public API** — no API key or login required.  
On first load the app automatically scrapes the 2024 and 2025 regular seasons (~18 weeks each).

## Running Locally

```bash
pip install -r requirements.txt
streamlit run dashboard2.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).  
Data is fetched live from the ESPN API on first load and cached for the session.
