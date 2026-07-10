import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="NFL Prop Betting Dashboard",
    page_icon="🏈",
    layout="wide",
)

# ==============================================================================
# 1. DATA LOADING & PROCESSING  (cached so it only runs once)
# ==============================================================================

@st.cache_data
def load_data():
    df_2024 = pd.read_csv("nfl_2024_player_game_logs.csv")
    df_2025 = pd.read_csv("nfl_2025_player_game_logs.csv")

    for df in [df_2024, df_2025]:
        df.columns = df.columns.str.lower().str.strip()

    df_2024 = df_2024.drop_duplicates()
    df_2025 = df_2025.drop_duplicates()

    # --- Detect team changes between seasons ---
    teams_2024 = (
        df_2024.sort_values('game_id')
        .groupby('player_name')['team']
        .last()
        .reset_index()
        .rename(columns={'team': 'team_2024'})
    )
    teams_2025 = (
        df_2025.sort_values('game_id')
        .groupby('player_name')['team']
        .last()
        .reset_index()
        .rename(columns={'team': 'team_2025'})
    )

    team_changes = teams_2024.merge(teams_2025, on='player_name', how='inner')
    team_changes['changed_team'] = team_changes['team_2024'] != team_changes['team_2025']

    df_2024 = df_2024.merge(
        team_changes[['player_name', 'changed_team']],
        on='player_name',
        how='left'
    )
    df_2024['changed_team'] = df_2024['changed_team'].fillna(False)

    nfl_df = pd.concat([df_2024, df_2025], ignore_index=True)
    nfl_df = nfl_df.sort_values(['player_name', 'season', 'game_id']).reset_index(drop=True)

    # --- Season weights ---
    def get_season_weight(row):
        if row['season'] == 2025:
            return 1.0
        elif row['season'] == 2024 and row.get('changed_team', False):
            return 0.3
        else:
            return 0.6

    nfl_df['weight'] = nfl_df.apply(get_season_weight, axis=1)

    # --- Efficiency metrics ---
    nfl_df['completion_percentage'] = np.where(
        nfl_df['attempts'] > 0,
        nfl_df['completions'] / nfl_df['attempts'],
        0
    )
    nfl_df['yards_per_attempt'] = np.where(
        nfl_df['attempts'] > 0,
        nfl_df['passing_yards'] / nfl_df['attempts'],
        0
    )
    nfl_df['yards_per_reception'] = np.where(
        nfl_df['receptions'] > 0,
        nfl_df['receiving_yards'] / nfl_df['receptions'],
        0
    )

    # --- Prop hit indicators ---
    nfl_df['pass_yards_over'] = np.where(nfl_df['passing_yards'] > 225.5, 1, 0)
    nfl_df['rush_yards_over'] = np.where(nfl_df['rush_yards']    > 65.5,  1, 0)
    nfl_df['rec_yards_over']  = np.where(nfl_df['receiving_yards'] > 70.5, 1, 0)

    # --- Rolling averages ---
    nfl_df = nfl_df.sort_values(['player_name', 'season', 'game_id'])
    nfl_df['last_3_pass_yards_avg'] = (
        nfl_df.groupby('player_name')['passing_yards']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    nfl_df['last_3_rec_yards_avg'] = (
        nfl_df.groupby('player_name')['receiving_yards']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    nfl_df['last_3_rush_yards_avg'] = (
        nfl_df.groupby('player_name')['rush_yards']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # --- Weighted consistency ---
    def weighted_std(group):
        vals    = group['passing_yards'].values
        weights = group['weight'].values
        w_mean  = np.average(vals, weights=weights)
        w_var   = np.average((vals - w_mean) ** 2, weights=weights)
        return np.sqrt(w_var)

    player_consistency = (
        nfl_df.groupby('player_name')
        .apply(weighted_std)
        .reset_index()
    )
    player_consistency.columns = ['player_name', 'passing_yards_std_weighted']
    nfl_df = nfl_df.merge(player_consistency, on='player_name', how='left')

    # --- Fill missing values ---
    numeric_cols     = nfl_df.select_dtypes(include=['number']).columns
    categorical_cols = nfl_df.select_dtypes(include=['object']).columns
    nfl_df[numeric_cols]     = nfl_df[numeric_cols].fillna(0)
    nfl_df[categorical_cols] = nfl_df[categorical_cols].fillna('Unknown')

    nfl_df['player_name'] = nfl_df['player_name'].str.strip()
    nfl_df = nfl_df.reset_index(drop=True)

    return nfl_df, team_changes


# ==============================================================================
# 2. CHART FUNCTION  (returns fig instead of plt.show)
# ==============================================================================

CAT_MAP = {
    'pass yards': ('passing_yards',  'Passing Yards'),
    'rush yards': ('rush_yards',     'Rush Yards'),
    'rec yards' : ('receiving_yards','Receiving Yards'),
    'receptions': ('receptions',     'Receptions'),
    'pass tds'  : ('passing_tds',    'Passing TDs'),
    'fantasy'   : ('fantasy_points', 'Fantasy Points'),
}

COLOR_2024  = '#5B8DB8'
COLOR_2025  = '#E07B54'
COLOR_AVG   = '#2C2C2C'
COLOR_LINE  = '#D62828'
COLOR_OVER  = '#2DC653'
COLOR_UNDER = '#E07B54'


def plot_player_stats(nfl_df, player_name, category, line=None, game_window='Season'):
    col, col_label = CAT_MAP[category.lower()]

    player_df = nfl_df[nfl_df['player_name'].str.contains(player_name, case=False, na=False)].copy()
    if player_df.empty:
        return None, f"Player '{player_name}' not found."

    full_name   = player_df['player_name'].iloc[0]

    n_games = {'Last 3': 3, 'Last 5': 5, 'Season': None}[game_window]

    player_2024 = player_df[player_df['season'] == 2024].reset_index(drop=True)
    player_2025 = player_df[player_df['season'] == 2025].reset_index(drop=True)

    # For windowed views, keep only the tail across both seasons combined,
    # preferring the most-recent (2025 first, then 2024 to fill remaining slots)
    if n_games is not None:
        combined = player_df.tail(n_games).copy()
        player_2024 = combined[combined['season'] == 2024].reset_index(drop=True)
        player_2025 = combined[combined['season'] == 2025].reset_index(drop=True)

    player_2024['week'] = range(1, len(player_2024) + 1)
    player_2025['week'] = range(1, len(player_2025) + 1)

    has_2024 = not player_2024.empty
    has_2025 = not player_2025.empty

    ncols = 2 if (has_2024 and has_2025) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    window_str = f"Last {n_games} Games" if n_games else "Full Season"
    fig.suptitle(f"{full_name}  —  {col_label}  |  {window_str}",
                 fontsize=14, fontweight='bold', y=1.02)

    def draw_season(ax, season_df, season_label, bar_color):
        weeks  = season_df['week'].values
        values = season_df[col].values
        avg    = values.mean()

        colors = [COLOR_OVER if (line is not None and v > line) else
                  (COLOR_UNDER if line is not None else bar_color)
                  for v in values]

        bars = ax.bar(weeks, values, color=colors, edgecolor='white',
                      linewidth=0.6, alpha=0.85, zorder=2)

        ax.axhline(avg, color=COLOR_AVG, linewidth=1.8, linestyle='--',
                   label=f'Season Avg: {avg:.1f}', zorder=3)

        if line is not None:
            ax.axhline(line, color=COLOR_LINE, linewidth=1.8, linestyle='-',
                       label=f'Prop Line: {line}', zorder=3)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    str(int(val)) if val == int(val) else f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7.5, color='#333333'
                )

        ax.set_title(season_label, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel(col_label, fontsize=10)
        ax.set_xticks(weeks)
        ax.set_xticklabels([str(w) for w in weeks], fontsize=8)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 10)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.85)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    idx = 0
    if has_2024:
        draw_season(axes[idx], player_2024, '2024 Season', COLOR_2024)
        idx += 1
    if has_2025:
        draw_season(axes[idx], player_2025, '2025 Season', COLOR_2025)

    if line is not None:
        legend_patches = [
            mpatches.Patch(color=COLOR_OVER,  label=f'Over {line}'),
            mpatches.Patch(color=COLOR_UNDER, label=f'Under {line}'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=2,
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    return fig, full_name


# ==============================================================================
# 3. PROP ANALYSIS  (returns dict instead of printing)
# ==============================================================================

def get_prop_analysis(nfl_df, player_name, category, line, use_weighted=True, game_window='Season'):
    col = CAT_MAP[category.lower()][0]

    player_df = nfl_df[nfl_df['player_name'].str.contains(player_name, case=False, na=False)]
    if player_df.empty:
        return None

    full_name   = player_df['player_name'].iloc[0]
    player_2025 = player_df[player_df['season'] == 2025]
    player_2024 = player_df[player_df['season'] == 2024]
    changed     = player_df['changed_team'].any() if 'changed_team' in player_df.columns else False

    # --- Window slice (applied to most-recent games across the combined sorted df) ---
    n_games = {'Last 3': 3, 'Last 5': 5, 'Season': None}[game_window]
    if n_games is not None:
        window_df = player_df.tail(n_games)
    else:
        window_df = player_df

    def hit_rate(df, col, line):
        if df.empty:
            return None, None, None
        over  = (df[col] > line).sum()
        total = len(df)
        return (over / total) * 100, over, total

    hr_2025, over_2025, total_2025 = hit_rate(player_2025, col, line)
    hr_2024, over_2024, total_2024 = hit_rate(player_2024, col, line)

    # Weighted avg / hit-rate calculated over the selected window
    if use_weighted:
        weights = window_df['weight'].values
        vals    = window_df[col].values
        w_avg   = np.average(vals, weights=weights) if len(vals) else 0.0
        w_hit   = np.average((vals > line).astype(float), weights=weights) * 100 if len(vals) else 0.0
    else:
        w_avg = window_df[col].mean() if not window_df.empty else 0.0
        w_hit = (window_df[col] > line).mean() * 100 if not window_df.empty else 0.0

    window_avg = window_df[col].mean() if not window_df.empty else 0.0
    window_hit = (window_df[col] > line).mean() * 100 if not window_df.empty else 0.0

    std_dev        = window_df[col].std() if not window_df.empty else 0.0
    recommendation = 'OVER' if w_avg > line else 'UNDER'

    team_24 = player_2024['team'].iloc[-1] if not player_2024.empty else 'N/A'
    team_25 = player_2025['team'].iloc[-1] if not player_2025.empty else 'N/A'

    return {
        'full_name'    : full_name,
        'changed'      : changed,
        'team_24'      : team_24,
        'team_25'      : team_25,
        'hr_2025'      : hr_2025,
        'over_2025'    : over_2025,
        'total_2025'   : total_2025,
        'avg_2025'     : player_2025[col].mean() if not player_2025.empty else None,
        'hr_2024'      : hr_2024,
        'over_2024'    : over_2024,
        'total_2024'   : total_2024,
        'avg_2024'     : player_2024[col].mean() if not player_2024.empty else None,
        'w_avg'        : w_avg,
        'w_hit'        : w_hit,
        'weight_label' : '0.3' if changed else '0.6',
        'window_avg'   : window_avg,
        'window_hit'   : window_hit,
        'window_label' : game_window,
        'window_games' : len(window_df),
        'std_dev'      : std_dev,
        'recommendation': recommendation,
    }


# ==============================================================================
# 4. STREAMLIT LAYOUT
# ==============================================================================

st.title("🏈 NFL Prop Betting Dashboard")
st.caption("2024 + 2025 season game logs · weighted prop analysis")

# Load data
with st.spinner("Loading & processing data..."):
    try:
        nfl_df, team_changes = load_data()
        data_ok = True
    except FileNotFoundError as e:
        st.error(f"CSV file not found: {e}\n\nMake sure `nfl_2024_player_game_logs.csv` and "
                 f"`nfl_2025_player_game_logs.csv` are in the same folder as `app.py`.")
        data_ok = False

if data_ok:
    all_players = sorted(nfl_df['player_name'].unique().tolist())

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Prop Analysis Controls")

        player_input = st.selectbox(
            "Player",
            options=all_players,
            index=all_players.index("Drake Maye") if "Drake Maye" in all_players else 0,
        )

        category = st.selectbox(
            "Category",
            options=list(CAT_MAP.keys()),
            format_func=lambda x: x.title(),
        )

        line = st.number_input(
            "Prop Line",
            min_value=0.0,
            value=200.5,
            step=0.5,
            format="%.1f",
        )

        use_weighted = st.toggle("Use Season Weighting", value=True,
                                 help="2025 = 1.0 · 2024 = 0.6 · 2024 team-changer = 0.3")

        game_window = st.radio(
            "Game Window",
            options=["Last 3", "Last 5", "Season"],
            index=2,
            horizontal=True,
            help="Limit the analysis and chart to the most recent N games.",
        )

        analyze = st.button("Analyze", type="primary", use_container_width=True)

        st.divider()
        st.subheader("Dataset Info")
        st.metric("Total Rows",    f"{len(nfl_df):,}")
        st.metric("Total Players", f"{nfl_df['player_name'].nunique():,}")
        team_changed_count = team_changes['changed_team'].sum()
        st.metric("Team Changes (2024→2025)", int(team_changed_count))

    # ── Main panel ──────────────────────────────────────────────────────────
    if analyze:
        result = get_prop_analysis(nfl_df, player_input, category, line, use_weighted, game_window)

        if result is None:
            st.error(f"Player '{player_input}' not found in dataset.")
        else:
            # Team change warning
            if result['changed']:
                st.warning(
                    f"⚠️ Team change detected: **{result['team_24']}** (2024) → "
                    f"**{result['team_25']}** (2025).  "
                    f"2024 data is down-weighted to {result['weight_label']}."
                )

            # ── Recommendation banner ──
            rec = result['recommendation']
            banner_color = "#2DC653" if rec == "OVER" else "#D62828"
            st.markdown(
                f"""
                <div style="background:{banner_color};color:#fff;padding:14px 20px;
                            border-radius:8px;font-size:22px;font-weight:700;
                            text-align:center;margin-bottom:16px;">
                    Suggested Bet: {rec} &nbsp;{line}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Key metrics row ──
            window_label = result['window_label']
            window_games = result['window_games']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Weighted Avg",      f"{result['w_avg']:.1f}",
                        help=f"Weighted average over {window_label} ({window_games} games)")
            col2.metric("Weighted Hit Rate", f"{result['w_hit']:.1f}%",
                        help=f"Hit rate over {window_label} ({window_games} games)")
            col3.metric(f"{window_label} Avg", f"{result['window_avg']:.1f}",
                        help=f"Simple average over {window_label} ({window_games} games)")
            col4.metric("Std Deviation",     f"{result['std_dev']:.1f}",
                        help=f"Std dev over {window_label} ({window_games} games)")

            # ── Season split table ──
            st.subheader("Season Split")
            rows = []
            if result['hr_2025'] is not None:
                rows.append({
                    "Season"   : "2025",
                    "Hit Rate" : f"{result['hr_2025']:.1f}%",
                    "Over/Total": f"{int(result['over_2025'])}/{result['total_2025']}",
                    "Average"  : f"{result['avg_2025']:.1f}",
                    "Weight"   : "1.0",
                })
            if result['hr_2024'] is not None:
                rows.append({
                    "Season"   : "2024",
                    "Hit Rate" : f"{result['hr_2024']:.1f}%",
                    "Over/Total": f"{int(result['over_2024'])}/{result['total_2024']}",
                    "Average"  : f"{result['avg_2024']:.1f}",
                    "Weight"   : result['weight_label'],
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Chart ──
            st.subheader("Week-by-Week Performance")
            fig, full_name = plot_player_stats(nfl_df, player_input, category, line=line, game_window=game_window)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    else:
        # Landing state — show team-level overview
        st.info("👈 Select a player and prop line in the sidebar, then click **Analyze**.")

        st.subheader("2025 Average Fantasy Points by Team")
        team_avg = (
            nfl_df[nfl_df['season'] == 2025]
            .groupby('team')['fantasy_points']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={'team': 'Team', 'fantasy_points': 'Avg Fantasy Points'})
        )
        team_avg['Avg Fantasy Points'] = team_avg['Avg Fantasy Points'].round(2)
        st.dataframe(team_avg, use_container_width=True, hide_index=True)

        st.subheader("Players Who Changed Teams (2024 → 2025)")
        changed_df = team_changes[team_changes['changed_team']][['player_name', 'team_2024', 'team_2025']]
        changed_df = changed_df.rename(columns={
            'player_name': 'Player', 'team_2024': '2024 Team', 'team_2025': '2025 Team'
        })
        st.dataframe(changed_df, use_container_width=True, hide_index=True)
