"""
Run this once locally to build nfl_game_logs.csv.
Then commit the CSV — the app will load from it instantly on Streamlit Cloud.

    python _scrape_to_csv.py
"""
import time, re, requests, pandas as pd

headers = {'User-Agent': 'Mozilla/5.0'}
FP = {'passing_yards': 0.04, 'passing_tds': 4.0, 'interceptions': -1.0,
      'rush_yards': 0.1, 'rush_tds': 6.0, 'receptions': 1.0,
      'receiving_yards': 0.1, 'receiving_tds': 6.0}

def _safe_int(v):
    try: return int(v)
    except: return 0

def _parse_ca(s):
    m = re.match(r'(\d+)/(\d+)', str(s))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def _get_json(url):
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def scrape_game(game_id, season, week, home, away):
    url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=' + str(game_id)
    data = _get_json(url)
    if not data:
        return []
    rows = []
    for grp in data.get('boxscore', {}).get('players', []):
        team = grp.get('team', {}).get('abbreviation', 'UNK')
        sbn = {s['name']: s for s in grp.get('statistics', [])}
        aids = {}
        for cat in sbn.values():
            for e in cat.get('athletes', []):
                a = e.get('athlete', {})
                if a.get('id') and a['id'] not in aids:
                    aids[a['id']] = a.get('displayName', 'Unknown')
        for aid, name in aids.items():
            row = dict(
                player_id=aid,
                game_id=str(season) + '_' + str(week).zfill(2) + '_' + away + '_' + home,
                completions=0, attempts=0, passing_yards=0, passing_tds=0,
                interceptions=0, rush_attempts=0, rush_yards=0, rush_tds=0,
                receptions=0, targets=0, receiving_yards=0, receiving_tds=0,
                season=season, player_name=name, team=team
            )
            for e in sbn.get('passing', {}).get('athletes', []):
                if e['athlete']['id'] == aid:
                    s = e.get('stats', [])
                    if len(s) >= 5:
                        c, a2 = _parse_ca(s[0])
                        row.update(completions=c, attempts=a2,
                                   passing_yards=_safe_int(s[1]),
                                   passing_tds=_safe_int(s[3]),
                                   interceptions=_safe_int(s[4]))
                    break
            for e in sbn.get('rushing', {}).get('athletes', []):
                if e['athlete']['id'] == aid:
                    s = e.get('stats', [])
                    if len(s) >= 4:
                        row.update(rush_attempts=_safe_int(s[0]),
                                   rush_yards=_safe_int(s[1]),
                                   rush_tds=_safe_int(s[3]))
                    break
            for e in sbn.get('receiving', {}).get('athletes', []):
                if e['athlete']['id'] == aid:
                    s = e.get('stats', [])
                    if len(s) >= 4:
                        row.update(receptions=_safe_int(s[0]),
                                   receiving_yards=_safe_int(s[1]),
                                   receiving_tds=_safe_int(s[3]),
                                   targets=_safe_int(s[5]) if len(s) >= 6 else 0)
                    break
            is_passer   = row['attempts'] > 0
            is_rusher   = row['rush_attempts'] > 0
            is_receiver = row['targets'] > 0 or row['receptions'] > 0
            if not is_passer and not is_rusher and not is_receiver:
                continue
            if is_passer and not is_rusher and not is_receiver and row['attempts'] < 5:
                continue
            if is_rusher and not is_passer and not is_receiver and row['rush_attempts'] < 3:
                continue
            fp = sum(row.get(k, 0) * v for k, v in FP.items())
            row['fantasy_points'] = round(fp, 4)
            rows.append(row)
    return rows

all_rows = []
import datetime
today = datetime.date.today()
cal_year = today.year if today.month >= 9 else today.year - 1

# probe whether cal_year has completed games yet
probe = _get_json('https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week=1&dates=' + str(cal_year))
has_games = probe and any(
    e.get('competitions', [{}])[0].get('status', {}).get('type', {}).get('completed', False)
    for e in probe.get('events', [])
)
cur_year  = cal_year if has_games else cal_year - 1
prev_year = cur_year - 1

print('Scraping seasons: ' + str(prev_year) + ' and ' + str(cur_year))

for year in [prev_year, cur_year]:
    print('--- ' + str(year) + ' ---')
    for week in range(1, 19):
        sb_url = ('https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
                  '?seasontype=2&week=' + str(week) + '&dates=' + str(year))
        data = _get_json(sb_url)
        if not data:
            continue
        events = data.get('events', [])
        if not events:
            break
        completed_any = False
        game_count = 0
        for event in events:
            comp = event.get('competitions', [{}])[0]
            if not comp.get('status', {}).get('type', {}).get('completed', False):
                continue
            completed_any = True
            gid = event['id']
            comps = comp.get('competitors', [])
            home = away = 'UNK'
            for c in comps:
                ab = c.get('team', {}).get('abbreviation', 'UNK')
                if c.get('homeAway') == 'home':
                    home = ab
                else:
                    away = ab
            rows = scrape_game(gid, year, week, home, away)
            all_rows.extend(rows)
            game_count += 1
            time.sleep(0.15)
        print('  week ' + str(week) + ': ' + str(game_count) + ' games, ' + str(len(all_rows)) + ' total rows')
        if not completed_any:
            break

df = pd.DataFrame(all_rows).drop_duplicates()
df = df.sort_values(['player_name', 'season', 'game_id']).reset_index(drop=True)
df.to_csv('nfl_game_logs.csv', index=False)
print('')
print('Done! Saved ' + str(len(df)) + ' rows to nfl_game_logs.csv')
print('Seasons: ' + str(df['season'].value_counts().to_dict()))
print('Players: ' + str(df['player_name'].nunique()))
print('')
print('Now run:  git add nfl_game_logs.csv && git commit -m "data: add bundled game logs CSV" && git push')
