"""MLB Stats API 데이터 수집 모듈"""

import requests
import time
from datetime import datetime, timedelta, timezone

BASE_URL = "https://statsapi.mlb.com/api/v1"
LIVE_URL = "https://statsapi.mlb.com/api/v1.1"


def get_schedule(date_str):
    """날짜(YYYY-MM-DD)의 MLB 경기 목록 반환"""
    url = f"{BASE_URL}/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "team,linescore",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date in data.get("dates", []):
        for game in date.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            games.append({
                "gamePk":       game["gamePk"],
                "status":       game["status"]["detailedState"],
                "home_team":    home.get("team", {}).get("name", "Home"),
                "away_team":    away.get("team", {}).get("name", "Away"),
                "home_team_id": home.get("team", {}).get("id"),   # ← 팀 ID 추가
                "away_team_id": away.get("team", {}).get("id"),   # ← 팀 ID 추가
                "home_score":   home.get("score", 0) or 0,
                "away_score":   away.get("score", 0) or 0,
                "game_date":    game.get("gameDate", ""),
                "venue":        game.get("venue", {}).get("name", ""),
            })
    return games


_KST = timezone(timedelta(hours=9))


def get_schedule_kst(kst_date_str):
    """
    한국 시간(KST) 기준 날짜(YYYY-MM-DD)에 해당하는 MLB 경기 목록 반환.

    MLB 스케줄 API의 'date'는 미국 현지 날짜 기준이라, 한국 시간으로는
    경기가 다음날 새벽에 진행되는 경우가 많다. 이를 보정하기 위해
    인접한 미국 날짜(전날~다음날)의 경기를 모두 가져온 뒤,
    실제 경기 시작 시각(UTC -> KST 변환)이 요청한 KST 날짜와
    일치하는 경기만 걸러서 반환한다.
    """
    target = datetime.strptime(kst_date_str, "%Y-%m-%d").date()

    games_by_pk = {}
    for offset in (-1, 0, 1):
        us_date = (target + timedelta(days=offset)).strftime("%Y-%m-%d")
        for g in get_schedule(us_date):
            games_by_pk[g["gamePk"]] = g

    result = []
    for g in games_by_pk.values():
        game_date = g.get("game_date", "")
        if not game_date:
            continue
        try:
            utc_dt = datetime.strptime(game_date, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        kst_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(_KST)
        if kst_dt.date() == target:
            result.append(g)

    result.sort(key=lambda g: g.get("game_date", ""))
    return result


def get_game_feed(game_pk):
    """경기 전체 play-by-play 피드 반환"""
    url = f"{LIVE_URL}/game/{game_pk}/feed/live"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def is_game_live(game):
    """경기가 현재 진행 중인지 여부"""
    status = game.get("status", "")
    return "Progress" in status or "Warmup" in status or "Delayed" in status


def get_current_at_bat(game_feed):
    """
    진행 중인 경기의 현재 타석 정보 반환.
    linescore + currentPlay 조합으로 구성.
    """
    live = game_feed.get("liveData", {})
    linescore = live.get("linescore", {})
    current_play = live.get("plays", {}).get("currentPlay", {})
    game_data = game_feed.get("gameData", {})

    # 팀 이름
    home_team = game_data.get("teams", {}).get("home", {}).get("name", "Home")
    away_team = game_data.get("teams", {}).get("away", {}).get("name", "Away")

    # 점수
    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0) or 0
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0) or 0

    # 이닝
    inning = linescore.get("currentInning", 0)
    inning_half = linescore.get("inningHalf", "Top")   # "Top" or "Bottom"
    is_top = inning_half.lower() in ("top", "상")

    # 카운트
    count = current_play.get("count", {})
    balls   = count.get("balls", 0)
    strikes = count.get("strikes", 0)
    outs    = count.get("outs", 0)

    # 타자 / 투수
    matchup = current_play.get("matchup", {})
    batter  = matchup.get("batter", {}).get("fullName", "-")
    pitcher = matchup.get("pitcher", {}).get("fullName", "-")

    # 주자 (linescore.offense 에서 first/second/third)
    offense = linescore.get("offense", {})
    runner_1b = bool(offense.get("first"))
    runner_2b = bool(offense.get("second"))
    runner_3b = bool(offense.get("third"))

    # 경기 상태
    abstract_state = game_data.get("status", {}).get("abstractGameState", "")
    is_live = abstract_state == "Live"
    is_final = abstract_state == "Final"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "inning": inning,
        "is_top": int(is_top),
        "inning_half": inning_half,
        "balls": balls,
        "strikes": strikes,
        "outs": outs,
        "batter": batter,
        "pitcher": pitcher,
        "runner_1b": int(runner_1b),
        "runner_2b": int(runner_2b),
        "runner_3b": int(runner_3b),
        "is_live": is_live,
        "is_final": is_final,
        "abstract_state": abstract_state,
    }


def get_standings(season=None):
    """
    MLB 리그 순위 반환.
    반환 형식:
    {
      "AL": {"동부": [...], "중부": [...], "서부": [...]},
      "NL": {"동부": [...], "중부": [...], "서부": [...]}
    }
    각 팀 dict: name, wins, losses, pct, gb, home, away, last10, streak, division_id
    """
    if season is None:
        season = datetime.now().year

    url = f"{BASE_URL}/standings"
    params = {
        "leagueId": "103,104",
        "season": season,
        "standingsTypes": "regularSeason",
        "hydrate": "team,record(overallRecords)",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # division id 매핑
    _DIV = {
        200: ("AL", "서부"), 201: ("AL", "동부"), 202: ("AL", "중부"),
        203: ("NL", "서부"), 204: ("NL", "동부"), 205: ("NL", "중부"),
    }

    result = {
        "AL": {"동부": [], "중부": [], "서부": []},
        "NL": {"동부": [], "중부": [], "서부": []},
    }

    for record in data.get("records", []):
        div_id = record.get("division", {}).get("id")
        if div_id not in _DIV:
            continue
        league, div_name = _DIV[div_id]

        for tr in record.get("teamRecords", []):
            team_name = tr.get("team", {}).get("name", "")
            wins   = tr.get("wins", 0)
            losses = tr.get("losses", 0)
            pct    = tr.get("winningPercentage", ".000")
            gb     = tr.get("gamesBack", "-")

            # 홈/원정
            splits = {s.get("type", ""): s for s in tr.get("records", {}).get("splitRecords", [])}
            home_r = splits.get("home", {})
            away_r = splits.get("away", {})
            home_str  = f"{home_r.get('wins',0)}-{home_r.get('losses',0)}"
            away_str  = f"{away_r.get('wins',0)}-{away_r.get('losses',0)}"

            # 최근 10경기
            last10_r  = splits.get("lastTen", {})
            last10_str = f"{last10_r.get('wins',0)}-{last10_r.get('losses',0)}"

            # 연속
            streak = tr.get("streak", {}).get("streakCode", "-")

            result[league][div_name].append({
                "name":    team_name,
                "wins":    wins,
                "losses":  losses,
                "pct":     pct,
                "gb":      gb,
                "home":    home_str,
                "away":    away_str,
                "last10":  last10_str,
                "streak":  streak,
            })

    return result


def get_historical_game_pks(start_date, end_date, max_games=300):
    """학습용 완료 경기 PK 목록 수집"""
    game_pks = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end and len(game_pks) < max_games:
        date_str = current.strftime("%Y-%m-%d")
        try:
            games = get_schedule(date_str)
            for g in games:
                if "Final" in g["status"] and len(game_pks) < max_games:
                    game_pks.append(g["gamePk"])
        except Exception as e:
            print(f"  {date_str} 오류: {e}")
        current += timedelta(days=1)
        time.sleep(0.05)

    return game_pks
