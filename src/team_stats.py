"""
MLB 경기 전 예측용 팀/투수 통계 수집 모듈
MLB Stats API를 이용해 시즌 성적, 투구/타격 지표, 선발투수, 상대전적을 수집한다.
"""

import datetime
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"

# 간단 캐시 (30분)
_cache: dict = {}
_cache_ts: dict = {}
_CACHE_TTL = 1800  # 초


def _get(url: str, params: dict | None = None) -> dict:
    key = url + str(sorted((params or {}).items()))
    now = datetime.datetime.now().timestamp()
    if key in _cache and now - _cache_ts.get(key, 0) < _CACHE_TTL:
        return _cache[key]
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        _cache[key] = data
        _cache_ts[key] = now
        return data
    except Exception as e:
        print(f"  [team_stats] 요청 실패 ({url}): {e}")
        return {}


def _current_season() -> int:
    n = datetime.datetime.now()
    return n.year if n.month >= 3 else n.year - 1


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_pct(val, default: float = 0.250) -> float:
    """'.280' / '0.280' / '.---' 형태 파싱"""
    if val is None:
        return default
    try:
        s = str(val).strip()
        if s in ("", "---", ".---", "-.---"):
            return default
        return float(s) if "." in s else float(f"0.{s}")
    except (ValueError, TypeError):
        return default


# 시즌 팀 순위/기록

def get_team_records(season: int | None = None) -> dict:
    """
    팀별 시즌 기록 반환.  key = team_id (int)
    포함 정보: win_pct, last10_pct, home_win_pct, away_win_pct,
               streak, runs_scored, runs_allowed, wins, losses
    """
    if season is None:
        season = _current_season()

    data = _get(
        f"{MLB_API}/standings",
        {"leagueId": "103,104", "season": season,
         "standingsTypes": "regularSeason",
         "hydrate": "team,record(overallRecords)"},
    )

    records: dict = {}
    for div in data.get("records", []):
        for tr in div.get("teamRecords", []):
            tid = tr.get("team", {}).get("id")
            if not tid:
                continue

            wins   = tr.get("wins", 0)
            losses = tr.get("losses", 0)
            games  = wins + losses

            splits = {
                s.get("type", ""): s
                for s in tr.get("records", {}).get("splitRecords", [])
            }

            def _spct(stype: str) -> float:
                s = splits.get(stype, {})
                w, l = s.get("wins", 0), s.get("losses", 0)
                return w / (w + l) if (w + l) > 0 else None  # type: ignore

            home_s   = splits.get("home", {})
            away_s   = splits.get("away", {})
            last10_s = splits.get("lastTen", {})

            # 시즌 전체 득점/실점 (runsScored/runsAllowed 없으면 0)
            runs_scored  = tr.get("runsScored", 0) or 0
            runs_allowed = tr.get("runsAllowed", 0) or 0

            records[tid] = {
                "team_id":       tid,
                "team_name":     tr.get("team", {}).get("name", ""),
                "wins":          wins,
                "losses":        losses,
                "games":         games,
                "win_pct":       wins / games if games > 0 else 0.500,
                "home_wins":     home_s.get("wins", 0),
                "home_losses":   home_s.get("losses", 0),
                "home_win_pct":  _spct("home") or 0.500,
                "away_wins":     away_s.get("wins", 0),
                "away_losses":   away_s.get("losses", 0),
                "away_win_pct":  _spct("away") or 0.500,
                "last10_wins":   last10_s.get("wins", 0),
                "last10_losses": last10_s.get("losses", 0),
                "last10_pct":    _spct("lastTen") or 0.500,
                "streak":        tr.get("streak", {}).get("streakCode", ""),
                "runs_scored":   runs_scored,
                "runs_allowed":  runs_allowed,
                "rpg":           runs_scored / games if games > 0 else 4.5,
                "rapg":          runs_allowed / games if games > 0 else 4.5,
            }
    return records


# 팀 투구 통계

def get_team_pitching(team_id: int, season: int | None = None) -> dict:
    """ERA, WHIP, K/9, BB/9, 세이브 등"""
    if season is None:
        season = _current_season()

    data = _get(
        f"{MLB_API}/teams/{team_id}/stats",
        {"stats": "season", "group": "pitching", "season": season},
    )
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            st = s.get("stat", {})
            ip = _safe_float(st.get("inningsPitched"), 0)
            sv = int(st.get("saves", 0) or 0)
            bs = int(st.get("blownSaves", 0) or 0)
            return {
                "era":  _safe_float(st.get("era"),                4.50),
                "whip": _safe_float(st.get("whip"),               1.30),
                "k9":   _safe_float(st.get("strikeoutsPer9Inn"),  8.0),
                "bb9":  _safe_float(st.get("walksPer9Inn"),       3.0),
                "hr9":  _safe_float(st.get("homeRunsPer9",
                                            st.get("hrPer9")),    1.10),
                "saves":  sv,
                "blown":  bs,
                "sv_pct": sv / (sv + bs) if (sv + bs) > 0 else 0.5,
                "ip":     ip,
            }
    return {"era": 4.50, "whip": 1.30, "k9": 8.0, "bb9": 3.0,
            "hr9": 1.10, "saves": 0, "blown": 0, "sv_pct": 0.5, "ip": 0}


# 팀 타격 통계

def get_team_hitting(team_id: int, season: int | None = None) -> dict:
    """타율, OPS, OBP, SLG, 경기당 득점 등"""
    if season is None:
        season = _current_season()

    data = _get(
        f"{MLB_API}/teams/{team_id}/stats",
        {"stats": "season", "group": "hitting", "season": season},
    )
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            st = s.get("stat", {})
            gp   = int(st.get("gamesPlayed", 1) or 1)
            runs = int(st.get("runs", 0) or 0)
            return {
                "avg":  _parse_pct(st.get("avg"),  0.250),
                "obp":  _parse_pct(st.get("obp"),  0.320),
                "slg":  _parse_pct(st.get("slg"),  0.400),
                "ops":  _parse_pct(st.get("ops"),  0.720),
                "rpg":  runs / gp,
                "hr":   int(st.get("homeRuns", 0) or 0),
                "sb":   int(st.get("stolenBases", 0) or 0),
                "rbi":  int(st.get("rbi", 0) or 0),
                "games": gp,
                "runs":  runs,
            }
    return {"avg": 0.250, "obp": 0.320, "slg": 0.400, "ops": 0.720,
            "rpg": 4.5, "hr": 0, "sb": 0, "rbi": 0, "games": 1, "runs": 0}


# 선발 투수 정보

def get_probable_pitchers(game_pk: int, season: int | None = None) -> dict:
    """선발투수 이름 + 시즌 성적 반환"""
    if season is None:
        season = _current_season()

    data = _get(
        f"{MLB_API}/schedule",
        {"gamePk": game_pk, "hydrate": "probablePitcher(note)", "season": season},
    )

    result = {
        "home_pitcher_name":   None,
        "away_pitcher_name":   None,
        "home_pitcher_stats":  {},
        "away_pitcher_stats":  {},
        "home_pitcher_recent": {},   # 최근 5선발 ERA
        "away_pitcher_recent": {},
    }

    for date in data.get("dates", []):
        for game in date.get("games", []):
            teams = game.get("teams", {})
            for side in ("home", "away"):
                pp = teams.get(side, {}).get("probablePitcher")
                if not pp:
                    continue
                name = pp.get("fullName", "")
                pid  = pp.get("id")
                result[f"{side}_pitcher_name"] = name
                if pid:
                    result[f"{side}_pitcher_stats"]  = _get_pitcher_stats(pid, season)
                    result[f"{side}_pitcher_recent"] = _get_pitcher_recent_era(pid, season)

    return result


def _get_pitcher_stats(person_id: int, season: int) -> dict:
    data = _get(
        f"{MLB_API}/people/{person_id}/stats",
        {"stats": "season", "group": "pitching", "season": season},
    )
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            st = s.get("stat", {})
            return {
                "era":    _safe_float(st.get("era"),    4.50),
                "whip":   _safe_float(st.get("whip"),   1.30),
                "wins":   int(st.get("wins", 0) or 0),
                "losses": int(st.get("losses", 0) or 0),
                "k":      int(st.get("strikeOuts", 0) or 0),
                "gs":     int(st.get("gamesStarted", 0) or 0),
                "ip":     _safe_float(st.get("inningsPitched"), 0),
                "k9":     _safe_float(st.get("strikeoutsPer9Inn"), 8.0),
                "bb9":    _safe_float(st.get("walksPer9Inn"), 3.0),
            }
    return {"era": 4.50, "whip": 1.30, "wins": 0, "losses": 0,
            "k": 0, "gs": 0, "ip": 0, "k9": 8.0, "bb9": 3.0}


def _get_pitcher_recent_era(person_id: int, season: int, last_n: int = 5) -> dict:
    """
    선발투수 최근 last_n 선발 등판 ERA.
    gameLog API → 최근 순 정렬 → 선발(IP≥3) 경기만 집계.
    Returns: {"era": float|None, "games": int, "ip": float}
    """
    data = _get(
        f"{MLB_API}/people/{person_id}/stats",
        {"stats": "gameLog", "group": "pitching",
         "season": season, "gameType": "R"},
    )
    starts: list = []
    for sg in data.get("stats", []):
        for s in sg.get("splits", []):
            stat   = s.get("stat", {})
            ip_str = str(stat.get("inningsPitched", "0.0") or "0.0")
            try:
                parts = ip_str.split(".")
                ip    = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 else 0)
            except (ValueError, IndexError):
                ip = 0.0
            if ip < 3.0:          # 구원 출전 제외
                continue
            er = int(stat.get("earnedRuns", 0) or 0)
            starts.append((ip, er))

    # 최신 순서대로 들어오므로 앞에서 last_n개
    starts = starts[:last_n]
    if not starts:
        return {"era": None, "games": 0, "ip": 0.0}

    total_ip = sum(ip for ip, _ in starts)
    total_er = sum(er for _,  er in starts)
    if total_ip < 3.0:
        return {"era": None, "games": len(starts), "ip": round(total_ip, 1)}

    era = round((total_er * 9.0) / total_ip, 2)
    return {"era": era, "games": len(starts), "ip": round(total_ip, 1)}


# 상대 전적

def get_head_to_head(home_id: int, away_id: int, season: int | None = None) -> dict:
    """홈팀 기준 올 시즌 상대전적"""
    if season is None:
        season = _current_season()

    data = _get(
        f"{MLB_API}/schedule",
        {"sportId": 1, "season": season, "teamId": home_id,
         "opponentId": away_id, "gameType": "R"},
    )

    hw = aw = 0
    for date in data.get("dates", []):
        for game in date.get("games", []):
            if game.get("status", {}).get("abstractGameState", "") != "Final":
                continue
            teams = game.get("teams", {})
            h_id = teams.get("home", {}).get("team", {}).get("id")
            a_id = teams.get("away", {}).get("team", {}).get("id")
            hs   = teams.get("home", {}).get("score", 0) or 0
            as_  = teams.get("away", {}).get("score", 0) or 0

            if h_id == home_id:
                if hs > as_: hw += 1
                else: aw += 1
            elif a_id == home_id:
                if as_ > hs: hw += 1
                else: aw += 1

    total = hw + aw
    return {
        "home_wins": hw,
        "away_wins": aw,
        "total":     total,
        "home_pct":  hw / total if total > 0 else 0.500,
    }


# 라인업

def get_lineup(game_pk: int) -> dict:
    """
    예고 / 발표된 타순 반환.
    Returns: {"home": [...], "away": [...]}
    각 항목: {"order": 1~9, "name": "...", "pos": "CF"}
    미발표 시 빈 리스트.
    """
    data = _get(
        f"{MLB_API}/schedule",
        {"gamePk": game_pk, "hydrate": "lineups"},
    )
    result: dict = {"home": [], "away": []}
    for date in data.get("dates", []):
        for game in date.get("games", []):
            lineups = game.get("lineups", {})
            for side in ("home", "away"):
                players = lineups.get(f"{side}Players", [])
                starters = []
                for p in players:
                    try:
                        bo = int(str(p.get("battingOrder", 0)))
                    except (ValueError, TypeError):
                        continue
                    if bo > 0 and bo % 100 == 0:   # 100, 200 … 900 = 선발
                        starters.append({
                            "order": bo // 100,
                            "name":  p.get("fullName", ""),
                            "pos":   p.get("primaryPosition", {}).get("abbreviation", ""),
                        })
                starters.sort(key=lambda x: x["order"])
                result[side] = starters
    return result


# 전체 수집

def collect_pregame_data(game_pk: int, home_team_id: int, away_team_id: int,
                         season: int | None = None) -> dict:
    """
    경기 전 예측에 필요한 모든 데이터를 수집해 dict로 반환.
    실패한 항목은 빈 dict / 기본값으로 채워 항상 완전한 dict를 돌려준다.
    """
    if season is None:
        season = _current_season()

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  [team_stats] {fn.__name__} 오류: {e}")
            return {}

    team_records  = _safe(get_team_records, season)
    home_record   = team_records.get(home_team_id, {})
    away_record   = team_records.get(away_team_id, {})

    home_pitching = _safe(get_team_pitching, home_team_id, season)
    away_pitching = _safe(get_team_pitching, away_team_id, season)

    home_hitting  = _safe(get_team_hitting,  home_team_id, season)
    away_hitting  = _safe(get_team_hitting,  away_team_id, season)

    pitchers      = _safe(get_probable_pitchers, game_pk, season)
    h2h           = _safe(get_head_to_head, home_team_id, away_team_id, season)
    lineup        = _safe(get_lineup, game_pk)

    return {
        "home_record":   home_record,
        "away_record":   away_record,
        "home_pitching": home_pitching or {"era": 4.50, "whip": 1.30},
        "away_pitching": away_pitching or {"era": 4.50, "whip": 1.30},
        "home_hitting":  home_hitting  or {"rpg": 4.5, "ops": 0.720},
        "away_hitting":  away_hitting  or {"rpg": 4.5, "ops": 0.720},
        "pitchers":      pitchers      or {},
        "h2h":           h2h           or {"home_wins": 0, "away_wins": 0, "total": 0, "home_pct": 0.5},
        "lineup":        lineup        or {"home": [], "away": []},
        "season":        season,
    }
