"""경기 피드 → 머신러닝 특징(feature) 변환 모듈"""

import numpy as np
import pandas as pd

# 모델에 사용하는 특징 컬럼 (순서 고정)
FEATURE_COLS = [
    "inning",       # 이닝 (1~9+, 9로 cap)
    "is_top",       # 초=1, 말=0
    "outs",         # 아웃카운트 (0~2)
    "score_diff",   # 홈팀 점수 - 원정팀 점수 (-10~+10 clip)
    "runner_1b",    # 1루 주자 (0/1)
    "runner_2b",    # 2루 주자 (0/1)
    "runner_3b",    # 3루 주자 (0/1)
]


def _get_runner_state(runners):
    """
    타석 시작 시점의 주자 상태 파악.
    runners 리스트에서 originBase(출발지)가 있는 항목 = 기존 주자.
    """
    on_base = set()
    for r in runners:
        mov = r.get("movement", {})
        origin = mov.get("originBase")
        if origin in ("1B", "2B", "3B"):
            on_base.add(origin)
    return (
        1 if "1B" in on_base else 0,
        1 if "2B" in on_base else 0,
        1 if "3B" in on_base else 0,
    )


def extract_game_states(game_feed):
    """
    경기 피드 하나를 게임 상태 dict 리스트로 변환.
    각 dict = 타석 시작 시점의 상태 + 최종 결과(home_wins).
    """
    try:
        live_data = game_feed.get("liveData", {})
        game_data = game_feed.get("gameData", {})
        all_plays = live_data.get("plays", {}).get("allPlays", [])
        if not all_plays:
            return []

        linescore = live_data.get("linescore", {})
        final_home = linescore.get("teams", {}).get("home", {}).get("runs", 0) or 0
        final_away = linescore.get("teams", {}).get("away", {}).get("runs", 0) or 0

        abstract_state = game_data.get("status", {}).get("abstractGameState", "")
        print(f"[features] extract_game_states: abstract_state={abstract_state!r}, "
              f"linescore_home={final_home}, linescore_away={final_away}, "
              f"all_plays={len(all_plays)}")

        is_final = (abstract_state == "Final")

        if is_final and final_home == final_away:
            # 종료된 무승부 경기 -> 학습 라벨(home_wins) 산출 불가 -> 학습 데이터에서 제외
            print(f"[features] extract_game_states: 종료된 경기인데 동점({final_home}:{final_away}) "
                  f"-> 무승부 경기 -> 빈 리스트 반환 (학습 데이터 제외)")
            return []

        if not is_final and final_home == final_away:
            # LIVE/Preview 중 동점 -> 승자 라벨은 아직 알 수 없지만,
            # ML 예측(승률 계산)에는 home_wins 라벨이 필요 없으므로 states는 계속 생성한다.
            print(f"[features] LIVE 동점 경기지만 states 생성 계속 "
                  f"(abstract_state={abstract_state!r}, score={final_home}:{final_away})")

        # 종료 경기: 실제 승자. LIVE/Preview 동점: 라벨 미정(None) -> 학습에는 사용하지 말 것.
        if final_home == final_away:
            home_wins = None
        else:
            home_wins = 1 if final_home > final_away else 0
        home_team = game_data.get("teams", {}).get("home", {}).get("name", "Home")
        away_team = game_data.get("teams", {}).get("away", {}).get("name", "Away")

        home_score = 0
        away_score = 0
        states = []

        for play in all_plays:
            about = play.get("about", {})
            if not about.get("isComplete", False):
                continue

            count = play.get("count", {})
            inning_actual = about.get("inning", 1)          # 실제 이닝 (연장전 포함)
            inning = min(inning_actual, 9)                   # ML 모델 피처: 9 cap 유지
            is_top = 1 if about.get("isTopInning", True) else 0
            outs = count.get("outs", 0)

            runner_1b, runner_2b, runner_3b = _get_runner_state(play.get("runners", []))
            score_diff = int(np.clip(home_score - away_score, -10, 10))

            result = play.get("result", {})
            rbi = result.get("rbi", 0) or 0
            is_scoring_play = bool(result.get("isScoringPlay", False) or rbi)

            states.append({
                # ML 특징 (inning은 9 cap — 학습 데이터와 일치)
                "inning": inning,
                "is_top": is_top,
                "outs": outs,
                "score_diff": score_diff,
                "runner_1b": runner_1b,
                "runner_2b": runner_2b,
                "runner_3b": runner_3b,
                # 타겟
                "home_wins": home_wins,
                # 표시용 (실제 이닝 — 연장전 그대로)
                "display_inning": inning_actual,
                "home_score": home_score,
                "away_score": away_score,
                "home_team": home_team,
                "away_team": away_team,
                "event": result.get("event", ""),
                "description": result.get("description", ""),
                "batter": play.get("matchup", {}).get("batter", {}).get("fullName", ""),
                "pitcher": play.get("matchup", {}).get("pitcher", {}).get("fullName", ""),
                "at_bat_index": about.get("atBatIndex", len(states)),
                # 표시용 추가 정보 (볼카운트 / 득점 여부)
                "balls": count.get("balls", 0),
                "strikes": count.get("strikes", 0),
                "is_scoring_play": is_scoring_play,
                "rbi": rbi,
            })

            # 득점 반영 (다음 타석부터 적용)
            # rbi가 아닌 실제 득점한 주자 수(movement.end == "score") 기준으로 집계
            # (폭투/포일 등으로 타점 없이 득점하는 경우 rbi만으로는 누락됨)
            runners_scored = sum(
                1 for r in play.get("runners", [])
                if r.get("movement", {}).get("end") == "score"
            )
            if runners_scored:
                if is_top:
                    away_score += runners_scored
                else:
                    home_score += runners_scored

        # [디버그] FEATURE_COLS 결측/None/NaN 점검
        bad = []
        for i, s in enumerate(states):
            for c in FEATURE_COLS:
                v = s.get(c, "<MISSING>")
                if v == "<MISSING>" or v is None:
                    bad.append((i, c, v))
                else:
                    try:
                        if np.isnan(float(v)):
                            bad.append((i, c, v))
                    except (TypeError, ValueError):
                        bad.append((i, c, v))
        if bad:
            print(f"[features] extract_game_states: FEATURE_COLS 결측/None/NaN 발견 "
                  f"({len(bad)}건, 처음 5개): {bad[:5]}")
        else:
            print(f"[features] extract_game_states: FEATURE_COLS 결측/None/NaN 없음 "
                  f"(states={len(states)}개)")
        if states:
            print(f"[features] extract_game_states: 마지막 state의 FEATURE_COLS = "
                  f"{ {c: states[-1].get(c) for c in FEATURE_COLS} }")
        print(f"[features] len(states)={len(states)}")

        return states

    except Exception:
        import traceback
        print(f"[features] extract_game_states 예외 발생:\n{traceback.format_exc()}")
        return []


def build_training_df(game_feeds):
    """여러 경기 피드 → 학습용 DataFrame 생성"""
    all_states = []
    for feed in game_feeds:
        states = extract_game_states(feed)
        all_states.extend(states)

    if not all_states:
        return pd.DataFrame()

    return pd.DataFrame(all_states)
