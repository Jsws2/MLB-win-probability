"""
MLB 경기 전 승리 확률 예측 모델 (Pregame Prediction)

■ 구조: log-odds 가중 합산
  - 각 지표를 확률 p → logit(p) = log(p/(1-p)) 로 변환
  - 가중합한 뒤 sigmoid 로 복원
  - 장점: 중립 지표(p≈0.5)는 기여=0, 강한 지표는 비선형 증폭
         → 선형 평균보다 50:50 수렴이 적고 분산도 큼

■ 사용 지표 (가중치)
  시즌 승률          0.50  — 큰 표본, 팀 기본 실력
  Pythagorean 기대    0.50  — 득실 기반, 운 보정
  최근 10경기 폼      0.30  — 최근 상태
  홈/원정 성적        0.20  — 이 경기 컨텍스트
  팀 ERA 비교         0.20  — 투수력
  선발 시즌 ERA       0.40  — 당일 경기 최대 변수
  선발 최근 5경기 ERA  0.30  — 현재 컨디션 (가용 시에만)
  팀 OPS             0.15  — 타격 지표
  불펜 세이브율        0.10  — 마무리 안정성
  상대 전적           0.15  — 5경기 이상일 때만
  연속 기록           0.10  — 4연승/패 이상일 때만
"""

import math


# 수학 유틸

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _logit(p: float, eps: float = 0.005) -> float:
    """확률 → log-odds (안전 클램프)"""
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


def _ratio_prob(a: float, b: float, default: float = 0.5) -> float:
    total = a + b
    if total < 1e-6:
        return default
    return a / total


def _clamp(v: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, v))


# 예측기

class PregamePredictor:
    """
    collect_pregame_data() 반환 dict를 입력받아 홈팀 승리 확률 등을 반환.
    """

    def predict(self, data: dict) -> dict:
        hr   = data.get("home_record",   {})
        ar   = data.get("away_record",   {})
        hp   = data.get("home_pitching", {})
        ap   = data.get("away_pitching", {})
        hh   = data.get("home_hitting",  {})
        ah   = data.get("away_hitting",  {})
        pits = data.get("pitchers",      {})
        h2h  = data.get("h2h",           {})

        # 1. 시즌 승률
        home_wp = hr.get("win_pct", 0.500)
        away_wp = ar.get("win_pct", 0.500)
        p_season = _ratio_prob(home_wp, away_wp)

        # 2. 최근 10경기 폼
        hl10 = hr.get("last10_pct", home_wp)
        al10 = ar.get("last10_pct", away_wp)
        p_last10 = _ratio_prob(hl10, al10)

        # 3. 홈/원정 성적
        h_home_wp = hr.get("home_win_pct", 0.540)
        a_away_wp = ar.get("away_win_pct", 0.460)
        p_home_adv = _ratio_prob(h_home_wp, a_away_wp, 0.540)

        # 4. Pythagorean 기대 승률
        home_rpg = hh.get("rpg", 4.5)
        away_rpg = ah.get("rpg", 4.5)
        home_era = hp.get("era", 4.50)
        away_era = ap.get("era", 4.50)

        exp_home_rs = max(0.5, (home_rpg + away_era) / 2)
        exp_away_rs = max(0.5, (away_rpg + home_era) / 2)
        pyth_h = exp_home_rs ** 1.83
        pyth_a = exp_away_rs ** 1.83
        p_pyth = pyth_h / (pyth_h + pyth_a)

        # 5. 팀 ERA 비교 (감도 0.4 → 1점 차 ≈ 10% 영향)
        era_diff = away_era - home_era   # 양수 = 홈팀 유리
        p_era = _sigmoid(era_diff * 0.4)

        # 6. 선발투수 시즌 ERA
        hs_stats  = pits.get("home_pitcher_stats", {})
        as_stats  = pits.get("away_pitcher_stats", {})
        hs_era    = hs_stats.get("era", home_era)
        as_era    = as_stats.get("era", away_era)
        p_starter = _sigmoid((as_era - hs_era) * 0.45)

        # 7. 선발투수 최근 5경기 ERA (데이터 있을 때만)
        hs_recent = pits.get("home_pitcher_recent", {})
        as_recent = pits.get("away_pitcher_recent", {})
        hs_rec_era = hs_recent.get("era")   # None 가능
        as_rec_era = as_recent.get("era")

        p_starter_recent = None
        if hs_rec_era is not None and as_rec_era is not None:
            p_starter_recent = _sigmoid((as_rec_era - hs_rec_era) * 0.45)
        elif hs_rec_era is not None:
            # 홈팀 선발만: 시즌 ERA 대비 최근 폼 방향
            trend = hs_era - hs_rec_era   # 양수 = 최근 더 좋아짐
            p_starter_recent = _sigmoid(trend * 0.30)
        elif as_rec_era is not None:
            # 원정팀 선발만
            trend = as_rec_era - as_era   # 양수 = 최근 나빠짐 → 홈 유리
            p_starter_recent = _sigmoid(trend * 0.30)

        # 8. OPS
        home_ops = hh.get("ops", 0.720)
        away_ops = ah.get("ops", 0.720)
        p_ops = _ratio_prob(home_ops, away_ops)

        # 9. 불펜 세이브율
        h_svpct = hp.get("sv_pct", 0.5)
        a_svpct = ap.get("sv_pct", 0.5)
        p_bullpen = _ratio_prob(h_svpct + 0.01, a_svpct + 0.01)

        # 10. 상대 전적 (5경기 이상만 반영)
        h2h_total = h2h.get("total", 0)
        p_h2h = h2h.get("home_pct", 0.500) if h2h_total >= 5 else None

        # 11. 연속 기록 (4연승/패 이상만)
        hs_stk = hr.get("streak", "")
        as_stk = ar.get("streak", "")

        def _streak_n(s):
            return int(s[1:]) if s and len(s) > 1 and s[1:].isdigit() else 0

        hs_n = _streak_n(hs_stk); as_n = _streak_n(as_stk)
        hs_w = hs_stk.startswith("W");  as_w = as_stk.startswith("W")
        p_streak = None
        if max(hs_n, as_n) >= 4:
            # 홈팀 연승 or 원정팀 연패 → 홈 유리
            if hs_w and hs_n >= 4:
                p_streak = min(0.68, 0.5 + hs_n * 0.030)
            elif not hs_w and hs_n >= 4:
                p_streak = max(0.32, 0.5 - hs_n * 0.030)
            elif as_w and as_n >= 4:
                p_streak = max(0.32, 0.5 - as_n * 0.030)
            elif not as_w and as_n >= 4:
                p_streak = min(0.68, 0.5 + as_n * 0.030)

        # log-odds 가중 합산
        contributions = [
            (p_season,    0.50),
            (p_pyth,      0.50),
            (p_last10,    0.30),
            (p_home_adv,  0.20),
            (p_era,       0.20),
            (p_starter,   0.40),
            (p_ops,       0.15),
            (p_bullpen,   0.10),
        ]
        if p_starter_recent is not None:
            contributions.append((p_starter_recent, 0.30))
        if p_h2h is not None:
            contributions.append((p_h2h, 0.15))
        if p_streak is not None:
            contributions.append((p_streak, 0.10))

        log_odds_total = sum(w * _logit(p) for p, w in contributions)
        home_prob = _clamp(_sigmoid(log_odds_total), 0.05, 0.95)
        away_prob = 1.0 - home_prob

        # 예상 득점 / 안타
        exp_home_runs = round(exp_home_rs, 1)
        exp_away_runs = round(exp_away_rs, 1)
        _AB = 35
        exp_home_hits = round(hh.get("avg", 0.250) * _AB, 1)
        exp_away_hits = round(ah.get("avg", 0.250) * _AB, 1)

        factors = self._make_factors(
            data, home_era, away_era, hs_era, as_era,
            home_ops, away_ops, p_pyth,
            h_svpct, a_svpct,
            hs_recent, as_recent,
        )
        lineup = data.get("lineup", {"home": [], "away": []})

        raw = {
            "home_wp": home_wp, "away_wp": away_wp,
            "home_l10": hl10,   "away_l10": al10,
            "home_era": home_era, "away_era": away_era,
            "hs_era": hs_era,   "as_era": as_era,
            "hs_recent_era": hs_rec_era,
            "as_recent_era": as_rec_era,
            "home_ops": home_ops, "away_ops": away_ops,
            "exp_home_rs": round(exp_home_rs, 2),
            "exp_away_rs": round(exp_away_rs, 2),
            "h2h": h2h,
            "log_odds": round(log_odds_total, 3),
        }

        return {
            "home_win_pct":        home_prob,
            "away_win_pct":        away_prob,
            "factors":             factors,
            "home_starter":        pits.get("home_pitcher_name"),
            "away_starter":        pits.get("away_pitcher_name"),
            "home_starter_stats":  hs_stats,
            "away_starter_stats":  as_stats,
            "home_starter_recent": hs_recent,
            "away_starter_recent": as_recent,
            "exp_home_runs":       exp_home_runs,
            "exp_away_runs":       exp_away_runs,
            "exp_home_hits":       exp_home_hits,
            "exp_away_hits":       exp_away_hits,
            "lineup":              lineup,
            "raw":                 raw,
        }

    # 예측 근거 생성

    def _make_factors(self, data, home_era, away_era, hs_era, as_era,
                      home_ops, away_ops, p_pyth,
                      h_svpct, a_svpct,
                      hs_recent, as_recent) -> list:
        hr   = data.get("home_record",   {})
        ar   = data.get("away_record",   {})
        hp   = data.get("home_pitching", {})
        ap   = data.get("away_pitching", {})
        hh   = data.get("home_hitting",  {})
        ah   = data.get("away_hitting",  {})
        pits = data.get("pitchers",      {})
        h2h  = data.get("h2h",           {})

        home_wp = hr.get("win_pct", 0.500)
        away_wp = ar.get("win_pct", 0.500)
        hl10    = hr.get("last10_pct", home_wp)
        al10    = ar.get("last10_pct", away_wp)

        factors = []

        # 1. 시즌 승률
        factors.append({
            "name":       "시즌 승률",
            "home_edge":  home_wp - away_wp >  0.010,
            "away_edge":  home_wp - away_wp < -0.010,
            "detail":     f"홈 {home_wp:.3f}  vs  원정 {away_wp:.3f}",
            "importance": "높음",
        })

        # 2. 최근 10경기
        hl10w = hr.get("last10_wins",0); hl10l = hr.get("last10_losses",0)
        al10w = ar.get("last10_wins",0); al10l = ar.get("last10_losses",0)
        factors.append({
            "name":       "최근 10경기 폼",
            "home_edge":  hl10 - al10 >  0.05,
            "away_edge":  hl10 - al10 < -0.05,
            "detail":     f"홈 {hl10w}승{hl10l}패  vs  원정 {al10w}승{al10l}패",
            "importance": "높음",
        })

        # 3. 홈/원정 성적
        h_hwp = hr.get("home_win_pct", 0.5)
        a_awp = ar.get("away_win_pct", 0.5)
        hh_w = hr.get("home_wins",0); hh_l = hr.get("home_losses",0)
        aw_w = ar.get("away_wins",0); aw_l = ar.get("away_losses",0)
        factors.append({
            "name":       "홈/원정 성적",
            "home_edge":  h_hwp - a_awp >  0.03,
            "away_edge":  h_hwp - a_awp < -0.03,
            "detail":     f"홈팀 홈 {hh_w}승{hh_l}패({h_hwp:.3f})  /  원정팀 원정 {aw_w}승{aw_l}패({a_awp:.3f})",
            "importance": "중간",
        })

        # 4. Pythagorean
        h_rpg = hh.get("rpg", 4.5); a_rpg = ah.get("rpg", 4.5)
        factors.append({
            "name":       "득점력 (Pythagorean)",
            "home_edge":  p_pyth > 0.53,
            "away_edge":  p_pyth < 0.47,
            "detail":     f"홈 {h_rpg:.1f}점/경기  /  원정 {a_rpg:.1f}점/경기",
            "importance": "높음",
        })

        # 5. 팀 ERA
        factors.append({
            "name":       "팀 평균자책점(ERA)",
            "home_edge":  home_era - away_era < -0.20,
            "away_edge":  home_era - away_era >  0.20,
            "detail":     f"홈 {home_era:.2f}  vs  원정 {away_era:.2f}",
            "importance": "중간",
        })

        # 6. WHIP
        hw = hp.get("whip", 1.30); aw = ap.get("whip", 1.30)
        factors.append({
            "name":       "팀 WHIP",
            "home_edge":  hw < aw - 0.05,
            "away_edge":  hw > aw + 0.05,
            "detail":     f"홈 {hw:.2f}  vs  원정 {aw:.2f}",
            "importance": "중간",
        })

        # 7. OPS
        factors.append({
            "name":       "팀 OPS",
            "home_edge":  home_ops - away_ops >  0.010,
            "away_edge":  home_ops - away_ops < -0.010,
            "detail":     f"홈 {home_ops:.3f}  vs  원정 {away_ops:.3f}",
            "importance": "낮음",
        })

        # 8. 선발투수 시즌 ERA
        hn = pits.get("home_pitcher_name"); an = pits.get("away_pitcher_name")
        if hn and an:
            hs_w = hs_era; as_w2 = as_era
            hp_s = pits.get("home_pitcher_stats", {})
            ap_s = pits.get("away_pitcher_stats", {})
            hw_l = f"{hp_s.get('wins',0)}-{hp_s.get('losses',0)}" if hp_s else "─"
            aw_l = f"{ap_s.get('wins',0)}-{ap_s.get('losses',0)}" if ap_s else "─"
            factors.append({
                "name":       "선발투수 ERA (시즌)",
                "home_edge":  hs_era - as_era < -0.25,
                "away_edge":  hs_era - as_era >  0.25,
                "detail":     f"{hn} {hw_l} ERA {hs_era:.2f}  vs  {an} {aw_l} ERA {as_era:.2f}",
                "importance": "높음",
            })

        # 9. 선발투수 최근 5경기 ERA (데이터 있을 때)
        hs_re = hs_recent.get("era"); as_re = as_recent.get("era")
        if hs_re is not None or as_re is not None:
            hn_str = f"홈 최근{hs_recent.get('games',0)}경기 ERA {hs_re:.2f}" if hs_re is not None else "홈 ─"
            an_str = f"원정 최근{as_recent.get('games',0)}경기 ERA {as_re:.2f}" if as_re is not None else "원정 ─"
            home_edge = (hs_re is not None and as_re is not None and hs_re < as_re - 0.30)
            away_edge = (hs_re is not None and as_re is not None and hs_re > as_re + 0.30)
            if hs_re is None:
                away_edge = (as_re is not None and as_re < as_era - 0.30)
                home_edge = False
            if as_re is None:
                home_edge = (hs_re is not None and hs_re < hs_era - 0.30)
                away_edge = False
            factors.append({
                "name":       "선발 최근 5경기 폼",
                "home_edge":  home_edge,
                "away_edge":  away_edge,
                "detail":     f"{hn_str}  /  {an_str}",
                "importance": "높음",
            })

        # 10. 불펜
        if h_svpct > 0 or a_svpct > 0:
            hs_sv = hp.get("saves",0); hb = hp.get("blown",0)
            as_sv = ap.get("saves",0); ab = ap.get("blown",0)
            factors.append({
                "name":       "불펜 안정성",
                "home_edge":  h_svpct - a_svpct >  0.05,
                "away_edge":  h_svpct - a_svpct < -0.05,
                "detail":     f"홈 {hs_sv}세이브 {hb}블론  vs  원정 {as_sv}세이브 {ab}블론",
                "importance": "낮음",
            })

        # 11. 상대 전적 (5경기 이상)
        h2h_total = h2h.get("total", 0)
        if h2h_total >= 5:
            hw2 = h2h.get("home_wins",0); aw2 = h2h.get("away_wins",0)
            h2h_pct = h2h.get("home_pct", 0.5)
            factors.append({
                "name":       "올 시즌 상대 전적",
                "home_edge":  h2h_pct > 0.55,
                "away_edge":  h2h_pct < 0.45,
                "detail":     f"홈팀 {hw2}승 {aw2}패 ({h2h_total}경기)",
                "importance": "낮음",
            })

        # 12. 연속 기록 (4연승/패 이상)
        hs_stk = hr.get("streak", ""); as_stk = ar.get("streak", "")
        hs_n = int(hs_stk[1:]) if hs_stk and len(hs_stk)>1 and hs_stk[1:].isdigit() else 0
        as_n = int(as_stk[1:]) if as_stk and len(as_stk)>1 and as_stk[1:].isdigit() else 0
        if max(hs_n, as_n) >= 4:
            hw_e = (hs_stk.startswith("W") and hs_n>=4) or (as_stk.startswith("L") and as_n>=4)
            aw_e = (as_stk.startswith("W") and as_n>=4) or (hs_stk.startswith("L") and hs_n>=4)
            factors.append({
                "name":       "현재 연속 기록",
                "home_edge":  hw_e and not aw_e,
                "away_edge":  aw_e and not hw_e,
                "detail":     f"홈 {hs_stk or '─'}  vs  원정 {as_stk or '─'}",
                "importance": "낮음",
            })

        return factors
