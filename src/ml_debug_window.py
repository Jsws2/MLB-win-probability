"""
ML 모델 실시간 분석 창 (Toplevel) — 발표용 대시보드

타석(play-by-play)이 진행될 때마다
- 어떤 데이터(이닝/초말/아웃/주자/점수차/이벤트)가 모델에 입력되고
- predict_proba() 결과(홈팀 승률)가 어떻게 바뀌는지
- 어떤 feature가 그 변화에 가장 크게 기여했는지
를 큰 그래프 + 자연어 설명으로 한눈에 보여준다.

자동 재생뿐 아니라 버튼 / 키보드(←,→,Space,Home,End) / 그래프 클릭 / 진행 슬라이더로
원하는 타석을 자유롭게 이동하며 분석할 수 있다.

실제 경기 데이터(set_history)가 없으면 데모용 가상 타석 시퀀스로 미리보기.
"""

import random
import re
import tkinter as tk
from tkinter import simpledialog

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .features import FEATURE_COLS

# main.py와 동일한 다크 테마 색상
BG      = "#0f0f0f"
PANEL   = "#1a1a1a"
CARD    = "#222222"
CARD2   = "#2c2c2c"
DIMCARD = "#161616"
MINT    = "#00d4a0"
RED     = "#ff5555"
YELLOW  = "#ffb020"
ORANGE  = "#ff9f43"
BLUE    = "#4da6ff"
BORDER  = "#333333"

# 텍스트 계층 구조 (가독성 개선: 회색 위주 -> 밝은 회색/흰색 위주)
TITLE   = "#ffffff"   # 제목 (섹션 타이틀, 중요 라벨)
TEXT    = "#e0e0e0"   # 본문
MIDTEXT = "#b8b8b8"   # 보조 정보
SUBTEXT = "#a0a0a0"   # 설명 문구 (캡션, 안내 문구)
AXIS_TEXT = "#c8c8c8"  # 그래프 축 눈금 숫자 (기존보다 20~30% 밝게)

FEATURE_LABELS_KR = {
    "inning":     "이닝",
    "is_top":     "초/말",
    "outs":       "아웃카운트",
    "score_diff": "점수차(홈-원정)",
    "runner_1b":  "1루 주자",
    "runner_2b":  "2루 주자",
    "runner_3b":  "3루 주자",
}

# [모델 중요 변수] 패널용 짧은 라벨
FEATURE_LABELS_SHORT = {
    "inning":     "이닝",
    "is_top":     "공격/수비",
    "outs":       "아웃카운트",
    "score_diff": "점수차",
    "runner_1b":  "1루 주자",
    "runner_2b":  "2루 주자",
    "runner_3b":  "3루 주자",
}

TOP_N_IMPORTANCE = 5  # [모델 중요 변수] 패널에 표시할 feature 수

RECENT_N      = 20  # "최근 N개 타석" 보기 모드의 N
TOP_N_CONTRIB = 5   # 기여도 그래프에 표시할 feature 수
BIG_SWING     = 10  # 승률 급변 기준 (%p)
TOP_N_HIGHLIGHT = 5  # 주요 장면 카드 개수

# MLB 이벤트(영문) → 한글 결과 번역
EVENT_KR = {
    "Single": "안타",
    "Double": "2루타",
    "Triple": "3루타",
    "Home Run": "홈런",
    "Walk": "볼넷",
    "Intent Walk": "고의 4구",
    "Strikeout": "삼진",
    "Strikeout Double Play": "삼진 병살",
    "Groundout": "땅볼 아웃",
    "Flyout": "뜬공 아웃",
    "Pop Out": "내야 뜬공 아웃",
    "Lineout": "직선타 아웃",
    "Forceout": "포스 아웃",
    "Field Error": "야수 실책",
    "Fielders Choice": "필더스 초이스",
    "Fielders Choice Out": "필더스 초이스 아웃",
    "Double Play": "병살타",
    "Triple Play": "삼중살",
    "Sac Fly": "희생 플라이",
    "Sac Bunt": "희생 번트",
    "Sac Fly Double Play": "희생 플라이 병살",
    "Hit By Pitch": "몸에 맞는 볼",
    "Bunt Groundout": "번트 땅볼 아웃",
    "Bunt Pop Out": "번트 뜬공 아웃",
    "Wild Pitch": "폭투",
    "Passed Ball": "포일",
    "Balk": "보크",
    "Catcher Interference": "포수 방해",
    "Runner Out": "주자 아웃",
    "Pickoff": "견제사",
    "Stolen Base 2B": "2루 도루",
    "Stolen Base 3B": "3루 도루",
    "Stolen Base Home": "홈 도루",
    "Caught Stealing 2B": "2루 도루 실패",
    "Caught Stealing 3B": "3루 도루 실패",
    "Caught Stealing Home": "홈 도루 실패",
    "Pickoff Caught Stealing 2B": "견제 도루사",
    "Game Advisory": "경기 안내",
}

# 수비 위치(영문) → 한글
_FIELD_POS_KR = {
    "left field": "좌익수",
    "left center field": "좌중간",
    "center field": "중견수",
    "right center field": "우중간",
    "right field": "우익수",
    "shortstop": "유격수",
    "third base": "3루수",
    "second base": "2루수",
    "first base": "1루수",
    "pitcher": "투수",
    "catcher": "포수",
}

_BATTED_BALL_KR = {
    "fly ball": "뜬공",
    "ground ball": "땅볼",
    "line drive": "직선타",
    "pop up": "팝업",
    "bunt": "번트",
}


def _has_batchim(word):
    """한글 단어 마지막 글자에 받침이 있는지 여부 (조사 선택용)"""
    if not word:
        return False
    last = word[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _josa(word, pair):
    """받침 유무에 따라 조사 선택. pair 예: '을/를', '이/가', '은/는', '와/과'"""
    with_batchim, without_batchim = pair.split("/")
    return with_batchim if _has_batchim(word) else without_batchim


def _event_kr(state):
    """이벤트(영문) -> 한글 결과 명칭"""
    event = state.get("event", "")
    return EVENT_KR.get(event, event or "타석 진행")


def _translate_situation(state):
    """타구 설명(영문)에서 핵심 상황만 간단한 한글로 번역 (best-effort)"""
    desc = state.get("description") or ""
    m = re.search(
        r"on a (fly ball|ground ball|line drive|pop up|bunt) to ([a-z ]+?)(\.|,|$)",
        desc, re.IGNORECASE,
    )
    if m:
        kind = _BATTED_BALL_KR.get(m.group(1).lower(), m.group(1))
        pos = _FIELD_POS_KR.get(m.group(2).strip().lower(), m.group(2).strip())
        return f"{pos} 방향 {kind}"
    return ""


def _attack_defense(state):
    """현재 타석의 (공격팀, 수비팀) 이름 반환"""
    if state.get("is_top"):
        return state.get("away_team", "원정팀"), state.get("home_team", "홈팀")
    return state.get("home_team", "홈팀"), state.get("away_team", "원정팀")


def _inning_half_kr(state):
    disp_inning = state.get("display_inning", state.get("inning", 1))
    half = "초" if state.get("is_top") else "말"
    return f"{disp_inning}회 {half}"


def _runners_text(state):
    bases = []
    if state.get("runner_1b"):
        bases.append("1루")
    if state.get("runner_2b"):
        bases.append("2루")
    if state.get("runner_3b"):
        bases.append("3루")
    return ", ".join(bases) if bases else "없음"


def _scored_runs(idx, all_states):
    """idx(1-indexed) 타석의 결과로 발생한 득점 수 (rbi 기준)"""
    state = all_states[idx - 1]
    if state.get("is_scoring_play"):
        return state.get("rbi", 0) or 0
    return 0


# 모델 입력 feature -> 일반인용 설명
def _feature_friendly_line(feat, raw_val, contrib_val, state):
    """특징별 기여도 그래프 옆에 표시할 일반인용 설명 한 줄 생성"""
    up = "승률 상승 요인" if contrib_val >= 0 else "승률 하락 요인"
    arrow = "↑" if contrib_val >= 0 else "↓"

    if feat == "score_diff":
        v = int(raw_val)
        if v > 0:
            base = f"점수차: 홈팀이 {v}점 앞섬"
        elif v < 0:
            base = f"점수차: 홈팀이 {abs(v)}점 뒤짐"
        else:
            base = "점수차: 동점"
    elif feat == "is_top":
        base = "원정팀 공격 중" if int(raw_val) == 1 else "홈팀 공격 중"
    elif feat == "outs":
        base = f"{int(raw_val)}아웃 상황"
    elif feat == "inning":
        base = f"{int(raw_val)}회 진행 중"
    elif feat in ("runner_1b", "runner_2b", "runner_3b"):
        base_map = {"runner_1b": "1루", "runner_2b": "2루", "runner_3b": "3루"}
        loc = base_map[feat]
        base = f"{loc} 주자 {'있음' if raw_val else '없음'}"
    else:
        base = FEATURE_LABELS_KR.get(feat, feat)

    return f"{base} → {up} {arrow}"


class MLDebugWindow(tk.Toplevel):
    """ML 모델 입력 특징 / 기여도 / 승률 변화를 보여주는 발표용 대시보드 창"""

    def __init__(self, master, model, on_close=None):
        super().__init__(master)
        self.title("🧠 ML 모델 실시간 분석 대시보드")
        self.geometry("1100x900")
        self.minsize(1000, 760)
        self.configure(bg=BG)

        self.model = model
        self._on_close = on_close

        # 전체 데이터 (play-by-play)
        self.all_states = []
        self.all_probs  = []
        self.highlights = []   # [(idx, delta_pct), ...] 승률 변화 TOP N

        # 재생 상태
        self.play_idx   = 0          # 현재까지 표시 중인 타석 수 (1-indexed)
        self.playing    = False
        self.anim_job   = None
        self.speed_ms   = 500
        self.view_var   = tk.StringVar(value="전체")

        self._is_demo   = False
        self._prob_annot = None

        # 발표용 모드 / 레이아웃 상태
        self.compact = False
        self.contrib_visible = True
        self._prob_fig_h = 3.4   # 승률 그래프 기본 높이(인치)
        self._resize_job = None
        self._is_fullscreen = False
        self._linescore = None

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._build_ui()
        self._bind_keys()

        # 실제 데이터가 없으면 데모용 가상 데이터로 미리보기
        self._load_demo_if_empty()
        self._compute_highlights()
        self._render()

        self.bind("<Configure>", self._on_window_configure)
        self.after(150, self._apply_auto_graph_height)
        self.after(100, self.focus_force)

    # UI 구성
    def _build_ui(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        # ════════════ 전체 세로 스크롤 영역 ════════════
        self._scroll_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=vsb.set)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        scroll_host = tk.Frame(self._scroll_canvas, bg=BG)
        self._content_window = self._scroll_canvas.create_window((0, 0), window=scroll_host, anchor="nw")

        def _on_content_configure(event):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            self._scroll_canvas.itemconfig(self._content_window, width=event.width)

        scroll_host.bind("<Configure>", _on_content_configure)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)

        top = tk.Frame(scroll_host, bg=BG)
        top.pack(fill="both", expand=True, padx=18, pady=(14, 16))

        # [경기명]
        self.title_lbl = tk.Label(top, text="경기를 선택하세요", bg=BG, fg=TITLE,
                                   font=("Arial", 18, "bold"))
        self.title_lbl.pack(pady=(0, 2))
        self.subtitle_lbl = tk.Label(
            top, text="🧠 ML 모델 실시간 분석 — 경기 분석을 시작하면 갱신됩니다.",
            bg=BG, fg=SUBTEXT, font=("Arial", 11))
        self.subtitle_lbl.pack(pady=(0, 10))

        # [머신러닝 모델 정보] / [모델 중요 변수 TOP N] — 기본 숨김, 버튼으로 펼치기
        self.btn_model_info = self._mkbtn(top, "🤖 모델 설명 보기", self._toggle_model_info)
        self.btn_model_info.pack(anchor="w", pady=(0, 10))

        self.model_info_container = tk.Frame(top, bg=BG)
        self._build_model_info_card(self.model_info_container)
        self._build_feature_importance_card(self.model_info_container)
        self._model_info_visible = False

        # [원정팀 / 홈팀 카드]
        team_row = tk.Frame(top, bg=BG)
        self.team_row = team_row
        team_row.pack(fill="x", pady=(0, 6))
        team_row.columnconfigure((0, 1), weight=1, uniform="team")

        self.away_card, self.away_widgets = self._make_team_card(team_row, "AWAY", RED, 0)
        self.home_card, self.home_widgets = self._make_team_card(team_row, "HOME", MINT, 1)

        meta_row = tk.Frame(top, bg=BG)
        meta_row.pack(fill="x", pady=(2, 12))
        self.attack_lbl = tk.Label(meta_row, text="현재 공격: -", bg=BG, fg=YELLOW,
                                    font=("Arial", 11, "bold"))
        self.attack_lbl.pack(side="left")
        self.scorediff_lbl = tk.Label(meta_row, text="홈팀 기준 점수차: -", bg=BG, fg=MIDTEXT,
                                       font=("Arial", 11))
        self.scorediff_lbl.pack(side="right")

        # [이닝별 득점표 (R H E)]
        self.linescore_canvas = tk.Canvas(top, bg=CARD, highlightthickness=0, height=70)
        self.linescore_canvas.pack(fill="x", pady=(0, 12))

        # [현재 홈팀 승률] + [현재 경기 상황 카드]
        info_top_row = tk.Frame(top, bg=BG)
        info_top_row.pack(fill="x", pady=(0, 12))

        prob_card = tk.Frame(info_top_row, bg=CARD)
        prob_card.pack(side="left", fill="both", padx=(0, 6))
        tk.Label(prob_card, text="현재 홈팀 승률", bg=CARD, fg=TITLE,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=18, pady=(14, 0))
        self.prob_lbl = tk.Label(prob_card, text="─.─%", bg=CARD, fg=MINT,
                                  font=("Arial", 40, "bold"))
        self.prob_lbl.pack(anchor="w", padx=18, pady=(0, 8))

        # 이전 승률 / 승률 변화 — 정보를 분리해서 표시
        delta_row = tk.Frame(prob_card, bg=CARD)
        delta_row.pack(anchor="w", fill="x", padx=18, pady=(0, 14))

        prev_col = tk.Frame(delta_row, bg=CARD)
        prev_col.pack(side="left", padx=(0, 32))
        tk.Label(prev_col, text="이전 승률", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 9)).pack(anchor="w")
        self.prev_prob_lbl = tk.Label(prev_col, text="─.─%", bg=CARD, fg=MIDTEXT,
                                       font=("Arial", 15, "bold"))
        self.prev_prob_lbl.pack(anchor="w")

        delta_col = tk.Frame(delta_row, bg=CARD)
        delta_col.pack(side="left")
        tk.Label(delta_col, text="승률 변화", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 9)).pack(anchor="w")
        self.delta_lbl = tk.Label(delta_col, text="", bg=CARD, fg=MIDTEXT,
                                   font=("Arial", 15, "bold"))
        self.delta_lbl.pack(anchor="w")

        state_card = tk.Frame(info_top_row, bg=CARD)
        state_card.pack(side="left", fill="both", expand=True, padx=(6, 0))
        tk.Label(state_card, text="현재 경기 상황", bg=CARD, fg=TITLE,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=18, pady=(14, 2))
        # 항목별로 색을 다르게 강조하기 위해 Text + 태그 사용
        self.state_lbl = tk.Text(
            state_card, bg=CARD, fg=TEXT, font=("Arial", 12),
            wrap="none", height=6, bd=0, highlightthickness=0,
            relief="flat", padx=0, pady=0, cursor="arrow", takefocus=0)
        self.state_lbl.tag_configure("idx", foreground=TITLE, font=("Arial", 12, "bold"))
        self.state_lbl.tag_configure("inning", foreground=MINT, font=("Arial", 12, "bold"))
        self.state_lbl.tag_configure("team", foreground=TITLE)
        self.state_lbl.tag_configure("label", foreground=MIDTEXT)
        self.state_lbl.tag_configure("name", foreground=YELLOW, font=("Arial", 12, "bold"))
        self.state_lbl.tag_configure("body", foreground=TEXT)
        self.state_lbl.insert("1.0", "대기 중...", "body")
        self.state_lbl.config(state="disabled")
        self.state_lbl.pack(anchor="w", fill="x", padx=18, pady=(0, 14))

        # [대형 승률 변화 그래프]
        graph_hdr = tk.Frame(top, bg=BG)
        graph_hdr.pack(fill="x")
        tk.Label(graph_hdr, text="📈 홈팀 승률 변화", bg=BG, fg=TITLE,
                 font=("Arial", 13, "bold")).pack(side="left", pady=(0, 4))
        tk.Label(graph_hdr, text="(점 클릭 = 해당 타석 이동, 마우스오버 = 상세 정보)",
                 bg=BG, fg=MIDTEXT, font=("Arial", 9)).pack(side="left", padx=(8, 0))

        self.fig_prob = Figure(figsize=(8.4, self._prob_fig_h), facecolor=BG, dpi=100)
        self.ax_prob = self.fig_prob.add_subplot(111)
        self.fig_prob.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.14)
        self.canvas_prob = FigureCanvasTkAgg(self.fig_prob, master=top)
        self.canvas_prob.get_tk_widget().configure(height=int(self._prob_fig_h * 100))
        self.canvas_prob.get_tk_widget().pack(fill="x", pady=(0, 4))
        self.canvas_prob.mpl_connect("button_press_event", self._on_prob_click)
        self.canvas_prob.mpl_connect("motion_notify_event", self._on_prob_hover)

        # 표시 범위 라디오
        view_row = tk.Frame(top, bg=BG)
        view_row.pack(fill="x", pady=(0, 4))
        tk.Label(view_row, text="표시 범위", bg=BG, fg=SUBTEXT, font=("Arial", 10)).pack(side="left")
        for label in ("전체", f"최근 {RECENT_N}개"):
            tk.Radiobutton(
                view_row, text=label, value=label, variable=self.view_var,
                command=self._render, bg=BG, fg=MIDTEXT,
                selectcolor=CARD2, activebackground=BG, activeforeground=TEXT,
                font=("Arial", 10), highlightthickness=0, takefocus=0,
            ).pack(side="left", padx=(6, 0))

        # 진행 슬라이더
        slider_row = tk.Frame(top, bg=BG)
        slider_row.pack(fill="x", pady=(0, 8))
        tk.Label(slider_row, text="타석 진행", bg=BG, fg=SUBTEXT, font=("Arial", 10)).pack(side="left")
        self.progress_scale = tk.Scale(
            slider_row, from_=1, to=1, orient="horizontal",
            bg=BG, fg=SUBTEXT, troughcolor=CARD2,
            highlightthickness=0, bd=0, showvalue=False, takefocus=0,
            command=self._on_slider_move,
        )
        self.progress_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.progress_lbl = tk.Label(slider_row, text="- / -", bg=BG, fg=SUBTEXT,
                                      font=("Arial", 10), width=10, anchor="e")
        self.progress_lbl.pack(side="right")

        # 재생 컨트롤 (한 줄)
        ctrl = tk.Frame(top, bg=BG)
        ctrl.pack(fill="x", pady=(0, 10))

        self._mkbtn(ctrl, "⏮ 처음", self._on_restart).pack(side="left", padx=(0, 4))
        self._mkbtn(ctrl, "◀ 이전", self._on_prev).pack(side="left", padx=4)
        self.btn_play = self._mkbtn(ctrl, "▶ 재생", self._on_toggle_play, fg=MINT)
        self.btn_play.pack(side="left", padx=4)
        self._mkbtn(ctrl, "다음 ▶", self._on_next).pack(side="left", padx=4)
        self._mkbtn(ctrl, "마지막 ⏭", self._on_last).pack(side="left", padx=4)
        self._mkbtn(ctrl, "🔁 다시보기", self._on_replay).pack(side="left", padx=4)
        self.btn_compact = self._mkbtn(ctrl, "🗜 컴팩트 모드", self._toggle_compact)
        self.btn_compact.pack(side="left", padx=(12, 4))

        tk.Label(ctrl, text="속도", bg=BG, fg=SUBTEXT, font=("Arial", 9)).pack(side="left", padx=(14, 2))
        self.speed_scale = tk.Scale(
            ctrl, from_=1500, to=80, orient="horizontal",
            bg=BG, fg=SUBTEXT, troughcolor=CARD2,
            highlightthickness=0, bd=0, length=140, showvalue=False, takefocus=0,
            command=lambda v: setattr(self, "speed_ms", int(float(v))),
        )
        self.speed_scale.set(self.speed_ms)
        self.speed_scale.pack(side="left")

        # 편의 기능 (검색 / 주요 장면 바로가기 / 도움말 / 전체화면)
        conv = tk.Frame(top, bg=BG)
        conv.pack(fill="x", pady=(0, 10))

        tk.Label(conv, text="타석 검색", bg=BG, fg=SUBTEXT, font=("Arial", 9)).pack(side="left", padx=(0, 4))
        self.search_entry = tk.Entry(
            conv, width=6, bg=CARD2, fg=TEXT, insertbackground=TEXT,
            relief="flat", justify="center")
        self.search_entry.pack(side="left", padx=(0, 4))
        self.search_entry.bind("<Return>", lambda e: self._on_search_jump())
        self._mkbtn(conv, "이동", self._on_search_jump).pack(side="left", padx=(0, 12))

        self._mkbtn(conv, "📈 최고 상승 장면", self._jump_to_best_rise, fg=MINT).pack(side="left", padx=4)
        self._mkbtn(conv, "📉 최고 하락 장면", self._jump_to_best_drop, fg=RED).pack(side="left", padx=4)
        self._mkbtn(conv, "❓ 단축키 도움말 (H)", self._show_help, fg=BLUE).pack(side="left", padx=(12, 4))
        self.btn_fullscreen = self._mkbtn(conv, "⛶ 전체화면 (F11)", self._toggle_fullscreen)
        self.btn_fullscreen.pack(side="left", padx=4)
        self.btn_debug = self._mkbtn(conv, "🐞 디버그 정보 (D)", self._toggle_debug, fg=ORANGE)
        self.btn_debug.pack(side="left", padx=4)

        # [중계형 요약 카드]
        self.summary_card = tk.Frame(top, bg=PANEL)
        self.summary_card.pack(fill="x", pady=(4, 12))

        self.summary_title_lbl = tk.Label(
            self.summary_card, text="-", bg=PANEL, fg=TITLE,
            font=("Arial", 16, "bold"), justify="left", anchor="w")
        self.summary_title_lbl.pack(anchor="w", padx=16, pady=(12, 2))

        self.summary_inning_lbl = tk.Label(
            self.summary_card, text="-", bg=PANEL, fg=YELLOW,
            font=("Arial", 11, "bold"), justify="left", anchor="w")
        self.summary_inning_lbl.pack(anchor="w", padx=16)

        self.summary_desc_lbl = tk.Label(
            self.summary_card, text="-", bg=PANEL, fg=TEXT,
            font=("Arial", 12), justify="left", anchor="w", wraplength=1000)
        self.summary_desc_lbl.pack(anchor="w", padx=16, pady=(4, 8))

        prob_change_row = tk.Frame(self.summary_card, bg=PANEL)
        prob_change_row.pack(anchor="w", padx=16, pady=(0, 4))
        tk.Label(prob_change_row, text="홈팀 승률", bg=PANEL, fg=MIDTEXT,
                 font=("Arial", 10, "bold")).pack(side="left", padx=(0, 8))
        self.summary_prob_lbl = tk.Label(
            prob_change_row, text="- → -", bg=PANEL, fg=TEXT,
            font=("Arial", 14, "bold"))
        self.summary_prob_lbl.pack(side="left")
        self.summary_delta_lbl = tk.Label(
            prob_change_row, text="", bg=PANEL, fg=SUBTEXT,
            font=("Arial", 12, "bold"))
        self.summary_delta_lbl.pack(side="left", padx=(8, 0))

        self.summary_orig_lbl = tk.Label(
            self.summary_card, text="", bg=PANEL, fg=SUBTEXT,
            font=("Arial", 9), justify="left", anchor="w", wraplength=1000)
        self.summary_orig_lbl.pack(anchor="w", padx=16, pady=(0, 12))

        # [AI 분석 요약]
        self.ai_summary_card = tk.Frame(top, bg=CARD)
        self.ai_summary_card.pack(fill="x", pady=(0, 12))
        tk.Label(self.ai_summary_card, text="🤖 AI 분석 요약", bg=CARD, fg=TITLE,
                 font=("Arial", 12, "bold")).pack(anchor="w", padx=18, pady=(14, 6))
        self.ai_summary_lbl = tk.Label(
            self.ai_summary_card, text="-", bg=CARD, fg=TEXT, font=("Arial", 11),
            justify="left", anchor="w", wraplength=1000)
        self.ai_summary_lbl.pack(anchor="w", padx=18, pady=(0, 14))

        # [이벤트 설명] [승률 변화 이유] [모델 판단 근거]
        self.info_row = tk.Frame(top, bg=BG)
        self.info_row.pack(fill="x", pady=(0, 12))
        self.info_row.columnconfigure((0, 1, 2), weight=1, uniform="info")

        self.event_card = self._make_info_card(self.info_row, "📍 이번 타석 결과", 0)
        self.reason_card = self._make_info_card(self.info_row, "🔍 승률 변화 이유", 1)
        self.basis_card = self._make_info_card(self.info_row, "🧠 모델이 주목한 부분", 2)

        # [현재 경기 상황 카드] (모델 입력 feature → 일반인용)
        feat_card = tk.Frame(top, bg=CARD)
        feat_card.pack(fill="x", pady=(0, 12))
        feat_hdr = tk.Frame(feat_card, bg=CARD)
        feat_hdr.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(feat_hdr, text="🧮 현재 경기 상황 (모델이 보는 정보)", bg=CARD, fg=TITLE,
                 font=("Arial", 10, "bold")).pack(side="left")
        self.btn_feat_detail = self._mkbtn(feat_hdr, "🔧 상세 보기", self._toggle_feature_detail)
        self.btn_feat_detail.pack(side="right")
        self.feature_lbl = tk.Label(
            feat_card, text="-", bg=CARD, fg=TEXT, font=("Arial", 11),
            justify="left", anchor="w")
        self.feature_lbl.pack(anchor="w", padx=14, pady=(0, 8))
        self.feature_detail_lbl = tk.Label(
            feat_card, text="-", bg=DIMCARD, fg=MIDTEXT, font=("Consolas", 10),
            justify="left", anchor="w")
        self.feature_detail_visible = False

        # [디버그 정보] (승률 변화 원인 분석용, 기본 숨김)
        self.debug_card = tk.Frame(top, bg=DIMCARD)
        debug_hdr = tk.Frame(self.debug_card, bg=DIMCARD)
        debug_hdr.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(debug_hdr, text="🐞 디버그 정보 — 승률 변화 원인 분석", bg=DIMCARD, fg=ORANGE,
                 font=("Arial", 10, "bold")).pack(side="left")
        self.debug_lbl = tk.Label(
            self.debug_card, text="-", bg=DIMCARD, fg=MIDTEXT, font=("Consolas", 10),
            justify="left", anchor="w")
        self.debug_lbl.pack(anchor="w", fill="x", padx=14, pady=(0, 12))
        self.debug_visible = False

        # [특징별 기여도 그래프 — 상위 5개] (접기/펼치기 가능)
        self.contrib_header = tk.Frame(top, bg=BG)
        self.contrib_header.pack(fill="x", pady=(0, 4))
        tk.Label(self.contrib_header, text=f"📊 승률에 영향을 준 요인 (상위 {TOP_N_CONTRIB}개)", bg=BG, fg=TITLE,
                 font=("Arial", 13, "bold")).pack(side="left")
        self.btn_contrib_toggle = self._mkbtn(self.contrib_header, "▼ 접기", self._toggle_contrib)
        self.btn_contrib_toggle.pack(side="right")

        self.contrib_container = tk.Frame(top, bg=BG)
        self.contrib_container.pack(fill="x", pady=(0, 16))

        self.fig_contrib = Figure(figsize=(8.4, 3.2), facecolor=BG, dpi=100)
        self.ax_contrib = self.fig_contrib.add_subplot(111)
        self.fig_contrib.subplots_adjust(left=0.30, right=0.97, top=0.82, bottom=0.22)
        self.canvas_contrib = FigureCanvasTkAgg(self.fig_contrib, master=self.contrib_container)
        self.canvas_contrib.get_tk_widget().pack(fill="x")
        self.contrib_explain_lbl = tk.Label(
            self.contrib_container, text="-", bg=BG, fg=MIDTEXT, font=("Arial", 10),
            justify="left", anchor="w")
        self.contrib_explain_lbl.pack(fill="x", anchor="w", pady=(4, 0))

        # [경기 MVP] (경기 종료 시에만 표시)
        self.mvp_card = tk.Frame(top, bg=CARD2)
        self.mvp_lbl = tk.Label(
            self.mvp_card, text="-", bg=CARD2, fg=YELLOW, font=("Arial", 13, "bold"),
            justify="left", anchor="w", padx=16, pady=12)
        self.mvp_lbl.pack(fill="x")
        self.mvp_visible = False

        # [주요 장면 TOP N]
        self.highlight_title_lbl = tk.Label(
            top, text=f"🔥 경기 명장면 TOP {TOP_N_HIGHLIGHT}", bg=BG, fg=TITLE,
            font=("Arial", 13, "bold"))
        self.highlight_title_lbl.pack(anchor="w", pady=(0, 4))
        self.highlight_frame = tk.Frame(top, bg=BG)
        self.highlight_frame.pack(fill="x", pady=(0, 20))

    def _make_team_card(self, parent, badge_text, badge_color, col):
        card = tk.Frame(parent, bg=CARD2)
        card.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6, 0))

        head = tk.Frame(card, bg=CARD2)
        head.pack(fill="x", padx=14, pady=(10, 0), anchor="w")
        badge = tk.Label(head, text=f" {badge_text} ", bg=badge_color, fg="#0a0a0a",
                          font=("Arial", 9, "bold"))
        badge.pack(side="left")

        body = tk.Frame(card, bg=CARD2)
        body.pack(fill="x", padx=14, pady=(4, 12))
        name_lbl = tk.Label(body, text="-", bg=CARD2, fg=TITLE, font=("Arial", 13, "bold"),
                             anchor="w")
        name_lbl.pack(side="left")
        score_lbl = tk.Label(body, text="0", bg=CARD2, fg=TITLE, font=("Arial", 22, "bold"))
        score_lbl.pack(side="right")

        widgets = {"badge": badge, "name": name_lbl, "score": score_lbl, "body": body, "head": head}
        return card, widgets

    def _make_info_card(self, parent, title, col):
        card = tk.Frame(parent, bg=CARD)
        card.grid(row=0, column=col, sticky="nsew", padx=4)
        tk.Label(card, text=title, bg=CARD, fg=TITLE,
                 font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
        body = tk.Label(card, text="─", bg=CARD, fg=TEXT, font=("Arial", 10),
                         justify="left", anchor="nw", wraplength=250)
        body.pack(anchor="w", fill="both", expand=True, padx=12, pady=(0, 10))
        card.body = body
        return card

    def _get_model_type_name(self):
        """현재 연결된 모델의 분류기 종류명을 반환 (모델 미연결 시 기본값)"""
        try:
            clf = self.model.pipeline.named_steps["clf"]
            return type(clf).__name__
        except Exception:
            return "(모델 미연결)"

    def _build_model_info_card(self, parent):
        """[머신러닝 모델 정보] — 승률이 어떻게 계산되는지 사용자에게 설명하는 카드"""
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x", pady=(0, 12))

        tk.Label(card, text="🤖 머신러닝 모델 정보", bg=CARD, fg=TITLE,
                 font=("Arial", 12, "bold")).pack(anchor="w", padx=18, pady=(14, 6))

        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=18, pady=(0, 4))

        # 모델 종류
        type_row = tk.Frame(body, bg=CARD)
        type_row.pack(anchor="w", fill="x", pady=(0, 2))
        tk.Label(type_row, text="모델 종류  ", bg=CARD, fg=MIDTEXT,
                 font=("Arial", 10, "bold")).pack(side="left")
        tk.Label(type_row, text=self._get_model_type_name(), bg=CARD, fg=TEXT,
                 font=("Arial", 10)).pack(side="left")

        # 예측 대상
        target_row = tk.Frame(body, bg=CARD)
        target_row.pack(anchor="w", fill="x", pady=(0, 8))
        tk.Label(target_row, text="예측 대상  ", bg=CARD, fg=MIDTEXT,
                 font=("Arial", 10, "bold")).pack(side="left")
        tk.Label(target_row, text="현재 경기 상황 기준 홈팀 승리 확률", bg=CARD, fg=TEXT,
                 font=("Arial", 10)).pack(side="left")

        cols = tk.Frame(body, bg=CARD)
        cols.pack(anchor="w", fill="x", pady=(0, 8))
        cols.columnconfigure((0, 1), weight=1, uniform="modelinfo")

        used_col = tk.Frame(cols, bg=CARD)
        used_col.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        tk.Label(used_col, text="모델 입력 변수", bg=CARD, fg=MINT,
                 font=("Arial", 10, "bold")).pack(anchor="w")
        for label in ["이닝", "공격/수비", "아웃카운트", "점수차",
                       "1루 주자 여부", "2루 주자 여부", "3루 주자 여부"]:
            tk.Label(used_col, text=f"· {label}", bg=CARD, fg=TEXT,
                     font=("Arial", 10)).pack(anchor="w")

        unused_col = tk.Frame(cols, bg=CARD)
        unused_col.grid(row=0, column=1, sticky="nw", padx=(12, 0))
        tk.Label(unused_col, text="모델이 사용하지 않는 정보", bg=CARD, fg=RED,
                 font=("Arial", 10, "bold")).pack(anchor="w")
        for label in ["팀 이름", "선수 이름", "타율", "홈런 개수", "ERA"]:
            tk.Label(unused_col, text=f"· {label}", bg=CARD, fg=MIDTEXT,
                     font=("Arial", 10)).pack(anchor="w")

        tk.Label(body, text="모델은 현재 경기 상황만 보고 홈팀 승리 확률을 계산합니다.",
                 bg=CARD, fg=SUBTEXT, font=("Arial", 10, "italic"),
                 justify="left", anchor="w", wraplength=1000).pack(anchor="w", pady=(0, 14))

    def _compute_feature_importance(self):
        """학습된 모델에서 feature별 중요도(%)를 자동 계산.
        - 트리 기반 모델: feature_importances_ 사용
        - 선형 모델(로지스틱 회귀 등): |계수| 사용 (StandardScaler로 스케일이 맞춰져 있어 비교 가능)
        반환: [(feature_col, 중요도%), ...] 중요도 내림차순. 계산 불가 시 None."""
        try:
            clf = self.model.pipeline.named_steps["clf"]
        except Exception:
            return None

        if hasattr(clf, "feature_importances_"):
            raw = np.abs(np.asarray(clf.feature_importances_, dtype=float))
        elif hasattr(clf, "coef_"):
            raw = np.abs(np.asarray(clf.coef_, dtype=float)).reshape(-1)
        else:
            return None

        if raw.size != len(FEATURE_COLS) or raw.sum() <= 0:
            return None

        pct = raw / raw.sum() * 100
        pairs = list(zip(FEATURE_COLS, pct))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def _build_feature_importance_card(self, parent):
        """[모델 중요 변수 TOP N] — feature_importances_(또는 |계수|) 기반 막대그래프"""
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x", pady=(0, 12))

        tk.Label(card, text=f"📊 모델 중요 변수 TOP {TOP_N_IMPORTANCE}", bg=CARD, fg=TITLE,
                 font=("Arial", 12, "bold")).pack(anchor="w", padx=18, pady=(14, 8))

        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=18, pady=(0, 14))

        importance = self._compute_feature_importance()
        if not importance:
            tk.Label(body, text="모델이 학습되지 않아 변수 중요도를 계산할 수 없습니다.",
                     bg=CARD, fg=SUBTEXT, font=("Arial", 10)).pack(anchor="w")
            return

        top = importance[:TOP_N_IMPORTANCE]
        max_pct = top[0][1] if top[0][1] > 0 else 1.0

        for i, (feat, pct) in enumerate(top, start=1):
            label = FEATURE_LABELS_SHORT.get(feat, feat)
            row = tk.Frame(body, bg=CARD)
            row.pack(fill="x", pady=(0, 7))

            tk.Label(row, text=f"{i}. {label}", bg=CARD, fg=TEXT,
                     font=("Arial", 10, "bold"), width=12, anchor="w").pack(side="left")

            track = tk.Frame(row, bg=CARD2, height=16)
            track.pack(side="left", fill="x", expand=True, padx=(8, 10))
            track.pack_propagate(False)

            ratio = max(pct / max_pct, 0.01)
            bar_color = MINT if i == 1 else BLUE
            fill = tk.Frame(track, bg=bar_color)
            fill.place(relx=0, rely=0, relwidth=ratio, relheight=1)

            tk.Label(row, text=f"{pct:.1f}%", bg=CARD, fg=bar_color,
                     font=("Arial", 10, "bold"), width=6, anchor="e").pack(side="left")

        tk.Label(body, text="중요도는 학습된 모델의 계수(feature_importances_/계수 절댓값)를 "
                             "정규화하여 자동 계산됩니다.",
                 bg=CARD, fg=SUBTEXT, font=("Arial", 9), justify="left", anchor="w",
                 wraplength=1000).pack(anchor="w", pady=(4, 0))

    def _toggle_model_info(self):
        """[머신러닝 모델 정보] / [모델 중요 변수 TOP N] 패널 접기/펼치기 (기본 숨김)"""
        self._model_info_visible = not self._model_info_visible
        if self._model_info_visible:
            self.model_info_container.pack(fill="x", before=self.team_row)
            self.btn_model_info.config(text="🤖 모델 설명 숨기기")
        else:
            self.model_info_container.pack_forget()
            self.btn_model_info.config(text="🤖 모델 설명 보기")

    def _mkbtn(self, parent, text, cmd, fg=MIDTEXT):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=CARD2, fg=fg, activebackground=CARD, activeforeground=fg,
            font=("Arial", 10, "bold"), relief="flat", bd=0,
            padx=10, pady=7, cursor="hand2", takefocus=0,
        )

    def _style_ax(self, ax):
        ax.set_facecolor(CARD2)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=AXIS_TEXT, labelsize=10)
        ax.title.set_color(TITLE)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.6)

    # 키보드 단축키
    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._on_prev())
        self.bind("<Right>", lambda e: self._on_next())
        self.bind("<space>", lambda e: self._on_toggle_play())
        self.bind("<Home>", lambda e: self._on_restart())
        self.bind("<End>", lambda e: self._on_last())
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<h>", lambda e: self._show_help())
        self.bind("<H>", lambda e: self._show_help())
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.bind("<d>", lambda e: self._toggle_debug())
        self.bind("<D>", lambda e: self._toggle_debug())

    def _on_mousewheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self._scroll_canvas.yview_scroll(delta, "units")
        return "break"

    # 외부(main.py)에서 호출하는 데이터 갱신 API
    def set_history(self, states, probs):
        """경기 전체(또는 LIVE 진행 중) play-by-play 데이터 반영"""
        print(f"[ml_debug] set_history 호출: len(states)={len(states) if states else 0}, "
              f"len(probs)={len(probs) if probs else 0}, "
              f"기존 play_idx={self.play_idx}, 기존 길이={len(self.all_states)}")

        if not states or not probs:
            print("[ml_debug] set_history: states/probs 비어있음 -> 무시")
            return

        is_new_game = (
            not self.all_states
            or len(states) < len(self.all_states)
            or states[0].get("home_team") != self.all_states[0].get("home_team")
            or states[0].get("away_team") != self.all_states[0].get("away_team")
        )

        was_caught_up = self.play_idx >= len(self.all_states)
        was_demo = self._is_demo

        self.all_states = list(states)
        self.all_probs  = list(probs)
        self._is_demo = False

        if is_new_game or was_demo:
            self._stop_play()
            self.play_idx = 1
        elif was_caught_up:
            # LIVE 모드: 끝까지 따라잡은 상태였으면 새 타석으로 계속 따라감
            self.play_idx = len(self.all_states)
        # 그 외(과거 타석을 되돌려보는 중)에는 play_idx 유지

        self.progress_scale.config(to=max(len(self.all_states), 1))
        self._compute_highlights()

        print(f"[ml_debug] set_history 반영 완료: 총 {len(self.all_states)}타석, "
              f"play_idx={self.play_idx}, is_new_game={is_new_game}, was_demo={was_demo}")

        self._render()

        # 새 경기(또는 데모 -> 실제 데이터 전환) 데이터가 들어오면
        # 사용자가 "▶ 재생"을 따로 누르지 않아도 자동으로 처음부터 재생 시작
        if (is_new_game or was_demo) and not self.playing:
            self._on_start()

    def _compute_highlights(self):
        """승률 변화(|delta|) 기준 TOP N 타석 계산
        idx번 타석의 효과 = win_probs[idx] - win_probs[idx-1] (1-indexed idx)"""
        deltas = []
        for idx in range(1, len(self.all_probs)):
            delta = (self.all_probs[idx] - self.all_probs[idx - 1]) * 100
            deltas.append((idx, delta))  # idx는 1-indexed, "idx번 타석의 결과 효과"
        deltas.sort(key=lambda t: abs(t[1]), reverse=True)
        self.highlights = deltas[:TOP_N_HIGHLIGHT]
        self._update_highlight_cards()

    # idx(1-indexed) 타석의 "전/후" 승률·상태 헬퍼
    #   - 타석 전 승률 = win_probs[idx-1] (= 모델이 실제 사용한 입력 기준 승률)
    #   - 타석 후 승률 = win_probs[idx]   (= 다음 타석의 "전" 승률 = 이번 결과 반영 후)
    #   - 마지막 타석은 다음 상태가 없으므로 "전" 값을 그대로 사용한다.
    def _prob_before(self, idx):
        return self.all_probs[idx - 1]

    def _prob_after(self, idx):
        if idx < len(self.all_probs):
            return self.all_probs[idx]
        return self.all_probs[idx - 1]

    def _post_state(self, idx):
        if idx < len(self.all_states):
            return self.all_states[idx]
        return self.all_states[idx - 1]

    def _post_score(self, idx):
        """idx번 타석 결과가 반영된 후의 (원정, 홈) 점수"""
        state = self.all_states[idx - 1]
        if idx < len(self.all_states):
            nxt = self.all_states[idx]
            return nxt.get("away_score", 0), nxt.get("home_score", 0)
        rbi = state.get("rbi", 0) or 0
        is_top = bool(state.get("is_top"))
        away = state.get("away_score", 0) + (rbi if is_top else 0)
        home = state.get("home_score", 0) + (rbi if not is_top else 0)
        return away, home

    # 재생 / 이동 컨트롤 핸들러
    def _jump_to(self, idx, pause=True):
        if not self.all_states:
            return
        idx = max(1, min(idx, len(self.all_states)))
        if pause:
            self._stop_play()
        if idx == self.play_idx:
            return
        self.play_idx = idx
        self._render()

    def _on_prev(self):
        self._jump_to(self.play_idx - 1)

    def _on_next(self):
        self._jump_to(self.play_idx + 1)

    def _on_last(self):
        self._jump_to(len(self.all_states))

    def _on_toggle_play(self):
        if self.playing:
            self._on_pause()
        else:
            self._on_start()

    def _on_restart(self):
        self._stop_play()
        self.play_idx = 1 if self.all_states else 0
        self._render()

    def _on_replay(self):
        self._stop_play()
        if not self.all_states:
            return
        self.play_idx = 1
        self._render()
        self._on_start()

    def _on_start(self):
        if self.playing or not self.all_states:
            return
        if self.play_idx >= len(self.all_states):
            self.play_idx = 1  # 끝까지 봤으면 처음부터 다시 재생
        self.playing = True
        self.btn_play.config(text="⏸ 일시정지", fg=YELLOW)
        self._play_step()

    def _on_pause(self):
        self._stop_play()

    def _stop_play(self):
        self.playing = False
        if self.anim_job:
            self.after_cancel(self.anim_job)
            self.anim_job = None
        if hasattr(self, "btn_play"):
            self.btn_play.config(text="▶ 재생", fg=MINT)

    def _play_step(self):
        if not self.playing:
            return
        if self.play_idx >= len(self.all_states):
            self._stop_play()
            return

        self.play_idx += 1

        try:
            self._render()
        except Exception:
            import traceback
            traceback.print_exc()

        state = self.all_states[self.play_idx - 1]
        prob_pct = self.all_probs[self.play_idx - 1] * 100
        event = state.get("description") or state.get("event") or "-"
        print(f"[{self.play_idx}/{len(self.all_states)}] {event} {prob_pct:.1f}%")

        if self.play_idx < len(self.all_states):
            self.anim_job = self.after(self.speed_ms, self._play_step)
        else:
            self._stop_play()

    # 진행 슬라이더
    def _on_slider_move(self, value):
        if not self.all_states:
            return
        idx = int(float(value))
        if idx != self.play_idx:
            self._jump_to(idx)

    # 승률 그래프 클릭 / 호버
    def _on_prob_click(self, event):
        if not self.all_states or event.inaxes != self.ax_prob or event.xdata is None:
            return
        idx = int(round(event.xdata))
        self._jump_to(idx)

    def _on_prob_hover(self, event):
        if not self.all_states or self._prob_annot is None:
            return
        if event.inaxes != self.ax_prob or event.xdata is None:
            if self._prob_annot.get_visible():
                self._prob_annot.set_visible(False)
                self.canvas_prob.draw_idle()
            return

        # 그래프 x=k 위치의 값(all_probs[k-1])은 "(k-1)번 타석 결과가 반영된 후" 승률이다.
        n = len(self.all_states)
        k = int(round(event.xdata))
        k = max(1, min(k, n))
        prob = self.all_probs[k - 1] * 100

        if k == 1:
            self._prob_annot.xy = (k, prob)
            self._prob_annot.set_text(f"경기 시작 시점 승률 {prob:.1f}%")
        else:
            idx = k - 1
            state = self.all_states[idx - 1]
            ev = state.get("description") or state.get("event") or "-"
            delta = (self.all_probs[idx] - self.all_probs[idx - 1]) * 100
            self._prob_annot.xy = (k, prob)
            self._prob_annot.set_text(f"#{idx}  {ev}\n결과 반영 후 승률 {prob:.1f}%  ({delta:+.1f}%p)")
        self._prob_annot.set_visible(True)
        self.canvas_prob.draw_idle()

    # 주요 장면 카드 (그리드 배치: 컴팩트=3열, 일반=5열)
    def _update_highlight_cards(self):
        for w in self.highlight_frame.winfo_children():
            w.destroy()

        if not self.highlights:
            tk.Label(self.highlight_frame, text="데이터가 더 필요합니다.",
                     bg=BG, fg=SUBTEXT, font=("Arial", 10)).pack(side="left")
            return

        ncols = 3 if self.compact else 5
        for c in range(5):
            if c < ncols:
                self.highlight_frame.columnconfigure(c, weight=1, uniform="hl")
            else:
                self.highlight_frame.columnconfigure(c, weight=0, uniform="")

        wraplen = 110 if self.compact else 140
        ev_max  = 16 if self.compact else 22
        title_font = ("Arial", 9, "bold") if self.compact else ("Arial", 10, "bold")
        delta_font = ("Arial", 11, "bold") if self.compact else ("Arial", 12, "bold")
        sub_font = ("Arial", 8) if self.compact else ("Arial", 9)
        pad_y = (5, 6) if self.compact else (8, 8)

        for i, (idx, delta) in enumerate(self.highlights):
            state = self.all_states[idx - 1]
            batter = state.get("batter") or "타자"
            event_kr = _event_kr(state)
            highlight_text = f"{batter} {event_kr}"
            if len(highlight_text) > ev_max:
                highlight_text = highlight_text[:ev_max - 1] + "…"
            inning_line = _inning_half_kr(state)

            prev_pct = self._prob_before(idx) * 100
            cur_pct = self._prob_after(idx) * 100

            color = MINT if delta > 0 else (RED if delta < 0 else MIDTEXT)
            title = "🔥 경기 최대 분기점" if i == 0 else f"#{i + 1} 주요 장면"
            title_color = YELLOW if i == 0 else TITLE

            row, col = divmod(i, ncols)
            card = tk.Frame(self.highlight_frame, bg=CARD2, cursor="hand2")
            card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)

            title_lbl = tk.Label(card, text=title, bg=CARD2, fg=title_color, font=title_font)
            title_lbl.pack(anchor="w", padx=10, pady=(pad_y[0], 0))
            inning_lbl = tk.Label(card, text=inning_line, bg=CARD2, fg=MIDTEXT, font=sub_font)
            inning_lbl.pack(anchor="w", padx=10)
            ev_lbl = tk.Label(card, text=highlight_text, bg=CARD2, fg=MIDTEXT,
                               font=sub_font, wraplength=wraplen, justify="left")
            ev_lbl.pack(anchor="w", padx=10)
            delta_lbl = tk.Label(
                card, text=f"{prev_pct:.0f}% → {cur_pct:.0f}% ({delta:+.0f}%)",
                bg=CARD2, fg=color, font=delta_font)
            delta_lbl.pack(anchor="w", padx=10, pady=(0, pad_y[1]))

            for w in (card, title_lbl, inning_lbl, ev_lbl, delta_lbl):
                w.bind("<Button-1>", lambda e, i=idx: self._jump_to(i))

    # 데모(가상) 데이터 생성
    def _load_demo_if_empty(self):
        if self.all_states:
            return
        if not getattr(self.model, "is_trained", False):
            return

        states = self._generate_demo_states()
        probs = self.model.predict_batch(states)

        self.all_states = states
        self.all_probs  = probs
        self.play_idx   = 1
        self._is_demo   = True
        self.progress_scale.config(to=max(len(self.all_states), 1))
        self.subtitle_lbl.config(
            text="🧠 ML 모델 실시간 분석 — 실제 경기 데이터 없음 → 가상 시뮬레이션 미리보기 중")

    @staticmethod
    def _generate_demo_states():
        rng = random.Random(7)
        events = ["삼진", "땅볼 아웃", "뜬공 아웃", "1루타", "2루타", "3루타",
                  "홈런", "포볼", "병살타", "희생플라이"]

        states = []
        home_score = away_score = 0
        ab_index = 0

        for inning in range(1, 10):
            for is_top in (1, 0):
                outs = 0
                runners = [0, 0, 0]  # 1B, 2B, 3B
                while outs < 3:
                    event = rng.choice(events)

                    if event == "삼진":
                        outs += 1
                    elif event in ("땅볼 아웃", "뜬공 아웃"):
                        outs += 1
                        if event == "뜬공 아웃" and runners[2] and rng.random() < 0.4:
                            runners[2] = 0
                            if is_top:
                                away_score += 1
                            else:
                                home_score += 1
                    elif event == "병살타":
                        outs += 2
                        runners = [0, runners[1], 0]
                    elif event == "희생플라이":
                        outs += 1
                        if runners[2]:
                            runners[2] = 0
                            if is_top:
                                away_score += 1
                            else:
                                home_score += 1
                    elif event == "포볼":
                        if runners[0] and runners[1] and runners[2]:
                            if is_top:
                                away_score += 1
                            else:
                                home_score += 1
                        elif runners[0] and runners[1]:
                            runners[2] = 1
                        elif runners[0]:
                            runners[1] = 1
                        runners[0] = 1
                    elif event == "1루타":
                        scored = runners[2] + runners[1]
                        if is_top:
                            away_score += scored
                        else:
                            home_score += scored
                        runners = [1, runners[0], 0]
                    elif event == "2루타":
                        scored = runners[2] + runners[1] + runners[0]
                        if is_top:
                            away_score += scored
                        else:
                            home_score += scored
                        runners = [0, 1, 0]
                    elif event == "3루타":
                        scored = sum(runners)
                        if is_top:
                            away_score += scored
                        else:
                            home_score += scored
                        runners = [0, 0, 1]
                    elif event == "홈런":
                        scored = sum(runners) + 1
                        if is_top:
                            away_score += scored
                        else:
                            home_score += scored
                        runners = [0, 0, 0]

                    if outs > 3:
                        outs = 3

                    score_diff = int(np.clip(home_score - away_score, -10, 10))

                    states.append({
                        "inning": min(inning, 9),
                        "is_top": is_top,
                        "outs": min(outs, 2) if outs < 3 else 2,
                        "score_diff": score_diff,
                        "runner_1b": runners[0],
                        "runner_2b": runners[1],
                        "runner_3b": runners[2],
                        "display_inning": inning,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_team": "홈팀 (데모)",
                        "away_team": "원정팀 (데모)",
                        "event": event,
                        "description": f"{'원정' if is_top else '홈'}팀 타자 — {event}",
                        "batter": "타자 (데모)",
                        "pitcher": "투수 (데모)",
                        "at_bat_index": ab_index,
                    })
                    ab_index += 1

        return states

    # 렌더링
    def _render(self):
        try:
            self._render_impl()
        except Exception:
            import traceback
            traceback.print_exc()

    def _render_impl(self):
        if not self.all_states or self.play_idx <= 0:
            self._draw_empty()
            return

        idx = min(self.play_idx, len(self.all_states))
        state = self.all_states[idx - 1]          # 타석 전 상태 (모델 입력)
        post_state = self._post_state(idx)        # 타석 후 상태 (결과 반영)

        prob_before = self._prob_before(idx)
        prob_after = self._prob_after(idx)
        prev_pct = prob_before * 100
        pct = prob_after * 100
        delta = pct - prev_pct

        # [경기명]
        away = state.get("away_team", "원정팀")
        home = state.get("home_team", "홈팀")
        prefix = "(데모) " if self._is_demo else ""
        self.title_lbl.config(text=f"{prefix}{away}  @  {home}")
        self.subtitle_lbl.config(text="🧠 ML 모델 실시간 분석 — 타석이 진행될 때마다 자동 갱신")

        # [원정/홈 팀 카드]
        self._update_team_cards(idx)

        # [이닝별 득점표]
        self._draw_linescore()

        # [현재 홈팀 승률] + [이전 승률] + [승률 변화] — "이 타석 결과가 반영된 후" 승률을 표시
        self.prob_lbl.config(text=f"{pct:.1f}%", fg=MINT if pct >= 50 else RED)
        d_color = MINT if delta > 0.05 else (RED if delta < -0.05 else MIDTEXT)
        fire = " 🔥" if abs(delta) >= BIG_SWING else ""
        self.prev_prob_lbl.config(text=f"{prev_pct:.1f}%")
        if idx == len(self.all_states):
            self.delta_lbl.config(text=f"{delta:+.1f}%p{fire} (마지막)", fg=d_color)
        else:
            self.delta_lbl.config(text=f"{delta:+.1f}%p{fire}", fg=d_color)

        # [현재 경기 상황 카드] — 점수/아웃/주자/점수차는 "결과 반영 후" 상황을 표시
        half = "초" if post_state.get("is_top") else "말"
        bases_txt = _runners_text(post_state)
        sd = post_state.get("score_diff", 0)
        sd_txt = f"+{sd}" if sd > 0 else str(sd)
        attack_team, defense_team = _attack_defense(post_state)

        scored = _scored_runs(idx, self.all_states)
        score_change_txt = f"이번 타석 득점: {scored}점" if scored > 0 else "이번 타석 득점: 없음"

        post_away, post_home = self._post_score(idx)
        disp_inning = post_state.get('display_inning', post_state.get('inning', '?'))

        self._set_state_text([
            [(f"현재 타석 #{idx}", "idx"), ("  ", "body"), (f"({disp_inning}회 {half})", "inning")],
            [("공격: ", "label"), (attack_team, "team"), ("   /   수비: ", "label"), (defense_team, "team")],
            [("타자: ", "label"), (state.get('batter', '-'), "name"),
             ("   투수: ", "label"), (state.get('pitcher', '-'), "name")],
            [(f"볼카운트: {state.get('balls', 0)}-{state.get('strikes', 0)}   "
              f"결과: {_event_kr(state)}", "body")],
            [(f"스코어 (원정:홈) {post_away} : {post_home}   ({score_change_txt})", "body")],
            [(f"아웃 {post_state.get('outs', 0)}   주자 {bases_txt}   점수차(홈기준) {sd_txt}", "body")],
        ])
        self.progress_lbl.config(text=f"{idx} / {len(self.all_states)} 타석")
        if int(self.progress_scale.get()) != idx:
            self.progress_scale.set(idx)

        # 중계형 요약 카드
        self._update_summary_card(idx)

        # AI 분석 요약
        self.ai_summary_lbl.config(text=self._build_ai_summary_text(idx))

        # 정보 카드 3종
        self.event_card.body.config(text=self._build_event_card_text(state))
        self.reason_card.body.config(text=self._build_reason_text(idx))
        self.basis_card.body.config(text=self._build_model_basis(post_state))

        # 현재 경기 상황(feature) — 결과 반영 후 상태 기준
        self.feature_lbl.config(text=self._build_feature_friendly_text(post_state))
        self.feature_detail_lbl.config(text=self._build_feature_text(post_state))

        # 디버그 정보 (승률 변화 원인 분석용)
        self.debug_lbl.config(text=self._build_debug_text(idx))

        # 그래프
        self._draw_prob_line(idx)
        self._draw_contribution(post_state)

        self.canvas_prob.draw_idle()
        self.canvas_contrib.draw_idle()

        # MVP (경기 종료 시)
        self._update_mvp(idx)

    def _update_team_cards(self, idx):
        """idx번 타석 결과가 반영된 후의 점수/공수 상황으로 팀 카드를 갱신"""
        state = self.all_states[idx - 1]
        post_state = self._post_state(idx)
        away_score, home_score = self._post_score(idx)

        away_attacking = bool(post_state.get("is_top"))
        home_attacking = not away_attacking

        self.away_widgets["name"].config(text=state.get("away_team", "원정팀"))
        self.away_widgets["score"].config(text=str(away_score))
        self.home_widgets["name"].config(text=state.get("home_team", "홈팀"))
        self.home_widgets["score"].config(text=str(home_score))

        self._set_team_card_style(self.away_card, self.away_widgets, away_attacking)
        self._set_team_card_style(self.home_card, self.home_widgets, home_attacking)

        attack_team, _ = _attack_defense(post_state)
        self.attack_lbl.config(text=f"⚾ 현재 공격: {attack_team}")

        sd = post_state.get("score_diff", 0)
        sd_txt = f"+{sd}" if sd > 0 else str(sd)
        self.scorediff_lbl.config(text=f"홈팀 기준 점수차: {sd_txt}")

    def _set_state_text(self, lines):
        """state_lbl(Text) 내용을 줄/구간별 태그로 갱신 — (텍스트, 태그) 튜플의 리스트의 리스트"""
        txt = self.state_lbl
        txt.config(state="normal")
        txt.delete("1.0", "end")
        for i, segs in enumerate(lines):
            if i > 0:
                txt.insert("end", "\n")
            for s, tag in segs:
                txt.insert("end", s, tag)
        txt.config(state="disabled")

    def _set_team_card_style(self, card, widgets, attacking):
        bg = CARD2 if attacking else DIMCARD
        fg = TITLE if attacking else MIDTEXT
        card.config(bg=bg)
        widgets["head"].config(bg=bg)
        widgets["body"].config(bg=bg)
        widgets["name"].config(bg=bg, fg=fg)
        widgets["score"].config(bg=bg, fg=fg if attacking else MIDTEXT)

    def _draw_empty(self):
        self.title_lbl.config(text="경기를 선택하세요")
        self.subtitle_lbl.config(text="🧠 ML 모델 실시간 분석 — 경기 분석을 시작하면 갱신됩니다.")
        self.prob_lbl.config(text="─.─%", fg=MINT)
        self.prev_prob_lbl.config(text="─.─%")
        self.delta_lbl.config(text="", fg=MIDTEXT)
        self._set_state_text([[("대기 중...", "body")]])
        self.progress_lbl.config(text="- / -")
        self.summary_title_lbl.config(text="-")
        self.summary_inning_lbl.config(text="-")
        self.summary_desc_lbl.config(text="-")
        self.summary_prob_lbl.config(text="- → -", fg=TEXT)
        self.summary_delta_lbl.config(text="")
        self.summary_orig_lbl.config(text="")
        self.feature_lbl.config(text="-")
        self.feature_detail_lbl.config(text="-")
        self.debug_lbl.config(text="-")
        self.contrib_explain_lbl.config(text="-")
        self.attack_lbl.config(text="현재 공격: -")
        self.scorediff_lbl.config(text="홈팀 기준 점수차: -")
        for card in (self.event_card, self.reason_card, self.basis_card):
            card.body.config(text="─")
        for widgets in (self.away_widgets, self.home_widgets):
            widgets["name"].config(text="-")
            widgets["score"].config(text="0")
        self._set_mvp_visible(False)
        self._draw_linescore()

        self.ax_prob.cla()
        self._style_ax(self.ax_prob)
        self.ax_prob.set_title("홈팀 승률 변화", fontsize=12)
        self.ax_prob.set_ylim(0, 100)
        self.ax_prob.axhline(50, color=BORDER, linewidth=1, linestyle="--", alpha=0.7)
        self.ax_prob.text(0.5, 0.5, "대기 중", transform=self.ax_prob.transAxes,
                           ha="center", va="center", color=SUBTEXT, fontsize=12)
        self._prob_annot = None

        self.ax_contrib.cla()
        self._style_ax(self.ax_contrib)
        self.ax_contrib.set_title(f"승률에 영향을 준 요인 (상위 {TOP_N_CONTRIB}개)", fontsize=11)

        self.canvas_prob.draw_idle()
        self.canvas_contrib.draw_idle()

    # 대형 승률 변화 그래프
    def _draw_prob_line(self, idx):
        n = len(self.all_states)
        # 현재 선택된 타석의 "결과 반영 후" 위치 = idx+1 (마지막 타석은 idx 그대로)
        marker_x = idx + 1 if idx < n else idx

        y_all = np.array(self.all_probs[:marker_x]) * 100
        x_all = np.arange(1, marker_x + 1)

        if self.view_var.get() == "전체":
            x, y = x_all, y_all
        else:
            x, y = x_all[-RECENT_N:], y_all[-RECENT_N:]

        self.ax_prob.cla()
        self._style_ax(self.ax_prob)
        self.ax_prob.set_title("타석 진행에 따른 홈팀 승률 변화", fontsize=13, pad=10)
        self.ax_prob.set_xlabel("타석 번호", fontsize=11)
        self.ax_prob.set_ylabel("홈팀 승률 (%)", fontsize=11)
        self.ax_prob.set_ylim(0, 100)
        self.ax_prob.axhline(50, color=BORDER, linewidth=1.2, linestyle="--", alpha=0.7)

        if len(y) > 0:
            if len(y) > 1:
                # 과거 구간: 연한 선 / 마지막 구간: 강조
                self.ax_prob.plot(x[:-1], y[:-1], color=MINT, linewidth=1.6, alpha=0.45)
                self.ax_prob.plot(x[-2:], y[-2:], color=MINT, linewidth=2.6, alpha=1.0)
            else:
                self.ax_prob.plot(x, y, color=MINT, linewidth=2.6)

            self.ax_prob.fill_between(x, y, 50, where=(y >= 50), color=MINT, alpha=0.12)
            self.ax_prob.fill_between(x, y, 50, where=(y < 50), color=RED, alpha=0.12)

            # 승률 급변(|delta| >= BIG_SWING) 타석 = 노란 별 표시 (결과 반영 후 위치)
            for h_idx, h_delta in self.highlights:
                if abs(h_delta) >= BIG_SWING:
                    star_x = h_idx + 1 if h_idx < n else h_idx
                    if star_x <= marker_x and x[0] <= star_x <= x[-1]:
                        self.ax_prob.scatter(
                            [star_x], [y_all[star_x - 1]], color=YELLOW, marker="*",
                            s=180, zorder=4, edgecolors=TEXT, linewidths=0.6)

            # 현재 선택된 타석 = 빨간 점 강조 (결과 반영 후 승률 위치)
            self.ax_prob.scatter([x[-1]], [y[-1]], color=RED, zorder=5, s=70,
                                  edgecolors=TEXT, linewidths=1)
            self.ax_prob.annotate(
                f"{y[-1]:.1f}%", (x[-1], y[-1]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", color=TEXT, fontsize=11, fontweight="bold")

        # 클릭/호버 안내용 빈 annotation (cla() 이후 매번 재생성)
        self._prob_annot = self.ax_prob.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round", fc=CARD2, ec=BORDER, alpha=0.95),
            color=TEXT, fontsize=9, visible=False, zorder=10)

    # 특징별 기여도 그래프 (상위 N개)
    def _draw_contribution(self, state):
        try:
            clf    = self.model.pipeline.named_steps["clf"]
            scaler = self.model.pipeline.named_steps["scaler"]
        except Exception:
            return

        raw = np.array([state.get(c, 0) for c in FEATURE_COLS], dtype=float)
        scaled = (raw - scaler.mean_) / scaler.scale_
        contrib = scaled * clf.coef_[0]

        # 상위 N개를 |기여도| 기준으로 선택, 중요도 순(큰 값이 위쪽)으로 정렬
        order_desc = np.argsort(np.abs(contrib))[::-1][:TOP_N_CONTRIB]
        order = order_desc[::-1]  # barh는 첫 항목이 아래쪽에 그려지므로 역순으로

        labels = [FEATURE_LABELS_KR.get(FEATURE_COLS[i], FEATURE_COLS[i]) for i in order]
        values = contrib[order]
        colors = [MINT if v >= 0 else RED for v in values]

        self.ax_contrib.cla()
        self._style_ax(self.ax_contrib)
        self.ax_contrib.barh(labels, values, color=colors)
        self.ax_contrib.axvline(0, color=SUBTEXT, linewidth=1)
        self.ax_contrib.set_title(
            "승률에 영향을 준 요인 — 초록=승률 상승 / 빨강=승률 하락",
            fontsize=11, pad=8)
        self.ax_contrib.set_xlabel("영향력 크기", fontsize=10)

        # 막대 옆 일반인용 설명 문장 (영향력이 큰 순서대로, 위에서부터)
        explain_lines = []
        for i in order_desc:
            feat = FEATURE_COLS[i]
            explain_lines.append(_feature_friendly_line(feat, raw[i], contrib[i], state))
        self.contrib_explain_lbl.config(text="\n".join(explain_lines))

    # 자연어 설명 생성
    def _update_summary_card(self, idx):
        """야구 중계형 요약 카드 갱신 (일반인용)
        타석 전 승률 = win_probs[idx-1], 타석 후 승률 = win_probs[idx] (결과 반영 후)"""
        state = self.all_states[idx - 1]
        prob_before = self._prob_before(idx) * 100
        prob_after = self._prob_after(idx) * 100
        delta = prob_after - prob_before

        batter = state.get("batter") or "타자"
        event_kr = _event_kr(state)
        inning_line = _inning_half_kr(state)
        rbi = state.get("rbi", 0) or 0
        scored = _scored_runs(idx, self.all_states)

        if abs(delta) >= BIG_SWING:
            emoji = "🔥"
        elif scored > 0:
            emoji = "⚾"
        else:
            emoji = "📍"

        result_text = f"{event_kr}, {rbi}타점" if rbi > 0 else event_kr
        self.summary_title_lbl.config(text=f"{emoji} {batter} {result_text}")
        self.summary_inning_lbl.config(text=inning_line)

        josa_ga = _josa(batter, "이/가")
        josa_ro = _josa(result_text, "으로/로")
        if delta > 0.05:
            direction = "상승"
        elif delta < -0.05:
            direction = "하락"
        else:
            direction = None

        if direction:
            desc = (f"{batter}{josa_ga} {result_text}{josa_ro} "
                    f"홈팀 승률이 {prob_before:.1f}%에서 {prob_after:.1f}%로 {direction}했습니다.")
        else:
            desc = (f"{batter}{josa_ga} {result_text}{josa_ro} "
                    f"홈팀 승률에는 거의 변화가 없습니다 ({prob_before:.1f}% → {prob_after:.1f}%).")
        if idx == len(self.all_states):
            desc += " (마지막 타석 — 다음 상태 없음)"
        self.summary_desc_lbl.config(text=desc)

        self.summary_prob_lbl.config(
            text=f"타석 전 {prob_before:.1f}% → 타석 후 {prob_after:.1f}%", fg=TEXT)
        if delta > 0.05:
            self.summary_delta_lbl.config(text=f"(+{delta:.1f}%p ▲)", fg=MINT)
        elif delta < -0.05:
            self.summary_delta_lbl.config(text=f"({delta:+.1f}%p ▼)", fg=RED)
        else:
            self.summary_delta_lbl.config(text="(변화 거의 없음 ─)", fg=SUBTEXT)

        original = state.get("description") or ""
        self.summary_orig_lbl.config(text=f"원문: {original}" if original else "")

    def _build_event_card_text(self, state):
        batter = state.get("batter") or "-"
        situation = _translate_situation(state)
        lines = [f"결과: {_event_kr(state)}", f"타자: {batter}"]
        if situation:
            lines.append(f"상황: {situation}")
        return "\n".join(lines)

    def _build_feature_friendly_text(self, state):
        """모델 입력 feature -> 일반인용 '현재 경기 상황' 설명"""
        attack_team, defense_team = _attack_defense(state)
        sd = state.get("score_diff", 0)
        if sd > 0:
            score_line = f"홈팀 {sd}점 리드"
        elif sd < 0:
            score_line = f"원정팀 {abs(sd)}점 리드"
        else:
            score_line = "동점"

        lines = [
            _inning_half_kr(state),
            f"{state.get('outs', 0)}아웃",
            f"주자 {_runners_text(state)}",
            score_line,
            f"{attack_team} 공격",
            f"{defense_team} 수비",
        ]
        return "   ·   ".join(lines)

    def _build_feature_text(self, state):
        lines = [f"{c} = {state.get(c, 0)}" for c in FEATURE_COLS]
        return "\n".join(lines)

    def _build_reason_text(self, idx):
        """일반인용 — 왜 승률이 이렇게 변했는지 쉬운 문장으로 설명
        prev = 타석 전 상태(states[idx-1]), cur = 타석 후 상태(states[idx])
        delta = win_probs[idx] - win_probs[idx-1] (이 타석 결과의 효과)

        모델 입력 feature(이닝/공수교대/아웃/점수차/주자)의 '변화량'을 로지스틱 회귀
        계수 기반으로 근사 분해하여, 승률을 끌어올린 요인은 '승률 상승 이유',
        끌어내린 요인은 '승률 하락 이유'로 나누어 일반인용 문장 + 기존 수치(%p)를 함께 보여준다."""
        prev = self.all_states[idx - 1]
        cur = self._post_state(idx)

        prev_pct = self._prob_before(idx) * 100
        cur_pct = self._prob_after(idx) * 100
        delta = cur_pct - prev_pct

        arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "─")
        header = f"{arrow} 홈팀 승률 {delta:+.1f}%p"

        # feature 변화량 기반 요인별 영향(%p) 계산 (로지스틱 회귀 계수 선형 근사)
        rising, falling = [], []
        try:
            clf    = self.model.pipeline.named_steps["clf"]
            scaler = self.model.pipeline.named_steps["scaler"]
            raw_prev = np.array([prev.get(c, 0) for c in FEATURE_COLS], dtype=float)
            raw_cur  = np.array([cur.get(c, 0) for c in FEATURE_COLS], dtype=float)
            scaled_prev = (raw_prev - scaler.mean_) / scaler.scale_
            scaled_cur  = (raw_cur  - scaler.mean_) / scaler.scale_
            contrib_prev = scaled_prev * clf.coef_[0]
            contrib_cur  = scaled_cur  * clf.coef_[0]
            delta_contrib = contrib_cur - contrib_prev

            p = self.all_probs[idx - 1]
            slope = p * (1 - p)  # dp/dz (로지스틱 미분, 선형 근사)

            groups = {
                "점수차": ["score_diff"],
                "주자 상황": ["runner_1b", "runner_2b", "runner_3b"],
                "아웃카운트": ["outs"],
                "공수교대/이닝": ["inning", "is_top"],
            }
            for label, feats in groups.items():
                dp = sum(
                    delta_contrib[FEATURE_COLS.index(f)]
                    for f in feats if f in FEATURE_COLS
                ) * slope * 100
                if abs(dp) <= 1e-2:
                    continue
                text = self._friendly_factor_text(label, prev, cur, dp)
                entry = (abs(dp), f"- {text} ({label} {dp:+.1f}%p)")
                if dp > 0:
                    rising.append(entry)
                else:
                    falling.append(entry)
            rising.sort(key=lambda t: t[0], reverse=True)
            falling.sort(key=lambda t: t[0], reverse=True)
        except Exception:
            pass

        parts = [header]
        if rising:
            parts.append("📈 승률 상승 이유")
            parts.extend(t for _, t in rising[:3])
        if falling:
            parts.append("📉 승률 하락 이유")
            parts.extend(t for _, t in falling[:3])

        # feature 변화가 거의 없는 경우(예: 마지막 타석) -> 일반적인 상황 설명으로 대체
        if not rising and not falling:
            parts.extend(f"- {r}" for r in self._build_general_reason_lines(cur)[:3])

        return "\n".join(parts)

    def _friendly_factor_text(self, label, prev, cur, dp):
        """요인 그룹(label)의 타석 전→후 변화를 일반인용 한 문장으로 설명.
        dp(이 요인의 %p 영향, 부호 포함)를 함께 받아 '홈팀에 유리/불리'한
        방향을 dp 부호와 항상 일치시킨다 (상승/하락 버킷과 모순되지 않도록)."""
        favorable = dp > 0  # 홈팀 승률에 유리한 방향인지

        if label == "점수차":
            if favorable:
                return "점수차가 홈팀에게 유리해졌습니다."
            return "점수차가 홈팀에게 불리하게 바뀌었습니다."

        if label == "주자 상황":
            r_prev = prev.get("runner_1b", 0) + prev.get("runner_2b", 0) + prev.get("runner_3b", 0)
            r_cur  = cur.get("runner_1b", 0)  + cur.get("runner_2b", 0)  + cur.get("runner_3b", 0)
            if r_cur > r_prev:
                if favorable:
                    return "주자가 출루하여 득점 가능성이 높아졌습니다."
                return "상대팀 주자가 출루하여 실점 위험이 높아졌습니다."
            if r_cur < r_prev:
                if favorable:
                    return "주자가 사라져 상대팀의 득점 위험이 줄었습니다."
                return "주자가 사라져 추가 득점 기회가 줄어들었습니다."
            return "주자 상황이 바뀌었습니다."

        if label == "아웃카운트":
            outs_prev, outs_cur = prev.get("outs", 0), cur.get("outs", 0)
            if outs_cur > outs_prev:
                if favorable:
                    return "상대 공격을 막아내며 아웃카운트가 늘었습니다."
                return "아웃카운트가 증가해 득점 기회가 줄어들었습니다."
            if outs_cur < outs_prev:
                if favorable:
                    return "이닝이 바뀌며 홈팀 공격 기회가 새로 시작됐습니다."
                return "이닝이 바뀌며 상대팀 공격 기회가 시작됐습니다."
            return "공격 기회가 유지되고 있습니다."

        if label == "공수교대/이닝":
            if prev.get("is_top") != cur.get("is_top"):
                if favorable:
                    return "공격권이 홈팀으로 넘어왔습니다."
                return "공격권이 원정팀으로 넘어갔습니다."
            return "이닝이 진행되었습니다."

        return "경기 상황이 변화했습니다."

    def _build_general_reason_lines(self, cur):
        """feature 변화량이 거의 없을 때(예: 경기 종료 직후) 사용하는 일반 상황 설명"""
        reasons = []

        sd_cur = cur.get("score_diff", 0)
        if sd_cur > 0:
            reasons.append(f"홈팀이 {sd_cur}점 앞서고 있어 승률이 높게 평가됩니다.")
        elif sd_cur < 0:
            reasons.append(f"홈팀이 {abs(sd_cur)}점 뒤지고 있어 승률이 낮게 평가됩니다.")
        else:
            reasons.append("동점 상황이라 승부를 예측하기 어렵습니다.")

        if cur.get("is_top"):
            reasons.append("현재 원정팀이 공격 중이라 홈팀에게 다소 불리한 상황입니다.")
        else:
            reasons.append("현재 홈팀이 공격 중이라 유리한 상황입니다.")

        outs = cur.get("outs", 0)
        if outs >= 2:
            reasons.append("2아웃이라 공격 기회가 얼마 남지 않았습니다.")

        runners_cur = cur.get("runner_1b", 0) + cur.get("runner_2b", 0) + cur.get("runner_3b", 0)
        scoring_pos = cur.get("runner_2b") or cur.get("runner_3b")
        if runners_cur == 0:
            reasons.append("주자가 없어 추가 득점 가능성은 낮습니다.")
        elif scoring_pos:
            reasons.append("득점권에 주자가 있어 추가 득점 가능성이 높습니다.")
        else:
            reasons.append("주자가 있어 추가 득점 가능성이 있습니다.")

        disp_inning = cur.get("display_inning", cur.get("inning", 1))
        if disp_inning >= 9:
            reasons.append("경기 막판이라 승률 변화가 커질 수 있습니다.")

        seen = set()
        uniq_reasons = []
        for r in reasons:
            if r not in seen:
                uniq_reasons.append(r)
                seen.add(r)
        return uniq_reasons

    def _build_ai_summary_text(self, idx):
        """[AI 분석 요약] — 현재 타석(결과 반영 후) 기준 승률 + 핵심 이유 + 모델 판단을
        일반 야구 팬도 이해할 수 있는 자연어로 요약. 모델 입력 feature(점수차/주자/
        공수/아웃카운트/이닝)를 그대로 근거로 사용한다."""
        post_state = self._post_state(idx)
        pct = self._prob_after(idx) * 100

        lines = [f"현재 홈팀 승률은 {pct:.1f}%입니다.", ""]

        reasons = self._build_ai_summary_reasons(post_state)
        lines.append("주요 이유:")
        lines.extend(f"- {r}" for r in reasons[:3])
        lines.append("")

        lines.append("모델 판단:")
        lines.append(self._build_model_judgment(pct))

        return "\n".join(lines)

    def _build_ai_summary_reasons(self, cur):
        """현재 상태(cur) 기준 — '왜 이런 승률인지'를 설명하는 핵심 이유 목록"""
        reasons = []

        sd = cur.get("score_diff", 0)
        if sd > 0:
            reasons.append(f"{sd}점 차로 앞서고 있습니다.")
        elif sd < 0:
            reasons.append(f"{abs(sd)}점 차로 뒤지고 있습니다.")
        else:
            reasons.append("동점 상황입니다.")

        runners = cur.get("runner_1b", 0) + cur.get("runner_2b", 0) + cur.get("runner_3b", 0)
        scoring_pos = cur.get("runner_2b") or cur.get("runner_3b")
        if scoring_pos:
            reasons.append("득점권에 주자가 있어 득점 가능성이 높습니다.")
        elif runners > 0:
            reasons.append("주자가 출루하여 득점 기회가 있습니다.")
        else:
            reasons.append("주자가 없는 상황입니다.")

        if cur.get("is_top"):
            reasons.append("현재 수비 중입니다.")
        else:
            reasons.append("현재 공격 중입니다.")

        outs = cur.get("outs", 0)
        if outs >= 2:
            reasons.append("2아웃 상황으로 공격 기회가 얼마 남지 않았습니다.")

        disp_inning = cur.get("display_inning", cur.get("inning", 1))
        if disp_inning >= 9:
            reasons.append("경기 막판으로 접어들었습니다.")

        return reasons

    def _build_model_judgment(self, pct):
        """현재 홈팀 승률(pct, %)을 바탕으로 한 모델의 종합 판단 문장"""
        if pct >= 85:
            return "거의 승리가 확정적인 상황으로 분석됩니다."
        if pct >= 65:
            return "유리한 흐름이 이어지고 있는 것으로 분석됩니다."
        if pct >= 50:
            return "근소하게 앞서고 있으나 안심할 수 없는 상황으로 분석됩니다."
        if pct >= 35:
            return "아직 역전 가능성이 있으나 추가 득점이 필요한 상황으로 분석됩니다."
        if pct >= 15:
            return "불리한 흐름이지만 역전 기회가 남아있는 것으로 분석됩니다."
        return "매우 불리한 상황으로 분석됩니다."

    def _build_model_basis(self, state):
        """일반인용 — 모델이 현재 상황에서 가장 주목한 부분 2가지"""
        try:
            clf    = self.model.pipeline.named_steps["clf"]
            scaler = self.model.pipeline.named_steps["scaler"]
        except Exception:
            return "─"

        raw = np.array([state.get(c, 0) for c in FEATURE_COLS], dtype=float)
        scaled = (raw - scaler.mean_) / scaler.scale_
        contrib = scaled * clf.coef_[0]

        order = np.argsort(np.abs(contrib))[::-1][:2]
        lines = []
        for i in order:
            feat = FEATURE_COLS[i]
            lines.append("- " + _feature_friendly_line(feat, raw[i], contrib[i], state))

        return "모델이 지금 가장 주목한 부분\n" + "\n".join(lines)

    # 레이아웃 — 기여도 그래프 접기/펼치기
    def _set_contrib_visible(self, visible):
        self.contrib_visible = visible
        if visible:
            self.contrib_container.pack(fill="x", pady=(0, 16), after=self.contrib_header)
            self.btn_contrib_toggle.config(text="▼ 접기")
        else:
            self.contrib_container.pack_forget()
            self.btn_contrib_toggle.config(text="▶ 펼치기")

    def _toggle_contrib(self):
        self._set_contrib_visible(not self.contrib_visible)

    # 레이아웃 — 컴팩트(발표용) 모드
    def _toggle_compact(self):
        self.compact = not self.compact
        self._apply_compact()

    def _apply_compact(self):
        info_cards = (self.event_card, self.reason_card, self.basis_card)

        if self.compact:
            self.btn_compact.config(text="🖥 일반 모드")
            self._prob_fig_h = 2.4
            for i, card in enumerate(info_cards):
                card.grid_forget()
                card.grid(row=i, column=0, columnspan=3, sticky="nsew", padx=4, pady=2)
            self._set_contrib_visible(False)
        else:
            self.btn_compact.config(text="🗜 컴팩트 모드")
            self._prob_fig_h = 3.4
            for i, card in enumerate(info_cards):
                card.grid_forget()
                card.grid(row=0, column=i, sticky="nsew", padx=4)
            self._set_contrib_visible(True)

        self.canvas_prob.get_tk_widget().configure(height=int(self._prob_fig_h * 100))
        w = self.fig_prob.get_size_inches()[0]
        self.fig_prob.set_size_inches(w, self._prob_fig_h)
        self.canvas_prob.draw_idle()

        self._update_highlight_cards()

    # 창 크기 변경 -> 승률 그래프 높이 자동 조절 (컴팩트 모드에서는 고정)
    def _on_window_configure(self, event):
        if event.widget is not self:
            return
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._apply_auto_graph_height)

    def _apply_auto_graph_height(self):
        self._resize_job = None
        if self.compact:
            return
        win_h = self.winfo_height()
        new_h = max(2.4, min(3.8, win_h / 280))
        if abs(new_h - self._prob_fig_h) >= 0.12:
            self._prob_fig_h = new_h
            self.canvas_prob.get_tk_widget().configure(height=int(new_h * 100))
            w = self.fig_prob.get_size_inches()[0]
            self.fig_prob.set_size_inches(w, new_h)
            self.canvas_prob.draw_idle()

    # 현재 경기 상황 카드 — feature 원본 값 [상세 보기] 토글
    def _toggle_feature_detail(self):
        self.feature_detail_visible = not self.feature_detail_visible
        if self.feature_detail_visible:
            self.feature_detail_lbl.pack(anchor="w", fill="x", padx=14, pady=(0, 10))
            self.btn_feat_detail.config(text="🔧 상세 숨기기")
        else:
            self.feature_detail_lbl.pack_forget()
            self.btn_feat_detail.config(text="🔧 상세 보기")

    # 디버그 정보 — 승률 변화 원인 분석용 (수정 없이 정보만 표시)
    def _toggle_debug(self):
        self.debug_visible = not self.debug_visible
        if self.debug_visible:
            self.debug_card.pack(fill="x", pady=(0, 12), before=self.contrib_header)
            self.btn_debug.config(text="🐞 디버그 정보 숨기기 (D)")
        else:
            self.debug_card.pack_forget()
            self.btn_debug.config(text="🐞 디버그 정보 (D)")

    def _build_debug_text(self, idx):
        """선택된 타석에 대한 모델 입력값 / 전후 상태 / 승률 계산 근거를 모두 출력 (분석용)"""
        if not self.all_states:
            return "-"

        state = self.all_states[idx - 1]
        rbi = state.get("rbi", 0) or 0
        runs = rbi if state.get("is_scoring_play") else 0

        lines = []

        # [1] 모델 입력값
        lines.append("[1] 모델 입력값 (predict_proba에 실제 사용된 feature)")
        for c in FEATURE_COLS:
            lines.append(f"    {c:<12} = {state.get(c, 0)}")
        lines.append(f"    {'balls':<12} = {state.get('balls', 0)}   (※ 모델 입력 아님 — 표시용)")
        lines.append(f"    {'strikes':<12} = {state.get('strikes', 0)}   (※ 모델 입력 아님 — 표시용)")
        lines.append("")

        # [2] 타석 전 상태
        pre_away = state.get("away_score", 0)
        pre_home = state.get("home_score", 0)
        pre_outs = state.get("outs", 0)
        pre_runners = _runners_text(state)
        lines.append("[2] 타석 전 상태")
        lines.append(f"    점수 (원정:홈) {pre_away} : {pre_home}")
        lines.append(f"    아웃 {pre_outs}")
        lines.append(f"    주자 {pre_runners}")
        lines.append("")

        # [3] 타석 결과
        lines.append("[3] 타석 결과")
        lines.append(f"    결과: {_event_kr(state)}")
        lines.append(f"    타점(RBI): {rbi}")
        lines.append(f"    득점: {runs}")
        lines.append("")

        # [4] 타석 후 상태
        post_away, post_home = self._post_score(idx)
        post_state = self._post_state(idx)
        lines.append("[4] 타석 후 상태")
        lines.append(f"    점수 (원정:홈) {post_away} : {post_home}")
        if idx < len(self.all_states):
            lines.append(f"    아웃 {post_state.get('outs', 0)}   (※ 이닝 교체 시 0으로 초기화될 수 있음)")
            lines.append(f"    주자 {_runners_text(post_state)}")
        else:
            lines.append("    아웃 -   (경기 종료 — 다음 타석 정보 없음)")
            lines.append("    주자 -   (경기 종료 — 다음 타석 정보 없음)")
        lines.append("")

        # [5] 화면 표시에 사용되는 상태 / 승률 매핑
        idx0 = idx - 1  # 0-indexed (states/win_probs 배열 기준)
        prob_before = self._prob_before(idx) * 100
        prob_after = self._prob_after(idx) * 100
        lines.append("[5] 화면 표시에 사용되는 상태 / 승률 매핑")
        lines.append(f"    현재 타석 전 상태 : states[{idx0}]              (모델 입력 → win_probs[{idx0}])")
        lines.append(f"    현재 타석 결과     : states[{idx0}][\"event\"] = {state.get('event', '-')}")
        if idx < len(self.all_states):
            lines.append(f"    타석 후 상태       : states[{idx0 + 1}]")
            lines.append(f"    타석 전 승률       : win_probs[{idx0}]   = {prob_before:.1f}%")
            lines.append(f"    타석 후 승률       : win_probs[{idx0 + 1}] = {prob_after:.1f}%")
        else:
            lines.append("    타석 후 상태       : 없음 (경기 종료 — 다음 타석 없음)")
            lines.append(f"    타석 전 승률       : win_probs[{idx0}]   = {prob_before:.1f}%")
            lines.append(f"    타석 후 승률       : 없음 → '다음 상태 없음', 타석 전 승률과 동일하게 처리 ({prob_after:.1f}%)")
        lines.append("")
        lines.append(f"    ※ 화면의 '#{idx} 결과'에는 위 '타석 후 승률'이 표시됩니다.")
        lines.append("       (이벤트 idx의 효과 = win_probs[idx] → win_probs[idx+1] 변화)")
        lines.append("")

        # [6] 승률 계산 근거
        delta = prob_after - prob_before
        lines.append("[6] 승률 계산 근거")
        lines.append(f"    타석 전 승률: {prob_before:.1f}%")
        lines.append(f"    타석 후 승률: {prob_after:.1f}%")
        lines.append(f"    변화: {delta:+.1f}%p")
        lines.append("")
        lines.append("    주요 원인 (근사치 — 로지스틱 회귀 계수 기반 선형 근사, 타석 전 → 타석 후 변화 기준)")
        try:
            clf    = self.model.pipeline.named_steps["clf"]
            scaler = self.model.pipeline.named_steps["scaler"]
            raw_prev = np.array([state.get(c, 0) for c in FEATURE_COLS], dtype=float)
            raw_cur  = np.array([post_state.get(c, 0) for c in FEATURE_COLS], dtype=float)
            scaled_prev = (raw_prev - scaler.mean_) / scaler.scale_
            scaled_cur  = (raw_cur  - scaler.mean_) / scaler.scale_
            contrib_prev = scaled_prev * clf.coef_[0]
            contrib_cur  = scaled_cur  * clf.coef_[0]
            delta_contrib = contrib_cur - contrib_prev

            p = self.all_probs[idx - 1]
            slope = p * (1 - p)  # dp/dz (로지스틱 미분, 선형 근사)

            groups = {
                "이닝/공수교대 변화": ["inning", "is_top"],
                "아웃카운트 변화": ["outs"],
                "점수차 변화": ["score_diff"],
                "주자 상황 변화": ["runner_1b", "runner_2b", "runner_3b"],
            }
            any_line = False
            for label, feats in groups.items():
                dp = sum(
                    delta_contrib[FEATURE_COLS.index(f)]
                    for f in feats if f in FEATURE_COLS
                ) * slope * 100
                if abs(dp) > 1e-3:
                    lines.append(f"    - {label}: {dp:+.1f}%p")
                    any_line = True
            if not any_line:
                lines.append("    - (모델 입력값 변화 없음 — 마지막 타석 등)")
        except Exception as e:
            lines.append(f"    (계산 불가: {e})")
        lines.append("")

        # [7] 모델 학습 feature 목록
        lines.append("[7] 모델 학습 feature 목록 (FEATURE_COLS — 실제 학습/예측에 사용됨)")
        for c in FEATURE_COLS:
            lines.append(f"    - {c}")
        lines.append("")
        lines.append("    기타 필드 포함 여부:")
        for extra in ["event", "runs_scored", "rbi", "balls", "strikes"]:
            included = "포함됨 (학습 사용)" if extra in FEATURE_COLS else "미포함 (표시용 데이터일 뿐, 학습/예측에는 사용되지 않음)"
            lines.append(f"    - {extra}: {included}")

        return "\n".join(lines)

    # 이닝별 득점표 (R H E)
    def set_linescore(self, linescore, states=None):
        """외부(main.py)에서 이닝별 득점표(R H E) 데이터 갱신"""
        self._linescore = linescore
        self._draw_linescore()

    def _draw_linescore(self):
        c = self.linescore_canvas
        c.delete("all")

        linescore = self._linescore
        states = self.all_states

        innings_data = (linescore or {}).get("innings", [])
        max_inn_from_states = max(
            (s.get("display_inning", s.get("inning", 0)) for s in states), default=0)

        if not linescore or (not innings_data and max_inn_from_states == 0):
            c.config(height=40)
            c.create_text(10, 20, text="이닝별 득점표 — 데이터 없음", fill=SUBTEXT,
                           font=("Arial", 11), anchor="w")
            return

        home_totals = linescore.get("teams", {}).get("home", {})
        away_totals = linescore.get("teams", {}).get("away", {})

        away_name = (states[0].get("away_team", "원정") if states else "원정")[:14]
        home_name = (states[0].get("home_team", "홈") if states else "홈")[:14]

        n_inn  = max(len(innings_data), max_inn_from_states, 9)
        col_w  = 30
        name_w = 130
        pad_x  = 8
        row_h  = 26
        hdr_h  = 22

        total_w = name_w + (n_inn + 3) * col_w + pad_x * 2
        total_h = hdr_h + row_h * 2 + 6
        c.config(height=total_h, scrollregion=(0, 0, total_w, total_h))

        def cx(col):
            return name_w + col * col_w + col_w // 2

        def draw_cell(col, row, text, fg=TEXT, bold=False, special=False):
            x = cx(col)
            y = hdr_h + row * row_h + row_h // 2
            if special:
                c.create_rectangle(x - col_w // 2 + 1, y - row_h // 2 + 1,
                                    x + col_w // 2 - 1, y + row_h // 2 - 1,
                                    fill=DIMCARD, outline="")
            fnt = ("Arial", 11, "bold") if bold else ("Arial", 11)
            c.create_text(x, y, text=str(text), fill=fg, font=fnt, anchor="center")

        # 헤더: 이닝 번호 + R H E
        for i in range(n_inn):
            fg_inn = YELLOW if i >= 9 else SUBTEXT
            c.create_text(cx(i), hdr_h // 2, text=str(i + 1),
                           fill=fg_inn, font=("Arial", 10, "bold" if i >= 9 else "normal"),
                           anchor="center")
        for j, lbl in enumerate(["R", "H", "E"]):
            c.create_text(cx(n_inn + j), hdr_h // 2, text=lbl,
                           fill=SUBTEXT, font=("Arial", 10, "bold"), anchor="center")

        # 팀 이름
        for row, name in enumerate([away_name, home_name]):
            y = hdr_h + row * row_h + row_h // 2
            c.create_text(pad_x + 4, y, text=name, fill=TEXT,
                           font=("Arial", 11, "bold"), anchor="w")

        # 이닝별 득점
        for i, inn in enumerate(innings_data):
            a_r = inn.get("away", {}).get("runs", "")
            h_r = inn.get("home", {}).get("runs", "")
            a_fg = MINT if isinstance(a_r, int) and a_r > 0 else TEXT
            h_fg = MINT if isinstance(h_r, int) and h_r > 0 else TEXT
            draw_cell(i, 0, a_r if a_r != "" else "-", fg=a_fg)
            draw_cell(i, 1, h_r if h_r != "" else "-", fg=h_fg)

        for i in range(len(innings_data), n_inn):
            draw_cell(i, 0, "-", fg=SUBTEXT)
            draw_cell(i, 1, "-", fg=SUBTEXT)

        # R H E 합계 칸
        for row, totals in enumerate([away_totals, home_totals]):
            r = totals.get("runs", 0)
            h = totals.get("hits", 0)
            e = totals.get("errors", 0)
            draw_cell(n_inn,     row, r, fg=TEXT,    bold=True, special=True)
            draw_cell(n_inn + 1, row, h, fg=SUBTEXT, special=True)
            draw_cell(n_inn + 2, row, e,
                      fg=RED if isinstance(e, int) and e > 0 else SUBTEXT,
                      special=True)

        c.create_line(0, hdr_h, total_w, hdr_h, fill=BORDER, width=1)

    # 경기 종료 후 MVP 자동 선정
    def _set_mvp_visible(self, visible):
        self.mvp_visible = visible
        if visible:
            self.mvp_card.pack(fill="x", pady=(0, 12), before=self.highlight_title_lbl)
        else:
            self.mvp_card.pack_forget()

    def _update_mvp(self, idx):
        # 경기가 끝까지 진행된 경우(마지막 타석)에만 MVP 표시
        if not self.all_states or idx != len(self.all_states) or len(self.all_states) < 2:
            self._set_mvp_visible(False)
            return

        final_state = self.all_states[-1]
        home_wins = final_state.get("home_wins")
        if home_wins is None:
            self._set_mvp_visible(False)
            return

        # 승리팀에 유리한 방향으로 가장 크게 승률을 움직인 장면을 찾는다
        # idx번 타석의 효과 = win_probs[idx] - win_probs[idx-1]
        best = None
        for idx in range(1, len(self.all_probs)):
            delta = (self.all_probs[idx] - self.all_probs[idx - 1]) * 100
            favors_home = delta > 0
            if (favors_home and home_wins == 1) or (not favors_home and home_wins == 0):
                if best is None or abs(delta) > abs(best[1]):
                    best = (idx, delta)

        if best is None:
            self._set_mvp_visible(False)
            return

        mvp_idx, mvp_delta = best
        state = self.all_states[mvp_idx - 1]
        batter = state.get("batter") or "선수"
        event_kr = _event_kr(state)
        inning_line = _inning_half_kr(state)
        winner = final_state.get("home_team") if home_wins else final_state.get("away_team")

        self.mvp_lbl.config(
            text=(
                f"🏆 경기 MVP: {batter}\n"
                f"{inning_line} — {event_kr}  (승률 {abs(mvp_delta):.1f}%p 변화)\n"
                f"{winner}의 승리에 결정적인 활약을 펼쳤습니다."
            )
        )
        self._set_mvp_visible(True)

    # 편의 기능 — 타석 검색 / 최고 상승·하락 장면 바로가기
    def _on_search_jump(self):
        text = self.search_entry.get().strip().lstrip("#")
        if not text.isdigit():
            return
        idx = int(text)
        if 1 <= idx <= len(self.all_states):
            self._jump_to(idx)

    def _jump_to_best_rise(self):
        rises = [h for h in self.highlights if h[1] > 0]
        if not rises:
            return
        idx, _ = max(rises, key=lambda t: t[1])
        self._jump_to(idx)

    def _jump_to_best_drop(self):
        drops = [h for h in self.highlights if h[1] < 0]
        if not drops:
            return
        idx, _ = min(drops, key=lambda t: t[1])
        self._jump_to(idx)

    # 키보드 단축키 도움말 / 전체화면
    def _show_help(self):
        if getattr(self, "_help_win", None) and self._help_win.winfo_exists():
            self._help_win.lift()
            return

        win = tk.Toplevel(self)
        win.title("⌨ 단축키 도움말")
        win.configure(bg=CARD)
        win.resizable(False, False)
        self._help_win = win

        tk.Label(win, text="⌨ 키보드 단축키", bg=CARD, fg=TEXT,
                 font=("Arial", 13, "bold")).pack(padx=24, pady=(18, 8), anchor="w")

        shortcuts = [
            ("← / →", "이전 / 다음 타석으로 이동"),
            ("Home / End", "첫 타석 / 마지막 타석으로 이동"),
            ("Space", "재생 / 일시정지"),
            ("H", "이 도움말 열기"),
            ("F11", "전체화면 전환"),
            ("Esc", "전체화면 종료"),
        ]
        for key, desc in shortcuts:
            row = tk.Frame(win, bg=CARD)
            row.pack(fill="x", padx=24, pady=2)
            tk.Label(row, text=key, bg=CARD, fg=MINT, font=("Consolas", 11, "bold"),
                     width=12, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg=CARD, fg=MIDTEXT, font=("Arial", 11),
                     anchor="w").pack(side="left")

        tk.Button(win, text="닫기", command=win.destroy, bg=CARD2, fg=TEXT,
                  relief="flat", bd=0, padx=16, pady=7, cursor="hand2",
                  font=("Arial", 10, "bold")).pack(pady=16)

    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.attributes("-fullscreen", self._is_fullscreen)
        self.btn_fullscreen.config(
            text="⛶ 창모드 (F11)" if self._is_fullscreen else "⛶ 전체화면 (F11)")

    def _exit_fullscreen(self):
        if self._is_fullscreen:
            self._is_fullscreen = False
            self.attributes("-fullscreen", False)
            self.btn_fullscreen.config(text="⛶ 전체화면 (F11)")

    def _handle_close(self):
        self._stop_play()
        if self._on_close:
            self._on_close()
        self.destroy()
