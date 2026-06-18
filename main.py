"""
MLB 실시간 승리 확률 예측 시스템 — ruta 스타일 GUI
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import traceback
import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# PIL/Pillow (선택)
try:
    from PIL import Image, ImageTk, ImageDraw
    _PIL = True
except ImportError:
    _PIL = False

# 한글 폰트 자동 감지 (macOS 우선, Windows/Linux 순)
_avail_fonts = {f.name for f in _fm.fontManager.ttflist}
for _pf in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Arial Unicode MS"]:
    if _pf in _avail_fonts:
        matplotlib.rcParams["font.family"] = _pf
        break
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

from tkcalendar import Calendar
from src.collector import get_schedule, get_schedule_kst, get_game_feed, get_current_at_bat, is_game_live, get_standings
from src.features import extract_game_states
from src.model import WinProbabilityModel
from src.ml_debug_window import MLDebugWindow

# Pregame 예측 모듈 (선택적 — 파일 없어도 기존 기능 정상 동작)
try:
    from src.team_stats import collect_pregame_data
    from src.pregame_model import PregamePredictor
    _PREGAME_OK = True
except ImportError as _pg_err:
    print(f"  [pregame] 모듈 로드 실패 (기능 비활성화): {_pg_err}")
    _PREGAME_OK = False

# 마우스 휠 / 트랙패드 스크롤 헬퍼
def _make_wheel_scroll(canvas):
    """canvas를 1 unit씩 부드럽게 스크롤하는 핸들러를 반환.
    Mac 트랙패드의 큰 delta 값도 방향만 보고 1 unit으로 고정해
    스크롤이 너무 빠르거나 튀지 않게 하고, 이벤트 중복 전파를 막는다."""
    def _handler(event):
        canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"
    return _handler


# ruta 색상 팔레트
BG      = "#0f0f0f"
PANEL   = "#1a1a1a"
CARD    = "#222222"
CARD2   = "#2c2c2c"
MINT    = "#00d4a0"   # 민트 그린 (primary) — 약간 밝게
RED     = "#ff5555"
YELLOW  = "#ffb020"
TEXT    = "#f0f0f0"
SUBTEXT = "#909090"   # 기존 #888 → 더 밝게
MIDTEXT = "#c0c0c0"   # 중간 강조 텍스트 (신규)
BORDER  = "#333333"
ACCENT  = MINT
GREEN   = MINT

LIVE_POLL_INTERVAL = 20_000  # 20초

# MLB 팀 한글명 매핑
_KR_TEAM = {
    # AL East
    "Baltimore Orioles":      "볼티모어 오리올스",
    "Boston Red Sox":         "보스턴 레드삭스",
    "New York Yankees":       "뉴욕 양키스",
    "Tampa Bay Rays":         "탬파베이 레이스",
    "Toronto Blue Jays":      "토론토 블루제이스",
    # AL Central
    "Chicago White Sox":      "시카고 화이트삭스",
    "Cleveland Guardians":    "클리블랜드 가디언즈",
    "Detroit Tigers":         "디트로이트 타이거스",
    "Kansas City Royals":     "캔자스시티 로열스",
    "Minnesota Twins":        "미네소타 트윈스",
    # AL West
    "Houston Astros":         "휴스턴 애스트로스",
    "Los Angeles Angels":     "LA 에인절스",
    "Oakland Athletics":      "오클랜드 애슬레틱스",
    "Athletics":              "애슬레틱스",
    "Seattle Mariners":       "시애틀 매리너스",
    "Texas Rangers":          "텍사스 레인저스",
    # NL East
    "Atlanta Braves":         "애틀랜타 브레이브스",
    "Miami Marlins":          "마이애미 말린스",
    "New York Mets":          "뉴욕 메츠",
    "Philadelphia Phillies":  "필라델피아 필리스",
    "Washington Nationals":   "워싱턴 내셔널스",
    # NL Central
    "Chicago Cubs":           "시카고 컵스",
    "Cincinnati Reds":        "신시내티 레즈",
    "Milwaukee Brewers":      "밀워키 브루어스",
    "Pittsburgh Pirates":     "피츠버그 파이러츠",
    "St. Louis Cardinals":    "세인트루이스 카디널스",
    # NL West
    "Arizona Diamondbacks":   "애리조나 다이아몬드백스",
    "Colorado Rockies":       "콜로라도 로키스",
    "Los Angeles Dodgers":    "LA 다저스",
    "San Diego Padres":       "샌디에이고 파드리스",
    "San Francisco Giants":   "샌프란시스코 자이언츠",
}

# 팀명 → 약칭 (로고 파일명에 사용)
_TEAM_ABBR = {
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "New York Yankees": "NYY", "Tampa Bay Rays": "TB",
    "Toronto Blue Jays": "TOR", "Chicago White Sox": "CWS",
    "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KC", "Minnesota Twins": "MIN",
    "Houston Astros": "HOU", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Athletics": "OAK",
    "Seattle Mariners": "SEA", "Texas Rangers": "TEX",
    "Atlanta Braves": "ATL", "Miami Marlins": "MIA",
    "New York Mets": "NYM", "Philadelphia Phillies": "PHI",
    "Washington Nationals": "WSH", "Chicago Cubs": "CHC",
    "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL",
    "Pittsburgh Pirates": "PIT", "St. Louis Cardinals": "STL",
    "Arizona Diamondbacks": "ARI", "Colorado Rockies": "COL",
    "Los Angeles Dodgers": "LAD", "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
}

def _team_abbr(name: str) -> str:
    return _TEAM_ABBR.get(name, name.split()[-1][:3].upper())

def team_kr(name: str) -> str:
    """영어 팀명 → 한국어만 반환. 매핑 없으면 원문."""
    return _KR_TEAM.get(name, name)

def _fmt_team(name: str) -> str:
    """'Los Angeles Dodgers' → 'LA 다저스(Dodgers)' 형식 (스코어보드용)"""
    kr = _KR_TEAM.get(name)
    if not kr:
        return name
    short_en = name.split()[-1]
    return f"{kr}({short_en})"

# 이벤트명 한국어 매핑
_EVENT_KR = {
    "Home Run":            "홈런",
    "Grand Slam":          "그랜드슬램",
    "Single":              "안타",
    "Double":              "2루타",
    "Triple":              "3루타",
    "Walk":                "볼넷",
    "Intent Walk":         "고의4구",
    "Intentional Walk":    "고의4구",
    "Hit By Pitch":        "몸에 맞는 공",
    "Strikeout":           "삼진",
    "Strikeout - DP":      "삼진",
    "Field Out":           "뜬공/땅볼 아웃",
    "Groundout":           "땅볼 아웃",
    "Flyout":              "뜬공 아웃",
    "Lineout":             "직선타 아웃",
    "Pop Out":             "내야뜬공 아웃",
    "Forceout":            "포스 아웃",
    "Field Error":         "실책",
    "Sac Fly":             "희생플라이",
    "Sac Bunt":            "희생번트",
    "Double Play":         "병살타",
    "Grounded Into DP":    "병살타",
    "Triple Play":         "삼중살",
    "Runner Out":          "주자 아웃",
    "Pickoff":             "견제 아웃",
    "Caught Stealing":     "도루 실패",
    "Stolen Base":         "도루",
    "Wild Pitch":          "폭투",
    "Passed Ball":         "포일",
    "Balk":                "보크",
    "Catcher Interference":"포수 방해",
    "Fan Interference":    "관중 방해",
}

def _disp_inn(s) -> int:
    """표시용 실제 이닝 반환 (display_inning 우선, 없으면 inning fallback)"""
    return s.get("display_inning", s.get("inning", 0))

def event_kr(event: str) -> str:
    """영어 이벤트명 → 한국어. 정확히 일치하면 반환, 없으면 부분 매칭 시도."""
    if not event:
        return event
    if event in _EVENT_KR:
        return _EVENT_KR[event]
    for en, kr in _EVENT_KR.items():
        if en.lower() in event.lower():
            return kr
    return event


# ═══════════════════════════════════════════════════════════════════════════
class MLBApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLB 실시간 승리 확률 예측 시스템")
        self.geometry("1340x860")
        self.configure(bg=BG)
        self.resizable(True, True)

        # 상태
        self.model        = WinProbabilityModel()
        self.model_ready  = False
        self.game_states  = []
        self.win_probs    = []
        self.anim_job     = None
        self.anim_idx     = 0
        self.anim_speed   = 500
        self._game_data   = []
        self._linescore   = None
        self._last_feed   = None

        # ML 실시간 분석 창
        self._ml_debug_win = None

        # 라이브 모드
        self.live_mode       = False
        self.live_poll_job   = None
        self.current_game_pk = None
        self._blink_on       = False

        # 현재 활성 탭
        self._active_tab = tk.StringVar(value="요약")

        # 팀명
        self._away_team_name = "원정팀"
        self._home_team_name = "홈팀"

        # 선발 투수
        self._away_pitcher = None
        self._home_pitcher = None

        # 이미지 캐시 (GC 방지)
        self._img_cache = {}

        # Pregame 예측 상태
        self._pregame_result   = None   # 마지막 예측 결과
        self._pregame_loading  = False  # 로딩 중 플래그
        if _PREGAME_OK:
            self._pregame = PregamePredictor()

        self._try_load_model()
        self._build_ui()
        self._set_today()

    # 초기화

    def _try_load_model(self):
        try:
            self.model.load()
            self.model_ready = True
        except FileNotFoundError:
            pass

    _KST = datetime.timezone(datetime.timedelta(hours=9))

    @staticmethod
    def _kst_today():
        return datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=9))
        ).date()

    def _set_today(self):
        self.date_var.set(self._kst_today().strftime("%Y-%m-%d"))

    # 이미지 로딩

    def _load_team_logo(self, team_name: str, size: int = 52):
        """assets/logos/{ABBR}.png 로드. 없으면 None 반환 + 약칭 텍스트 반환."""
        abbr = _team_abbr(team_name)
        path = os.path.join("assets", "logos", f"{abbr}.png")
        if _PIL and os.path.exists(path):
            try:
                img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._img_cache[f"logo_{abbr}_{size}"] = photo
                return photo, abbr
            except Exception:
                pass
        return None, abbr

    def _load_player_image(self, name: str, size: int = 90):
        """assets/players/{name}.png 로드, 원형 마스크 적용."""
        if not name:
            return None, False
        path = os.path.join("assets", "players", f"{name}.png")
        if _PIL and os.path.exists(path):
            try:
                img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
                # 원형 마스크
                mask = Image.new("L", (size, size), 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, size, size), fill=255)
                result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                result.paste(img, mask=mask)
                photo = ImageTk.PhotoImage(result)
                self._img_cache[f"player_{name}_{size}"] = photo
                return photo, True
            except Exception:
                pass
        return None, False

    def _draw_logo_canvas(self, canvas, team_name: str, size: int):
        """Canvas에 팀 로고를 그림. 로고 없으면 약칭 텍스트 fallback."""
        photo, abbr = self._load_team_logo(team_name, size)
        canvas.delete("all")
        if photo:
            canvas.create_image(size // 2, size // 2, image=photo)
        else:
            canvas.create_text(size // 2, size // 2, text=abbr,
                               fill=TEXT, font=("Arial", 14, "bold"))

    # 최상위 UI 뼈대

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_left(body)
        self._build_right(body)

    # 헤더

    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        # 헤더 하단 구분선
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        tk.Label(hdr, text="⚾  MLB Win Probability",
                 bg=PANEL, fg=TEXT, font=("Arial", 17, "bold")).pack(side="left", padx=20)

        # 순위 버튼 — 테두리형으로 더 눈에 띄게
        standings_btn = tk.Button(
            hdr, text="📊  순위",
            bg=CARD2, fg=MINT,
            activebackground=CARD, activeforeground=MINT,
            font=("Arial", 12, "bold"),
            relief="flat", bd=0, padx=14, pady=6,
            cursor="hand2",
            command=self._open_standings,
        )
        standings_btn.pack(side="left", padx=(10, 0))

        self.status_lbl = tk.Label(
            hdr,
            text="● 모델 로드 완료" if self.model_ready else "● 모델 없음 (train.py 실행 필요)",
            bg=PANEL,
            fg=MINT if self.model_ready else YELLOW,
            font=("Arial", 12),
        )
        self.status_lbl.pack(side="right", padx=20)

    # ═══════════════════════ 왼쪽 패널 ════════════════════════════════════

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=PANEL, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.pack_propagate(False)

        self._build_date_section(left)
        self._build_game_list(left)
        self._build_action_buttons(left)

    # 날짜 선택

    def _build_date_section(self, parent):
        self._lbl(parent, "날짜 선택", padtop=16)

        self.date_var = tk.StringVar()

        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=12, pady=4)

        self.date_display = tk.Label(
            row, textvariable=self.date_var,
            bg=CARD, fg=TEXT, font=("Arial", 14, "bold"),
            padx=10, pady=8, anchor="w", cursor="hand2", relief="flat",
        )
        self.date_display.pack(side="left", fill="x", expand=True)
        self.date_display.bind("<Button-1>", lambda e: self._open_calendar())

        self._mkbtn(row, "📅", self._open_calendar, CARD2, fg=MINT, w=3).pack(side="left", padx=(4, 0))
        self._mkbtn(row, "조회", self._fetch_games, MINT, w=5).pack(side="left", padx=(4, 0))

        # 빠른 날짜 버튼 — 테두리·폰트 개선으로 가독성 향상
        qrow = tk.Frame(parent, bg=PANEL)
        qrow.pack(fill="x", padx=12, pady=(4, 6))
        for lbl, delta in [("내일", 1), ("오늘", 0), ("어제", -1)]:
            d = (self._kst_today() + datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
            tk.Button(
                qrow, text=lbl,
                bg=CARD, fg=MIDTEXT,
                activebackground=CARD2, activeforeground=TEXT,
                font=("Arial", 11, "bold"), relief="flat", bd=0,
                padx=12, pady=6, cursor="hand2",
                command=lambda v=d: (self.date_var.set(v), self._fetch_games()),
            ).pack(side="left", padx=(0, 4))

    # 경기 목록

    def _build_game_list(self, parent):
        self._lbl(parent, "경기 목록", padtop=12)

        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="both", expand=True, padx=12)

        self.games_canvas = tk.Canvas(frame, bg=PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.games_canvas.yview)
        self.games_canvas.configure(yscrollcommand=sb.set)
        self.games_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.games_inner = tk.Frame(self.games_canvas, bg=PANEL)
        self._games_window = self.games_canvas.create_window(
            (0, 0), window=self.games_inner, anchor="nw"
        )
        self.games_inner.bind(
            "<Configure>",
            lambda e: self.games_canvas.configure(
                scrollregion=self.games_canvas.bbox("all")
            ),
        )
        self.games_canvas.bind(
            "<Configure>",
            lambda e: self.games_canvas.itemconfig(
                self._games_window, width=e.width
            ),
        )
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>",   self._on_mousewheel)
        self.bind_all("<Button-5>",   self._on_mousewheel)

        self._selected_game_idx = None
        self._game_card_frames  = []

    def _on_mousewheel(self, event):
        # 다른 창(Toplevel)에 떠 있는 위젯 위에서는 무시 (전역 bind_all 중복 방지)
        if event.widget.winfo_toplevel() is not self:
            return

        cx = self.games_canvas.winfo_rootx()
        cy = self.games_canvas.winfo_rooty()
        cw = self.games_canvas.winfo_width()
        ch = self.games_canvas.winfo_height()
        px = self.winfo_pointerx()
        py = self.winfo_pointery()
        if not (cx <= px <= cx + cw and cy <= py <= cy + ch):
            return

        if event.num == 4:
            self.games_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.games_canvas.yview_scroll(1, "units")
        else:
            direction = -1 if event.delta > 0 else 1
            self.games_canvas.yview_scroll(direction, "units")
        return "break"

    # 액션 버튼

    def _build_action_buttons(self, parent):
        # 구분선
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(8, 0))

        self._mkbtn(parent, "▶  경기 분석 시작", self._analyze_game, MINT).pack(
            fill="x", padx=12, pady=(10, 6)
        )
        self.live_btn = tk.Button(
            parent, text="🔴  LIVE 중계 시작",
            command=self._toggle_live,
            bg="#3a1212", fg="#ff8080",
            activebackground="#4a1818", activeforeground=TEXT,
            font=("Arial", 12, "bold"),
            relief="flat", bd=0, padx=10, pady=9, cursor="hand2",
        )
        self.live_btn.pack(fill="x", padx=12, pady=(0, 6))

        self._mkbtn(parent, "🧠  ML 분석 보기", self._toggle_ml_debug, CARD2, fg=MINT).pack(
            fill="x", padx=12, pady=(0, 14)
        )

    # ═══════════════════════ 오른쪽 패널 ══════════════════════════════════

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_scoreboard(right)
        self._build_live_panel(right)
        self._build_tab_area(right)
        self._build_pregame_overlay(right)   # 탭과 같은 row=2 에 오버레이
        # 초기 상태: 오버레이 숨김
        self._pregame_overlay.grid_remove()

    # 스코어보드 (카드형)

    def _build_scoreboard(self, parent):
        self.score_frame = tk.Frame(parent, bg=CARD)
        self.score_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # 원정팀 영역
        self._sb_away = tk.Frame(self.score_frame, bg=CARD)
        self._sb_away.pack(side="left", expand=True, fill="both", padx=(20, 0), pady=10)
        self._build_sb_team_side(self._sb_away, "away")

        # 중앙 구분
        mid = tk.Frame(self.score_frame, bg=CARD)
        mid.pack(side="left", padx=20)
        self._sb_vs_lbl = tk.Label(mid, text="vs", bg=CARD, fg=SUBTEXT,
                                    font=("Arial", 16))
        self._sb_vs_lbl.pack()
        self._sb_status_lbl = tk.Label(mid, text="", bg=CARD, fg=SUBTEXT,
                                        font=("Arial", 11))
        self._sb_status_lbl.pack(pady=(4, 0))

        # 홈팀 영역
        self._sb_home = tk.Frame(self.score_frame, bg=CARD)
        self._sb_home.pack(side="left", expand=True, fill="both", padx=(0, 20), pady=10)
        self._build_sb_team_side(self._sb_home, "home")

    def _build_sb_team_side(self, parent, side):
        """팀 한쪽 영역: 로고 + 점수 + 팀명 + 투수"""
        # 로고 캔버스
        logo_c = tk.Canvas(parent, width=52, height=52, bg=CARD,
                            highlightthickness=0)
        logo_c.pack(anchor="center")
        logo_c.create_text(26, 26, text="⚾", fill=SUBTEXT,
                            font=("Arial", 20))

        # 점수
        score_lbl = tk.Label(parent, text="─", bg=CARD, fg=TEXT,
                              font=("Arial", 36, "bold"))
        score_lbl.pack(anchor="center")

        # 팀명
        name_lbl = tk.Label(parent, text="팀명", bg=CARD, fg=TEXT,
                             font=("Arial", 13, "bold"), wraplength=180)
        name_lbl.pack(anchor="center")

        # 투수
        pitcher_lbl = tk.Label(parent, text="", bg=CARD, fg=SUBTEXT,
                                font=("Arial", 11))
        pitcher_lbl.pack(anchor="center", pady=(2, 0))

        # 경기 전 예상 승률 (평소에는 숨김)
        prob_lbl = tk.Label(parent, text="", bg=CARD, fg=MINT,
                             font=("Arial", 20, "bold"))
        prob_lbl.pack(anchor="center", pady=(0, 2))

        setattr(self, f"_sb_{side}_logo",    logo_c)
        setattr(self, f"_sb_{side}_score",   score_lbl)
        setattr(self, f"_sb_{side}_name",    name_lbl)
        setattr(self, f"_sb_{side}_pitcher", pitcher_lbl)
        setattr(self, f"_sb_{side}_prob",    prob_lbl)

    # LIVE 패널

    def _build_live_panel(self, parent):
        self.live_panel = tk.Frame(parent, bg=CARD2, height=86)
        self.live_panel.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.live_panel.grid_remove()
        self.live_panel.pack_propagate(False)

        info = tk.Frame(self.live_panel, bg=CARD2)
        info.pack(side="left", padx=16, pady=8)
        self.lv_inning_lbl  = tk.Label(info, text="─ 회 ─", bg=CARD2, fg=MINT,
                                        font=("Arial", 13, "bold"))
        self.lv_inning_lbl.pack(anchor="w")
        self.lv_batter_lbl  = tk.Label(info, text="타자: ─", bg=CARD2, fg=TEXT,
                                        font=("Arial", 12))
        self.lv_batter_lbl.pack(anchor="w")
        self.lv_pitcher_lbl = tk.Label(info, text="투수: ─", bg=CARD2, fg=TEXT,
                                        font=("Arial", 12))
        self.lv_pitcher_lbl.pack(anchor="w")

        cnt = tk.Frame(self.live_panel, bg=CARD2)
        cnt.pack(side="left", padx=20)
        tk.Label(cnt, text="볼    스트    아웃", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 10)).grid(row=0, column=0, columnspan=3)
        self.lv_balls_lbl   = tk.Label(cnt, text="0", bg=CARD2, fg=MINT,
                                        font=("Arial", 28, "bold"), width=2)
        self.lv_strikes_lbl = tk.Label(cnt, text="0", bg=CARD2, fg=YELLOW,
                                        font=("Arial", 28, "bold"), width=2)
        self.lv_outs_lbl    = tk.Label(cnt, text="0", bg=CARD2, fg=RED,
                                        font=("Arial", 28, "bold"), width=2)
        self.lv_balls_lbl.grid(row=1, column=0, padx=5)
        self.lv_strikes_lbl.grid(row=1, column=1, padx=5)
        self.lv_outs_lbl.grid(row=1, column=2, padx=5)

        dia = tk.Frame(self.live_panel, bg=CARD2)
        dia.pack(side="left", padx=16)
        tk.Label(dia, text="주자", bg=CARD2, fg=SUBTEXT, font=("Arial", 10)).pack()
        self.diamond_canvas = tk.Canvas(dia, width=70, height=56,
                                         bg=CARD2, highlightthickness=0)
        self.diamond_canvas.pack()
        self._draw_diamond(0, 0, 0)

        pf = tk.Frame(self.live_panel, bg=CARD2)
        pf.pack(side="right", padx=20)
        tk.Label(pf, text="홈팀 승률", bg=CARD2, fg=SUBTEXT, font=("Arial", 11)).pack()
        self.lv_prob_lbl = tk.Label(pf, text="─ %", bg=CARD2, fg=MINT,
                                     font=("Arial", 26, "bold"))
        self.lv_prob_lbl.pack()

        self.lv_badge = tk.Label(self.live_panel, text="● LIVE", bg=CARD2,
                                  fg=RED, font=("Arial", 12, "bold"))
        self.lv_badge.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=6)

    # 탭 영역

    def _build_tab_area(self, parent):
        tab_container = tk.Frame(parent, bg=BG)
        tab_container.grid(row=2, column=0, sticky="nsew")
        self._tab_container = tab_container   # 오버레이 전환 시 사용
        tab_container.rowconfigure(1, weight=1)
        tab_container.columnconfigure(0, weight=1)

        tab_bar = tk.Frame(tab_container, bg=PANEL, height=38)
        tab_bar.grid(row=0, column=0, sticky="ew")
        tab_bar.pack_propagate(False)
        tk.Frame(tab_bar, bg=PANEL, width=6).pack(side="left")

        self._tab_btns = {}
        tab_names = ["요약", "승률 차트", "타석 기록", "개인 기록", "기타 정보", "경기 후 요약"]
        for tab_name in tab_names:
            b = tk.Button(
                tab_bar, text=tab_name,
                bg=PANEL, fg=SUBTEXT,
                font=("Arial", 12),
                relief="flat", bd=0, padx=12, pady=9,
                cursor="hand2",
                activebackground=PANEL, activeforeground=MINT,
                command=lambda t=tab_name: self._switch_tab(t),
            )
            b.pack(side="left")
            self._tab_btns[tab_name] = b

        tk.Frame(tab_container, bg=BORDER, height=1).grid(
            row=0, column=0, sticky="sew"
        )

        self._tab_frames = {}
        stack = tk.Frame(tab_container, bg=BG)
        stack.grid(row=1, column=0, sticky="nsew")
        stack.rowconfigure(0, weight=1)
        stack.columnconfigure(0, weight=1)

        for tab_name in tab_names:
            f = tk.Frame(stack, bg=BG)
            f.grid(row=0, column=0, sticky="nsew")
            self._tab_frames[tab_name] = f

        self._build_tab_summary(self._tab_frames["요약"])
        self._build_tab_chart(self._tab_frames["승률 차트"])
        self._build_tab_records(self._tab_frames["타석 기록"])
        self._build_tab_player_stats(self._tab_frames["개인 기록"])
        self._build_tab_extra_info(self._tab_frames["기타 정보"])
        self._build_tab_postgame(self._tab_frames["경기 후 요약"])

        self._switch_tab("요약")

    def _switch_tab(self, name):
        self._active_tab.set(name)
        self._tab_frames[name].tkraise()
        for t, b in self._tab_btns.items():
            if t == name:
                b.config(fg=MINT, font=("Arial", 12, "bold"), bg=PANEL)
            else:
                b.config(fg=SUBTEXT, font=("Arial", 12), bg=PANEL)

    # 탭: 요약

    def _build_tab_summary(self, parent):
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        # 이닝별 득점표 (가로 스크롤 지원 — 연장전 대응)
        ls_frame = tk.Frame(parent, bg=CARD)
        ls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ls_frame.columnconfigure(0, weight=1)

        tk.Label(ls_frame, text="이닝별 득점", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 2))

        self.linescore_canvas = tk.Canvas(ls_frame, bg=CARD,
                                           height=72, highlightthickness=0)
        self._ls_xsb = ttk.Scrollbar(ls_frame, orient="horizontal",
                                      command=self.linescore_canvas.xview)
        self.linescore_canvas.configure(xscrollcommand=self._ls_xsb.set)
        self.linescore_canvas.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 0))
        self._ls_xsb.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))

        self._draw_empty_linescore()

        # 승리 확률 미니 차트
        mini_frame = tk.Frame(parent, bg=CARD)
        mini_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        mini_frame.columnconfigure(0, weight=1)
        tk.Label(mini_frame, text="승리 확률 추이", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=14, pady=(8, 0))

        self.summary_fig = Figure(figsize=(6, 1.9), facecolor=CARD, dpi=90)
        self.summary_ax  = self.summary_fig.add_subplot(111)
        self._style_summary_ax()
        self.summary_mpl = FigureCanvasTkAgg(self.summary_fig, master=mini_frame)
        self.summary_mpl.get_tk_widget().pack(fill="x", padx=8, pady=(2, 6))
        self._draw_empty_summary_chart()

        # 주요 장면 (스크롤 가능)
        kp_outer = tk.Frame(parent, bg=BG)
        kp_outer.grid(row=2, column=0, sticky="nsew")
        kp_outer.rowconfigure(1, weight=1)
        kp_outer.columnconfigure(0, weight=1)

        tk.Label(kp_outer, text="주요 장면", bg=BG, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(8, 4))

        kp_cv  = tk.Canvas(kp_outer, bg=BG, highlightthickness=0)
        kp_vsb = ttk.Scrollbar(kp_outer, orient="vertical", command=kp_cv.yview)
        kp_cv.configure(yscrollcommand=kp_vsb.set)
        kp_cv.grid(row=1, column=0, sticky="nsew", padx=(14, 0))
        kp_vsb.grid(row=1, column=1, sticky="ns")

        self.keyplay_frame = tk.Frame(kp_cv, bg=BG)
        _kp_win = kp_cv.create_window((0, 0), window=self.keyplay_frame, anchor="nw")
        self.keyplay_frame.bind("<Configure>",
            lambda e: kp_cv.configure(scrollregion=kp_cv.bbox("all")))
        kp_cv.bind("<Configure>",
            lambda e: kp_cv.itemconfig(_kp_win, width=e.width))
        _kp_wheel = _make_wheel_scroll(kp_cv)
        kp_cv.bind("<MouseWheel>", _kp_wheel)
        self.keyplay_frame.bind("<MouseWheel>", _kp_wheel)

        self._kp_placeholder = tk.Label(
            self.keyplay_frame,
            text="경기를 선택하면 주요 장면이 표시됩니다",
            bg=BG, fg=SUBTEXT, font=("Arial", 13),
        )
        self._kp_placeholder.pack(expand=True)

    def _style_summary_ax(self):
        self.summary_ax.set_facecolor(CARD2)
        self.summary_ax.spines[:].set_color(BORDER)
        self.summary_ax.tick_params(colors=SUBTEXT, labelsize=9, length=2)
        self.summary_ax.set_ylim(0, 100)
        self.summary_ax.axhline(50, color=BORDER, linewidth=1,
                                 linestyle="--", alpha=0.6)
        self.summary_ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
        self.summary_fig.subplots_adjust(left=0.06, right=0.97,
                                          top=0.92, bottom=0.14)

    def _draw_empty_summary_chart(self):
        self.summary_ax.cla()
        self._style_summary_ax()
        self.summary_ax.text(0.5, 0.5, "분석 후 표시됩니다",
                              transform=self.summary_ax.transAxes,
                              ha="center", va="center",
                              color=SUBTEXT, fontsize=11)
        self.summary_mpl.draw()

    def _draw_summary_chart(self):
        if not self.game_states or not self.win_probs:
            return
        self.summary_ax.cla()
        self._style_summary_ax()
        arr = np.array(self.win_probs)
        xs  = list(range(1, len(arr) + 1))
        self.summary_ax.plot(xs, arr, color=MINT, linewidth=1.5, zorder=3)
        self.summary_ax.fill_between(xs, arr, 50, where=(arr >= 50),
                                      color=MINT, alpha=0.2, interpolate=True)
        self.summary_ax.fill_between(xs, arr, 50, where=(arr < 50),
                                      color=RED, alpha=0.2, interpolate=True)
        prev_inn = None
        for i, s in enumerate(self.game_states):
            cur = _disp_inn(s)
            if cur != prev_inn:
                self.summary_ax.axvline(i + 1, color=BORDER,
                                         linewidth=0.6, alpha=0.4)
                prev_inn = cur
        if self.win_probs:
            last = self.win_probs[-1]
            self.summary_ax.scatter([xs[-1]], [last],
                                     color=MINT if last >= 50 else RED,
                                     s=28, zorder=5)
        self.summary_ax.set_xlim(0, max(len(xs) + 1, 10))
        self.summary_mpl.draw()

    # 탭: 승률 차트

    def _build_tab_chart(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        chart_frame = tk.Frame(parent, bg=CARD)
        chart_frame.grid(row=0, column=0, sticky="nsew")
        chart_frame.rowconfigure(1, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        ctrl = tk.Frame(chart_frame, bg=CARD)
        ctrl.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 0))

        tk.Label(ctrl, text="승리 확률 변화", bg=CARD, fg=TEXT,
                 font=("Arial", 14, "bold")).pack(side="left")

        rb = tk.Frame(ctrl, bg=CARD)
        rb.pack(side="right")
        self._mkbtn(rb, "▶ 재생", self._start_anim, MINT, w=7).pack(side="left", padx=2)
        self._mkbtn(rb, "■ 정지", self._stop_anim, CARD2, w=7).pack(side="left", padx=2)
        self._mkbtn(rb, "↺ 리셋", self._reset_anim, CARD2, w=7).pack(side="left", padx=2)

        sp = tk.Frame(ctrl, bg=CARD)
        sp.pack(side="right", padx=(0, 12))
        tk.Label(sp, text="재생 속도", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11)).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(sp, text="느림", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 10)).grid(row=1, column=0)
        self.speed_var = tk.IntVar(value=1100)
        tk.Scale(sp, from_=100, to=1500, orient="horizontal",
                 variable=self.speed_var, bg=CARD, fg=TEXT, troughcolor=CARD2,
                 highlightthickness=0, bd=0, length=90, showvalue=False,
                 command=lambda v: setattr(self, "anim_speed", 1600 - int(v)),
                 ).grid(row=1, column=1, padx=4)
        tk.Label(sp, text="빠름", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 10)).grid(row=1, column=2)

        self.fig = Figure(figsize=(9, 4), facecolor=CARD, dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax()
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._draw_empty_chart()
        self._setup_chart_tooltip()

        # Pregame 분석 패널 (차트와 같은 셀 — grid_remove로 교체 표시)
        self._pregame_panel = tk.Frame(chart_frame, bg=CARD)
        self._pregame_panel.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._build_pregame_panel(self._pregame_panel)
        # 기본 상태: pregame 패널 숨기고 차트 표시
        self._pregame_panel.grid_remove()

    # Pregame 분석 패널

    def _build_pregame_panel(self, parent):
        """경기 전 예측 결과를 표시하는 패널 위젯 구조 생성"""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # 상단: 타이틀 + 로딩 상태
        top = tk.Frame(parent, bg=CARD)
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        top.columnconfigure(1, weight=1)

        tk.Label(top, text="🔮  경기 시작 전 승부 예측", bg=CARD, fg=TEXT,
                 font=("Arial", 15, "bold")).grid(row=0, column=0, sticky="w")

        self._pg_status_lbl = tk.Label(top, text="", bg=CARD, fg=YELLOW,
                                        font=("Arial", 12))
        self._pg_status_lbl.grid(row=0, column=1, sticky="e")

        # 승률 바 영역 (원정 vs 홈)
        prob_frame = tk.Frame(parent, bg=CARD2, bd=0)
        prob_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        prob_frame.columnconfigure(0, weight=1)
        prob_frame.columnconfigure(2, weight=1)

        # 원정팀 승률
        self._pg_away_frame = tk.Frame(prob_frame, bg=CARD2)
        self._pg_away_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self._pg_away_name_lbl = tk.Label(
            self._pg_away_frame, text="원정팀", bg=CARD2, fg=TEXT,
            font=("Arial", 13, "bold"))
        self._pg_away_name_lbl.pack()
        self._pg_away_pct_lbl = tk.Label(
            self._pg_away_frame, text="─", bg=CARD2, fg=TEXT,
            font=("Arial", 32, "bold"))
        self._pg_away_pct_lbl.pack()

        # vs
        tk.Label(prob_frame, text="vs", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 15, "bold")).grid(row=0, column=1, padx=8)

        # 홈팀 승률
        self._pg_home_frame = tk.Frame(prob_frame, bg=CARD2)
        self._pg_home_frame.grid(row=0, column=2, sticky="ew", padx=10, pady=10)
        self._pg_home_name_lbl = tk.Label(
            self._pg_home_frame, text="홈팀", bg=CARD2, fg=TEXT,
            font=("Arial", 13, "bold"))
        self._pg_home_name_lbl.pack()
        self._pg_home_pct_lbl = tk.Label(
            self._pg_home_frame, text="─", bg=CARD2, fg=TEXT,
            font=("Arial", 32, "bold"))
        self._pg_home_pct_lbl.pack()

        # 예상 스코어 / 안타
        score_frame = tk.Frame(parent, bg=CARD2)
        score_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))
        score_frame.columnconfigure(1, weight=1)

        tk.Label(score_frame, text="예상 스코어", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(6, 2))

        # 원정 득점/안타
        self._pg_away_score_lbl = tk.Label(
            score_frame, text="─", bg=CARD2, fg=TEXT,
            font=("Arial", 20, "bold"), anchor="e")
        self._pg_away_score_lbl.grid(row=1, column=0, sticky="e", padx=(10, 4), pady=(0, 4))

        tk.Label(score_frame, text="─", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 15, "bold")).grid(row=1, column=1, sticky="ew")
        self._pg_score_sep = score_frame.winfo_children()[-1]   # 중간 구분 "-"

        # 홈 득점/안타
        self._pg_home_score_lbl = tk.Label(
            score_frame, text="─", bg=CARD2, fg=TEXT,
            font=("Arial", 20, "bold"), anchor="w")
        self._pg_home_score_lbl.grid(row=1, column=2, sticky="w", padx=(4, 10), pady=(0, 4))

        # 원정 안타 / 홈 안타 (서브텍스트)
        self._pg_away_hits_lbl = tk.Label(
            score_frame, text="", bg=CARD2, fg=SUBTEXT,
            font=("Arial", 11), anchor="e")
        self._pg_away_hits_lbl.grid(row=2, column=0, sticky="e", padx=(10, 4), pady=(0, 6))

        tk.Label(score_frame, text="", bg=CARD2).grid(row=2, column=1)

        self._pg_home_hits_lbl = tk.Label(
            score_frame, text="", bg=CARD2, fg=SUBTEXT,
            font=("Arial", 11), anchor="w")
        self._pg_home_hits_lbl.grid(row=2, column=2, sticky="w", padx=(4, 10), pady=(0, 6))

        # 선발투수
        pitcher_frame = tk.Frame(parent, bg=CARD)
        pitcher_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))
        pitcher_frame.columnconfigure(0, weight=1)
        pitcher_frame.columnconfigure(1, weight=1)

        self._pg_away_pitcher_lbl = tk.Label(
            pitcher_frame, text="", bg=CARD, fg=SUBTEXT,
            font=("Arial", 11), anchor="w")
        self._pg_away_pitcher_lbl.grid(row=0, column=0, sticky="w")

        self._pg_home_pitcher_lbl = tk.Label(
            pitcher_frame, text="", bg=CARD, fg=SUBTEXT,
            font=("Arial", 11), anchor="e")
        self._pg_home_pitcher_lbl.grid(row=0, column=1, sticky="e")

        # 요인 목록 (스크롤 가능)
        factor_outer = tk.Frame(parent, bg=CARD)
        factor_outer.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 10))
        factor_outer.columnconfigure(0, weight=1)
        factor_outer.rowconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)

        self._pg_factor_canvas = tk.Canvas(
            factor_outer, bg=CARD, highlightthickness=0)
        pg_vsb = ttk.Scrollbar(
            factor_outer, orient="vertical",
            command=self._pg_factor_canvas.yview)
        self._pg_factor_canvas.configure(yscrollcommand=pg_vsb.set)
        self._pg_factor_canvas.grid(row=0, column=0, sticky="nsew")
        pg_vsb.grid(row=0, column=1, sticky="ns")

        self._pg_factor_inner = tk.Frame(self._pg_factor_canvas, bg=CARD)
        self._pg_factor_win = self._pg_factor_canvas.create_window(
            (0, 0), window=self._pg_factor_inner, anchor="nw")
        self._pg_factor_inner.bind(
            "<Configure>",
            lambda e: self._pg_factor_canvas.configure(
                scrollregion=self._pg_factor_canvas.bbox("all")))
        self._pg_factor_canvas.bind(
            "<Configure>",
            lambda e: self._pg_factor_canvas.itemconfig(
                self._pg_factor_win, width=e.width))

    def _run_pregame_analysis(self, game):
        """백그라운드 스레드: pregame 데이터 수집 → 예측 → UI 업데이트"""
        if not _PREGAME_OK:
            return
        if self._pregame_loading:
            return
        self._pregame_loading = True
        # 오버레이 표시 + 로딩 상태
        self._show_pregame_overlay()
        self._pg_ov_status.config(text="⏳ 데이터 수집 중...", fg=YELLOW)
        self._show_pregame_panel()   # 차트탭 내 미니 패널도 같이

        game_pk      = game.get("gamePk")
        home_team_id = game.get("home_team_id")
        away_team_id = game.get("away_team_id")

        def worker():
            try:
                data   = collect_pregame_data(game_pk, home_team_id, away_team_id)
                result = self._pregame.predict(data)
                result["_game"] = game
                self.after(0, lambda: self._apply_pregame_result(result))
            except Exception as e:
                self.after(0, lambda: self._pg_status_lbl.config(
                    text=f"분석 실패: {e}", fg=RED))
            finally:
                self._pregame_loading = False

        threading.Thread(target=worker, daemon=True).start()

    def _apply_pregame_result(self, result):
        """예측 결과를 UI에 반영"""
        self._pregame_result = result
        game = result.get("_game", {})

        home_pct = result.get("home_win_pct", 0.5) * 100
        away_pct = result.get("away_win_pct", 0.5) * 100

        home_team = game.get("home_team", "홈팀")
        away_team = game.get("away_team", "원정팀")

        # 스코어보드 승률 라벨 업데이트
        self._sb_away_prob.config(text=f"{away_pct:.1f}%")
        self._sb_home_prob.config(text=f"{home_pct:.1f}%")

        # Pregame 패널 업데이트
        self._pg_status_lbl.config(text="✅ 예측 완료", fg=MINT)
        self._pg_away_name_lbl.config(text=team_kr(away_team))
        self._pg_home_name_lbl.config(text=team_kr(home_team))

        # 승리팀을 민트, 패배팀을 서브텍스트로 강조
        if home_pct >= away_pct:
            self._pg_home_pct_lbl.config(text=f"{home_pct:.1f}%", fg=MINT)
            self._pg_away_pct_lbl.config(text=f"{away_pct:.1f}%", fg=SUBTEXT)
        else:
            self._pg_away_pct_lbl.config(text=f"{away_pct:.1f}%", fg=MINT)
            self._pg_home_pct_lbl.config(text=f"{home_pct:.1f}%", fg=SUBTEXT)

        # 예상 스코어 / 안타
        exp_h_r = result.get("exp_home_runs", 0.0)
        exp_a_r = result.get("exp_away_runs", 0.0)
        exp_h_h = result.get("exp_home_hits", 0.0)
        exp_a_h = result.get("exp_away_hits", 0.0)

        # 높은 쪽 득점을 민트색으로 강조
        h_run_fg = MINT if exp_h_r >= exp_a_r else TEXT
        a_run_fg = MINT if exp_a_r >  exp_h_r else TEXT

        self._pg_away_score_lbl.config(text=f"{exp_a_r:.1f}", fg=a_run_fg)
        self._pg_home_score_lbl.config(text=f"{exp_h_r:.1f}", fg=h_run_fg)
        self._pg_away_hits_lbl.config(text=f"예상 안타 {exp_a_h:.1f}개")
        self._pg_home_hits_lbl.config(text=f"예상 안타 {exp_h_h:.1f}개")

        # 선발투수 정보
        pitchers = result.get("pitchers") or {}
        hp_name  = pitchers.get("home_pitcher_name") or "─"
        ap_name  = pitchers.get("away_pitcher_name") or "─"
        hp_stats = pitchers.get("home_pitcher_stats") or {}
        ap_stats = pitchers.get("away_pitcher_stats") or {}

        hp_era = hp_stats.get("era", "─")
        ap_era = ap_stats.get("era", "─")
        hw_l   = f"{hp_stats.get('wins',0)}-{hp_stats.get('losses',0)}" if hp_stats else "─"
        aw_l   = f"{ap_stats.get('wins',0)}-{ap_stats.get('losses',0)}" if ap_stats else "─"

        self._pg_away_pitcher_lbl.config(
            text=f"선발: {ap_name}  ERA {ap_era}  {aw_l}")
        self._pg_home_pitcher_lbl.config(
            text=f"선발: {hp_name}  ERA {hp_era}  {hw_l}")

        # 요인 목록 재구성
        for w in self._pg_factor_inner.winfo_children():
            w.destroy()

        factors = result.get("factors", [])
        tk.Label(
            self._pg_factor_inner, text="예측 요인", bg=CARD, fg=SUBTEXT,
            font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 2))

        for i, f in enumerate(factors):
            row_bg = CARD if i % 2 == 0 else CARD2
            name_lbl = tk.Label(
                self._pg_factor_inner, text=f.get("name", ""),
                bg=row_bg, fg=TEXT, font=("Arial", 11), anchor="w", width=16)
            name_lbl.grid(row=i + 1, column=0, sticky="ew", padx=(6, 2), pady=1)

            detail_lbl = tk.Label(
                self._pg_factor_inner, text=f.get("detail", ""),
                bg=row_bg, fg=SUBTEXT, font=("Arial", 10), anchor="w")
            detail_lbl.grid(row=i + 1, column=1, sticky="ew", padx=2, pady=1)

            # 엣지 바 (원정 vs 홈)
            h_edge = f.get("home_edge", 0)
            a_edge = f.get("away_edge", 0)
            edge_fg = MINT if h_edge >= a_edge else RED
            edge_lbl = tk.Label(
                self._pg_factor_inner,
                text=f"홈 {h_edge:.0f}% vs 원정 {a_edge:.0f}%",
                bg=row_bg, fg=edge_fg, font=("Arial", 10), anchor="e")
            edge_lbl.grid(row=i + 1, column=2, sticky="ew", padx=(4, 6), pady=1)

        self._pg_factor_inner.columnconfigure(1, weight=1)
        self._pg_factor_inner.columnconfigure(2, weight=1)

        # 오버레이 채우기
        self._fill_pregame_overlay(result)
        # 차트 탭 내 미니 패널도 업데이트
        self._show_pregame_panel()

    def _show_pregame_panel(self):
        """pregame 패널을 grid에 추가하고 차트는 grid_remove"""
        self.canvas.get_tk_widget().grid_remove()
        self._pregame_panel.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _hide_pregame_panel(self):
        """pregame 오버레이 + 차트 탭 미니 패널 숨기고 탭 영역 복원"""
        self._hide_pregame_overlay()
        self._pregame_panel.grid_remove()
        self.canvas.get_tk_widget().grid(
            row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._sb_away_prob.config(text="")
        self._sb_home_prob.config(text="")
        self._pregame_result = None

    # 경기 전 전체 프리뷰 오버레이

    def _build_pregame_overlay(self, parent):
        """탭 영역 전체를 덮는 경기 전 프리뷰 오버레이"""
        ov = tk.Frame(parent, bg=BG)
        ov.grid(row=2, column=0, sticky="nsew")
        ov.rowconfigure(1, weight=1)
        ov.columnconfigure(0, weight=1)
        self._pregame_overlay = ov

        # 헤더 바
        hdr = tk.Frame(ov, bg=PANEL, height=38)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="경기 시작 전 프리뷰", bg=PANEL, fg=TEXT,
                 font=("Arial", 13, "bold")).pack(side="left", padx=14)
        self._pg_ov_status = tk.Label(hdr, text="", bg=PANEL, fg=YELLOW,
                                       font=("Arial", 12))
        self._pg_ov_status.pack(side="right", padx=14)

        # 스크롤 가능 본문
        body_outer = tk.Frame(ov, bg=BG)
        body_outer.grid(row=1, column=0, sticky="nsew")
        body_outer.rowconfigure(0, weight=1)
        body_outer.columnconfigure(0, weight=1)

        body_cv = tk.Canvas(body_outer, bg=BG, highlightthickness=0)
        body_vsb = ttk.Scrollbar(body_outer, orient="vertical",
                                  command=body_cv.yview)
        body_cv.configure(yscrollcommand=body_vsb.set)
        body_cv.grid(row=0, column=0, sticky="nsew")
        body_vsb.grid(row=0, column=1, sticky="ns")

        body = tk.Frame(body_cv, bg=BG)
        body_win = body_cv.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda e: body_cv.configure(scrollregion=body_cv.bbox("all")))
        body_cv.bind("<Configure>",
                     lambda e: body_cv.itemconfig(body_win, width=e.width))

        # 마우스 휠 스크롤 (Mac 트랙패드 포함, 1 unit씩)
        def _on_wheel(e):
            body_cv.yview_scroll(-1 if e.delta > 0 else 1, "units")
            return "break"
        body_cv.bind("<MouseWheel>", _on_wheel)
        body.bind("<MouseWheel>", _on_wheel)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # 상단 통계 카드 행
        stats_row = tk.Frame(body, bg=BG)
        stats_row.grid(row=0, column=0, columnspan=2, sticky="ew",
                       padx=12, pady=(12, 6))
        stats_row.columnconfigure(0, weight=1)
        stats_row.columnconfigure(1, weight=1)

        # 예상 스코어 카드
        sc_card = tk.Frame(stats_row, bg=CARD)
        sc_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), ipady=8)
        sc_card.columnconfigure(0, weight=1)
        sc_card.columnconfigure(2, weight=1)
        tk.Label(sc_card, text="예상 스코어", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4))

        self._pg_ov_away_score = tk.Label(sc_card, text="─", bg=CARD, fg=TEXT,
                                           font=("Arial", 28, "bold"), anchor="e")
        self._pg_ov_away_score.grid(row=1, column=0, sticky="e", padx=(10, 6))

        tk.Label(sc_card, text=":", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 22, "bold")).grid(row=1, column=1)

        self._pg_ov_home_score = tk.Label(sc_card, text="─", bg=CARD, fg=TEXT,
                                           font=("Arial", 28, "bold"), anchor="w")
        self._pg_ov_home_score.grid(row=1, column=2, sticky="w", padx=(6, 10))

        self._pg_ov_away_hits = tk.Label(sc_card, text="", bg=CARD, fg=SUBTEXT,
                                          font=("Arial", 10), anchor="e")
        self._pg_ov_away_hits.grid(row=2, column=0, sticky="e", padx=(10, 6), pady=(0, 8))

        self._pg_ov_home_hits = tk.Label(sc_card, text="", bg=CARD, fg=SUBTEXT,
                                          font=("Arial", 10), anchor="w")
        self._pg_ov_home_hits.grid(row=2, column=2, sticky="w", padx=(6, 10), pady=(0, 8))

        # 선발투수 카드
        pit_card = tk.Frame(stats_row, bg=CARD)
        pit_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), ipady=8)
        pit_card.columnconfigure(0, weight=1)
        tk.Label(pit_card, text="선발 투수", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        self._pg_ov_away_pit = tk.Label(pit_card, text="─", bg=CARD, fg=TEXT,
                                         font=("Arial", 12, "bold"), anchor="w")
        self._pg_ov_away_pit.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 2))
        self._pg_ov_away_pit_stat = tk.Label(pit_card, text="", bg=CARD, fg=SUBTEXT,
                                              font=("Arial", 10), anchor="w")
        self._pg_ov_away_pit_stat.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))

        tk.Frame(pit_card, bg=BORDER, height=1).grid(
            row=3, column=0, sticky="ew", padx=10)

        self._pg_ov_home_pit = tk.Label(pit_card, text="─", bg=CARD, fg=TEXT,
                                         font=("Arial", 12, "bold"), anchor="w")
        self._pg_ov_home_pit.grid(row=4, column=0, sticky="ew", padx=10, pady=(8, 2))
        self._pg_ov_home_pit_stat = tk.Label(pit_card, text="", bg=CARD, fg=SUBTEXT,
                                              font=("Arial", 10), anchor="w")
        self._pg_ov_home_pit_stat.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 8))

        # 타순 영역 (가변 — _fill_pregame_overlay에서 채움)
        lineup_hdr = tk.Frame(body, bg=BG)
        lineup_hdr.grid(row=1, column=0, columnspan=2, sticky="ew",
                        padx=12, pady=(6, 4))
        tk.Label(lineup_hdr, text="타순", bg=BG, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).pack(side="left")

        # 원정 타순 프레임
        self._pg_ov_away_lineup = tk.Frame(body, bg=CARD)
        self._pg_ov_away_lineup.grid(row=2, column=0, sticky="nsew",
                                      padx=(12, 6), pady=(0, 12))
        self._pg_ov_away_lineup.columnconfigure(0, weight=1)

        # 홈 타순 프레임
        self._pg_ov_home_lineup = tk.Frame(body, bg=CARD)
        self._pg_ov_home_lineup.grid(row=2, column=1, sticky="nsew",
                                      padx=(6, 12), pady=(0, 12))
        self._pg_ov_home_lineup.columnconfigure(0, weight=1)

    def _fill_pregame_overlay(self, result):
        """예측 결과로 오버레이 위젯 채우기"""
        game = result.get("_game", {})
        away_team = game.get("away_team", "원정팀")
        home_team = game.get("home_team", "홈팀")

        self._pg_ov_status.config(text="✅ 예측 완료", fg=MINT)

        # 예상 스코어
        exp_a = result.get("exp_away_runs", 0.0)
        exp_h = result.get("exp_home_runs", 0.0)
        a_fg  = MINT if exp_a > exp_h else TEXT
        h_fg  = MINT if exp_h >= exp_a else TEXT
        self._pg_ov_away_score.config(text=f"{exp_a:.1f}", fg=a_fg)
        self._pg_ov_home_score.config(text=f"{exp_h:.1f}", fg=h_fg)
        self._pg_ov_away_hits.config(
            text=f"안타 {result.get('exp_away_hits', 0):.1f}개  ({team_kr(away_team)})")
        self._pg_ov_home_hits.config(
            text=f"({team_kr(home_team)})  안타 {result.get('exp_home_hits', 0):.1f}개")

        # 선발투수
        pits    = result.get("pitchers") or {}
        ap_name = pits.get("away_pitcher_name") or "미발표"
        hp_name = pits.get("home_pitcher_name") or "미발표"
        ap_stat = pits.get("away_pitcher_stats") or {}
        hp_stat = pits.get("home_pitcher_stats") or {}

        ap_recent = result.get("away_starter_recent") or {}
        hp_recent = result.get("home_starter_recent") or {}

        def pit_detail(stat, recent):
            if not stat:
                return ""
            base = (f"ERA {stat.get('era', '─')}  "
                    f"{stat.get('wins', 0)}승 {stat.get('losses', 0)}패  "
                    f"IP {stat.get('ip', 0):.0f}")
            if recent.get("era") is not None:
                base += f"  |  최근{recent.get('games',0)}경기 ERA {recent['era']:.2f}"
            return base

        self._pg_ov_away_pit.config(text=f"원정  {ap_name}")
        self._pg_ov_away_pit_stat.config(text=pit_detail(ap_stat, ap_recent))
        self._pg_ov_home_pit.config(text=f"홈  {hp_name}")
        self._pg_ov_home_pit_stat.config(text=pit_detail(hp_stat, hp_recent))

        # 타순
        lineup = result.get("lineup") or {"home": [], "away": []}
        self._fill_lineup_panel(
            self._pg_ov_away_lineup, team_kr(away_team),
            lineup.get("away", []),
            result.get("away_win_pct", 0.5))
        self._fill_lineup_panel(
            self._pg_ov_home_lineup, team_kr(home_team),
            lineup.get("home", []),
            result.get("home_win_pct", 0.5))

    def _fill_lineup_panel(self, frame, team_name, lineup, win_pct):
        """단일 팀 타순 패널 채우기"""
        for w in frame.winfo_children():
            w.destroy()

        # 팀명 헤더
        hdr = tk.Frame(frame, bg=CARD2)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text=team_name, bg=CARD2, fg=TEXT,
                 font=("Arial", 12, "bold")).pack(side="left", padx=12, pady=8)
        pct_fg = MINT if win_pct >= 0.5 else SUBTEXT
        tk.Label(hdr, text=f"승률 {win_pct*100:.1f}%", bg=CARD2, fg=pct_fg,
                 font=("Arial", 11, "bold")).pack(side="right", padx=12)

        if not lineup:
            tk.Label(frame, text="라인업 미발표", bg=CARD, fg=SUBTEXT,
                     font=("Arial", 11)).grid(
                row=1, column=0, sticky="ew", padx=12, pady=16)
            return

        for i, p in enumerate(lineup[:9]):
            row_bg = CARD if i % 2 == 0 else CARD2
            row_f  = tk.Frame(frame, bg=row_bg)
            row_f.grid(row=i + 1, column=0, sticky="ew")
            row_f.columnconfigure(2, weight=1)

            # 타순 번호
            tk.Label(row_f, text=f"{p['order']}",
                     bg=row_bg, fg=MINT, font=("Arial", 12, "bold"),
                     width=2, anchor="e").grid(row=0, column=0, padx=(10, 6), pady=5)
            # 포지션
            tk.Label(row_f, text=p.get("pos", ""),
                     bg=row_bg, fg=SUBTEXT, font=("Arial", 10),
                     width=3, anchor="w").grid(row=0, column=1, padx=(0, 6))
            # 선수 이름
            tk.Label(row_f, text=p.get("name", ""),
                     bg=row_bg, fg=TEXT, font=("Arial", 11),
                     anchor="w").grid(row=0, column=2, sticky="ew", padx=(0, 10))

    def _show_pregame_overlay(self):
        """탭 영역 숨기고 프리뷰 오버레이 표시"""
        self._tab_container.grid_remove()
        self._pregame_overlay.grid(row=2, column=0, sticky="nsew")

    def _hide_pregame_overlay(self):
        """프리뷰 오버레이 숨기고 탭 영역 복원"""
        self._pregame_overlay.grid_remove()
        self._tab_container.grid(row=2, column=0, sticky="nsew")

    # 차트 툴팁

    def _setup_chart_tooltip(self):
        """마우스 호버 시 타석 정보를 보여주는 툴팁 초기화"""
        self._tooltip_ann = None
        self._tooltip_vline = None
        self.canvas.mpl_connect("motion_notify_event", self._on_chart_hover)

    def _on_chart_hover(self, event):
        if event.inaxes != self.ax or not self.game_states or not self.win_probs:
            self._hide_tooltip()
            return

        x = event.xdata
        if x is None:
            self._hide_tooltip()
            return

        idx = int(round(x)) - 1
        if idx < 0 or idx >= len(self.game_states):
            self._hide_tooltip()
            return

        s = self.game_states[idx]
        p = self.win_probs[idx]

        half    = "초" if s.get("is_top") else "말"
        inn     = _disp_inn(s)
        batter  = s.get("batter",  "─")
        pitcher = s.get("pitcher", "─")
        evt     = event_kr(s.get("event", "─"))
        h_sc    = s.get("home_score", 0)
        a_sc    = s.get("away_score", 0)
        r1      = "●" if s.get("runner_1b") else "○"
        r2      = "●" if s.get("runner_2b") else "○"
        r3      = "●" if s.get("runner_3b") else "○"
        outs    = s.get("outs", 0)

        # WPA 계산
        if idx > 0:
            wpa     = p - self.win_probs[idx - 1]
            wpa_str = f"{'+' if wpa >= 0 else ''}{wpa:.1f}%"
        else:
            wpa_str = "─"

        text = "\n".join([
            f"  {inn}회 {half}  ·  아웃 {outs}개  ",
            f"  타자: {batter}  ",
            f"  투수: {pitcher}  ",
            f"  결과: {evt}  ",
            f"  스코어: {a_sc} - {h_sc}  ",
            f"  홈 승률: {p:.1f}%  (WPA {wpa_str})  ",
            f"  주자: 3루{r3} 2루{r2} 1루{r1}  ",
        ])

        # 기존 툴팁 제거 (stale 참조 안전 처리)
        self._hide_tooltip()

        try:
            # 수직선
            self._tooltip_vline = self.ax.axvline(
                idx + 1, color=YELLOW, linewidth=1, alpha=0.6, zorder=4)

            # 툴팁 위치 (오른쪽 60% 이상이면 왼쪽으로)
            x_frac = (idx + 1) / max(len(self.game_states), 1)
            ha    = "right" if x_frac > 0.6 else "left"
            x_off = -12    if ha == "right"  else 12
            # y 위치: 차트 위쪽 절반이면 아래로, 아래쪽이면 위로
            y_off = -80 if p > 60 else 20

            self._tooltip_ann = self.ax.annotate(
                text,
                xy=(idx + 1, p),
                xytext=(x_off, y_off),
                textcoords="offset points",
                ha=ha, va="bottom",
                fontsize=10.5,
                color="#f0f0f0",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="#1e2a1e",
                    edgecolor=YELLOW,
                    linewidth=1.2,
                    alpha=0.96,
                ),
                zorder=10,
            )
            self.canvas.draw_idle()
        except Exception:
            self._tooltip_ann   = None
            self._tooltip_vline = None

    def _hide_tooltip(self):
        changed = False
        if self._tooltip_ann is not None:
            try:
                self._tooltip_ann.remove()
            except Exception:
                pass
            self._tooltip_ann = None
            changed = True
        if self._tooltip_vline is not None:
            try:
                self._tooltip_vline.remove()
            except Exception:
                pass
            self._tooltip_vline = None
            changed = True
        if changed:
            self.canvas.draw_idle()

    # 탭: 타석 기록

    def _build_tab_records(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        frame = tk.Frame(parent, bg=CARD)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        # 헤더 + 팀 선택 버튼
        hdr = tk.Frame(frame, bg=CARD)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 4))

        tk.Label(hdr, text="타석 기록", bg=CARD, fg=TEXT,
                 font=("Arial", 14, "bold")).pack(side="left")

        self._rec_team_var = tk.StringVar(value="away")
        # HOME 먼저 팩(→ 오른쪽 끝), AWAY 다음(→ HOME 왼쪽) = [원정][홈] 순서
        self._rec_home_btn = tk.Button(
            hdr, text="홈팀 공격 (말)", bg=CARD2, fg=SUBTEXT,
            font=("Arial", 11, "bold"), relief="flat", bd=0,
            padx=12, pady=4, cursor="hand2",
            command=lambda: self._switch_records_team("home"))
        self._rec_home_btn.pack(side="right")
        self._rec_away_btn = tk.Button(
            hdr, text="원정팀 공격 (초)", bg=MINT, fg=BG,
            font=("Arial", 11, "bold"), relief="flat", bd=0,
            padx=12, pady=4, cursor="hand2",
            command=lambda: self._switch_records_team("away"))
        self._rec_away_btn.pack(side="right", padx=(4, 4))

        # 이닝 구분 라벨
        self._rec_team_lbl = tk.Label(frame, text="", bg=CARD, fg=SUBTEXT,
                                       font=("Arial", 11))
        self._rec_team_lbl.grid(row=1, column=0, columnspan=2, sticky="w",
                                 padx=16, pady=(0, 4))

        cols = ("이닝", "타자", "투수", "이벤트", "홈", "원정", "홈 승률")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=26,
                        font=("Arial", 12))
        style.configure("Treeview.Heading",
                        background=CARD2, foreground=SUBTEXT,
                        font=("Arial", 11, "bold"), relief="flat")
        style.map("Treeview",
                  background=[("selected", MINT)],
                  foreground=[("selected", BG)])

        self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        for col, w in zip(cols, [72, 130, 130, 210, 60, 70, 80]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=2, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        vsb.grid(row=2, column=1, sticky="ns", pady=(0, 8))

    # 탭: 개인 기록

    def _build_tab_player_stats(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        outer = tk.Frame(parent, bg=CARD)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(2, weight=1)
        outer.columnconfigure(0, weight=1)

        # 헤더 + 팀 선택 버튼
        hdr = tk.Frame(outer, bg=CARD)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=14, pady=(10, 4))
        tk.Label(hdr, text="개인 기록", bg=CARD, fg=TEXT,
                 font=("Arial", 14, "bold")).pack(side="left")

        self._ps_team = tk.StringVar(value="away")
        # 팩 순서: HOME 먼저(→ 오른쪽 끝), AWAY 다음(→ HOME 왼쪽)
        # 결과: [AWAY(원정)] [HOME(홈)] — 스코어보드와 동일 순서
        self._home_team_btn = tk.Button(
            hdr, text="홈팀", bg=CARD2, fg=SUBTEXT,
            font=("Arial", 12, "bold"), relief="flat", bd=0,
            padx=16, pady=4, cursor="hand2",
            command=lambda: self._switch_ps_team("home"))
        self._home_team_btn.pack(side="right")
        self._away_team_btn = tk.Button(
            hdr, text="원정팀", bg=MINT, fg=BG,
            font=("Arial", 12, "bold"), relief="flat", bd=0,
            padx=16, pady=4, cursor="hand2",
            command=lambda: self._switch_ps_team("away"))
        self._away_team_btn.pack(side="right", padx=(4, 4))

        # 타자 컬럼 헤더 (고정)
        # (label, char_width)  — 스탯 열은 4자 너비로 통일
        _B_COLS = [("타수",4),("득점",4),("안타",4),("홈런",4),("타점",4),
                   ("도루",4),("볼넷",4),("사구",4),("삼진",4),("병살",4),("실책",4)]
        self._b_col_labels = [c[0] for c in _B_COLS]
        self._b_col_w      = [c[1] for c in _B_COLS]

        self._ps_batter_hdr = tk.Frame(outer, bg=CARD2)
        self._ps_batter_hdr.grid(row=1, column=0, sticky="ew", padx=8, pady=(0,1))
        self._ps_batter_hdr.columnconfigure(1, weight=1)
        # 타순 고정
        tk.Label(self._ps_batter_hdr, text="타순", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 10, "bold"), width=5,
                 anchor="center").pack(side="left", padx=(4,0), pady=3)
        # 이름
        tk.Label(self._ps_batter_hdr, text="이름·포지션", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 10, "bold"),
                 anchor="w").pack(side="left", fill="x", expand=True, padx=4)
        # 스탯 열
        for lbl, w in _B_COLS:
            tk.Label(self._ps_batter_hdr, text=lbl, bg=CARD2, fg=SUBTEXT,
                     font=("Arial", 10, "bold"), width=w,
                     anchor="center").pack(side="left", padx=1, pady=3)

        # 스크롤 가능 선수 카드 영역
        ps_cv = tk.Canvas(outer, bg=CARD, highlightthickness=0)
        ps_vsb = ttk.Scrollbar(outer, orient="vertical", command=ps_cv.yview)
        ps_cv.configure(yscrollcommand=ps_vsb.set)
        ps_cv.grid(row=2, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        ps_vsb.grid(row=2, column=1, sticky="ns", pady=(0, 8))

        self._ps_inner = tk.Frame(ps_cv, bg=CARD)
        self._ps_win = ps_cv.create_window((0, 0), window=self._ps_inner, anchor="nw")
        self._ps_inner.bind("<Configure>",
            lambda e: ps_cv.configure(scrollregion=ps_cv.bbox("all")))
        ps_cv.bind("<Configure>",
            lambda e: ps_cv.itemconfig(self._ps_win, width=e.width))
        _ps_wheel = _make_wheel_scroll(ps_cv)
        ps_cv.bind("<MouseWheel>", _ps_wheel)
        self._ps_inner.bind("<MouseWheel>", _ps_wheel)

        tk.Label(self._ps_inner,
                 text="경기를 분석하면 개인 기록이 표시됩니다",
                 bg=CARD, fg=SUBTEXT, font=("Arial", 12)).pack(pady=30)

    def _switch_ps_team(self, team):
        self._ps_team.set(team)
        away_name = team_kr(self._away_team_name or "원정팀")
        home_name = team_kr(self._home_team_name or "홈팀")
        self._away_team_btn.config(
            text=away_name[:14],
            bg=MINT if team == "away" else CARD2,
            fg=BG   if team == "away" else SUBTEXT)
        self._home_team_btn.config(
            text=home_name[:14],
            bg=MINT if team == "home" else CARD2,
            fg=BG   if team == "home" else SUBTEXT)
        if self.game_states:
            self._update_player_stats(self.game_states)

    # 탭: 기타 정보

    def _build_tab_extra_info(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        outer = tk.Frame(parent, bg=BG)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text="기타 정보", bg=BG, fg=TEXT,
                 font=("Arial", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        # 스크롤 가능한 내부 영역
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        self._ei_inner = tk.Frame(canvas, bg=BG)
        self._ei_win = canvas.create_window((0, 0), window=self._ei_inner, anchor="nw")
        self._ei_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(self._ei_win, width=e.width))
        _ei_wheel = _make_wheel_scroll(canvas)
        canvas.bind("<MouseWheel>", _ei_wheel)
        self._ei_inner.bind("<MouseWheel>", _ei_wheel)

        # placeholder
        self._ei_placeholder = tk.Label(
            self._ei_inner,
            text="경기를 분석하면 기타 정보가 표시됩니다",
            bg=BG, fg=SUBTEXT, font=("Arial", 13),
        )
        self._ei_placeholder.pack(pady=40)

    def _ei_card(self, parent, title: str, rows_kv: list):
        """key-value 형태의 정보 카드"""
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x", padx=14, pady=4)

        tk.Label(card, text=title, bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=12)

        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x", padx=12, pady=(4, 8))
        for i, (k, v) in enumerate(rows_kv):
            col_offset = (i % 2) * 2
            row_idx    = i // 2
            tk.Label(grid, text=str(k), bg=CARD, fg=SUBTEXT,
                     font=("Arial", 11), anchor="w").grid(
                row=row_idx, column=col_offset, sticky="w", padx=(0, 6), pady=2)
            tk.Label(grid, text=str(v), bg=CARD, fg=TEXT,
                     font=("Arial", 12, "bold"), anchor="w").grid(
                row=row_idx, column=col_offset + 1, sticky="w", padx=(0, 20), pady=2)
        return card

    def _ei_list_card(self, parent, title: str, items: list):
        """목록형 정보 카드"""
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x", padx=14, pady=4)

        tk.Label(card, text=title, bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=12)

        for item in items:
            tk.Label(card, text=f"  {item}", bg=CARD, fg=TEXT,
                     font=("Arial", 12), anchor="w").pack(
                fill="x", padx=12, pady=2)
        if not items:
            tk.Label(card, text="  기록 없음", bg=CARD, fg=SUBTEXT,
                     font=("Arial", 12)).pack(fill="x", padx=12, pady=4)
        tk.Frame(card, bg=BG, height=4).pack()
        return card

    def _update_extra_info(self, feed, game, states):
        """기타 정보 탭 갱신"""
        for w in self._ei_inner.winfo_children():
            w.destroy()

        live_data = feed.get("liveData", {}) if feed else {}
        box_score = live_data.get("boxscore", {})
        game_info = box_score.get("info", [])
        linescore = live_data.get("linescore", {})
        game_data_f = feed.get("gameData", {}) if feed else {}

        # 경기 기본 정보
        venue     = game_data_f.get("venue", {}).get("name", game.get("venue", "─"))
        weather   = game_data_f.get("weather", {})
        wx_str    = ""
        if weather:
            wx_str = f"{weather.get('condition', '')} {weather.get('temp', '')}°F"

        # boxscore info에서 관중/시간 파싱
        attend = "─"; duration = "─"
        for item in game_info:
            label = item.get("label", "").lower()
            val   = item.get("value", "")
            if "attendance" in label or "관중" in label:
                attend = val
            elif "duration" in label or "time" in label:
                duration = val

        basic_rows = [
            ("구장", venue),
            ("날씨", wx_str or "─"),
            ("관중수", attend),
            ("경기 시간", duration),
        ]
        self._ei_card(self._ei_inner, "경기 정보", basic_rows)

        # 타팀별 H/R/E
        home_totals = linescore.get("teams", {}).get("home", {})
        away_totals = linescore.get("teams", {}).get("away", {})
        away_kr = team_kr(self._away_team_name)
        home_kr = team_kr(self._home_team_name)
        score_rows = [
            (f"{away_kr} 안타", away_totals.get("hits", "─")),
            (f"{home_kr} 안타", home_totals.get("hits", "─")),
            (f"{away_kr} 실책", away_totals.get("errors", "─")),
            (f"{home_kr} 실책", home_totals.get("errors", "─")),
        ]
        self._ei_card(self._ei_inner, "안타 / 실책", score_rows)

        # 개인 기록 (홈런, 2루타, 3루타, 도루, 병살타)
        from collections import defaultdict
        hr_list = []; dbl_list = []; tpl_list = []
        sb_list = []; dp_list = []
        for s in states:
            evt  = s.get("event", "")
            bat  = s.get("batter", "")
            inn  = _disp_inn(s)
            half = "초" if s.get("is_top") else "말"
            tag  = f"{inn}회 {half} {bat}"
            if evt == "Home Run":
                hr_list.append(tag)
            elif evt == "Double":
                dbl_list.append(tag)
            elif evt == "Triple":
                tpl_list.append(tag)
            elif evt == "Stolen Base":
                sb_list.append(tag)
            elif "Double Play" in evt or "Grounded Into DP" in evt:
                dp_list.append(tag)

        self._ei_list_card(self._ei_inner, f"홈런 ({len(hr_list)})", hr_list)
        self._ei_list_card(self._ei_inner, f"2루타 ({len(dbl_list)})", dbl_list)
        self._ei_list_card(self._ei_inner, f"3루타 ({len(tpl_list)})", tpl_list)
        self._ei_list_card(self._ei_inner, f"도루 ({len(sb_list)})", sb_list)
        self._ei_list_card(self._ei_inner, f"병살타 ({len(dp_list)})", dp_list)

        # 심판 정보
        officials = box_score.get("officials", [])
        ump_list = []
        for o in officials:
            role    = o.get("officialType", "")
            ump_name = o.get("official", {}).get("fullName", "")
            if ump_name:
                ump_list.append(f"{role}: {ump_name}")
        self._ei_list_card(self._ei_inner, "심판", ump_list)

    # 탭: 경기 후 요약

    def _build_tab_postgame(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        outer = tk.Frame(parent, bg=BG)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text="경기 후 요약", bg=BG, fg=TEXT,
                 font=("Arial", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        self._pg_inner = tk.Frame(canvas, bg=BG)
        self._pg_win = canvas.create_window((0, 0), window=self._pg_inner, anchor="nw")
        self._pg_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(self._pg_win, width=e.width))
        _pg_wheel = _make_wheel_scroll(canvas)
        canvas.bind("<MouseWheel>", _pg_wheel)
        self._pg_inner.bind("<MouseWheel>", _pg_wheel)

        self._pg_placeholder = tk.Label(
            self._pg_inner,
            text="경기를 분석하면 MVP와 WPA 순위가 표시됩니다",
            bg=BG, fg=SUBTEXT, font=("Arial", 13),
        )
        self._pg_placeholder.pack(pady=40)

    def _calc_wpa_plays(self, states, probs):
        """각 타석별 WPA(Win Probability Added) 계산"""
        results = []
        for i, (s, p) in enumerate(zip(states, probs)):
            if i == 0:
                wpa = 0.0
            else:
                prev_p = probs[i - 1]
                # 홈팀 관점: is_top=1(원정 공격)이면 홈팀에게 불리한 방향
                delta = p - prev_p
                is_top = s.get("is_top", 0)
                wpa = -delta if is_top else delta
            results.append((s, p, wpa))
        return results

    def _update_mvp_card(self, states, probs):
        """경기 후 요약 탭: MVP 카드 + WPA 리더보드 갱신"""
        for w in self._pg_inner.winfo_children():
            w.destroy()

        if not states or not probs:
            tk.Label(self._pg_inner, text="데이터 없음",
                     bg=BG, fg=SUBTEXT, font=("Arial", 13)).pack(pady=40)
            return

        wpa_plays = self._calc_wpa_plays(states, probs)

        # 선수별 WPA 합산
        from collections import defaultdict
        player_wpa = defaultdict(float)
        for s, p, wpa in wpa_plays:
            bat = s.get("batter", "")
            if bat:
                player_wpa[bat] += wpa

        if not player_wpa:
            tk.Label(self._pg_inner, text="WPA 데이터 없음",
                     bg=BG, fg=SUBTEXT, font=("Arial", 13)).pack(pady=40)
            return

        # MVP = WPA 절댓값 최대
        mvp_name = max(player_wpa, key=lambda k: abs(player_wpa[k]))
        mvp_wpa  = player_wpa[mvp_name]

        # MVP 카드
        mvp_card = tk.Frame(self._pg_inner, bg=CARD)
        mvp_card.pack(fill="x", padx=14, pady=(8, 4))

        tk.Label(mvp_card, text="⭐ 경기 MVP", bg=CARD, fg=YELLOW,
                 font=("Arial", 12, "bold")).pack(anchor="w", padx=14, pady=(10, 4))
        tk.Frame(mvp_card, bg=BORDER, height=1).pack(fill="x", padx=14)

        mvp_body = tk.Frame(mvp_card, bg=CARD)
        mvp_body.pack(fill="x", padx=14, pady=10)

        # 선수 이미지
        photo, found = self._load_player_image(mvp_name)
        if photo:
            img_lbl = tk.Label(mvp_body, image=photo, bg=CARD)
            img_lbl.pack(side="left", padx=(0, 14))
        else:
            # 이니셜 fallback
            initials = "".join(p[0] for p in mvp_name.split()[:2]) if mvp_name else "?"
            fb = tk.Canvas(mvp_body, width=70, height=70, bg=CARD2,
                            highlightthickness=1, highlightbackground=BORDER)
            fb.create_text(35, 35, text=initials, fill=TEXT, font=("Arial", 22, "bold"))
            fb.pack(side="left", padx=(0, 14))

        info_f = tk.Frame(mvp_body, bg=CARD)
        info_f.pack(side="left")
        tk.Label(info_f, text=mvp_name, bg=CARD, fg=TEXT,
                 font=("Arial", 16, "bold")).pack(anchor="w")
        wpa_col = MINT if mvp_wpa >= 0 else RED
        tk.Label(info_f, text=f"WPA: {'+' if mvp_wpa >= 0 else ''}{mvp_wpa:.3f}",
                 bg=CARD, fg=wpa_col, font=("Arial", 14)).pack(anchor="w")

        # 이 선수의 주요 타석
        key_abs = [(s, p, w) for s, p, w in wpa_plays
                   if s.get("batter") == mvp_name and abs(w) > 0.01]
        key_abs.sort(key=lambda x: abs(x[2]), reverse=True)
        for s, p, w in key_abs[:3]:
            half = "초" if s.get("is_top") else "말"
            evt  = event_kr(s.get("event", ""))
            inn  = _disp_inn(s)
            col  = MINT if w >= 0 else RED
            tk.Label(info_f,
                     text=f"  {inn}회 {half} {evt}  ({'+' if w >= 0 else ''}{w:.3f})",
                     bg=CARD, fg=col, font=("Arial", 11)).pack(anchor="w")

        # WPA 리더보드
        lb_card = tk.Frame(self._pg_inner, bg=CARD)
        lb_card.pack(fill="x", padx=14, pady=4)

        tk.Label(lb_card, text="WPA 리더보드 (Top 10)", bg=CARD, fg=SUBTEXT,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=14, pady=(10, 4))
        tk.Frame(lb_card, bg=BORDER, height=1).pack(fill="x", padx=14)

        sorted_players = sorted(player_wpa.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for rank, (name, wpa) in enumerate(sorted_players, 1):
            row = tk.Frame(lb_card, bg=CARD)
            row.pack(fill="x", padx=14, pady=2)

            rank_col = YELLOW if rank == 1 else (SUBTEXT if rank > 3 else TEXT)
            tk.Label(row, text=f"{rank}.", bg=CARD, fg=rank_col,
                     font=("Arial", 12, "bold"), width=3).pack(side="left")
            tk.Label(row, text=name, bg=CARD, fg=TEXT,
                     font=("Arial", 12)).pack(side="left", padx=(4, 0))
            wpa_col = MINT if wpa >= 0 else RED
            tk.Label(row, text=f"{'+' if wpa >= 0 else ''}{wpa:.3f}",
                     bg=CARD, fg=wpa_col,
                     font=("Arial", 12, "bold")).pack(side="right")

        tk.Frame(lb_card, bg=BG, height=6).pack()

    # ═══════════════════════ 헬퍼 ═════════════════════════════════════════

    def _lbl(self, parent, text, padtop=8):
        tk.Label(parent, text=text.upper(), bg=PANEL, fg=MIDTEXT,
                 font=("Arial", 10, "bold")).pack(
            anchor="w", padx=14, pady=(padtop, 3)
        )

    def _mkbtn(self, parent, text, cmd, bg, fg=None, w=None):
        if fg is None:
            fg = BG if bg in (MINT,) else TEXT
        kw = dict(text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=bg, activeforeground=fg,
                  font=("Arial", 12, "bold"),
                  relief="flat", bd=0, padx=10, pady=7, cursor="hand2")
        if w:
            kw["width"] = w
        return tk.Button(parent, **kw)

    def _style_ax(self):
        self.ax.set_facecolor(CARD2)
        self.ax.spines[:].set_color(BORDER)
        self.ax.tick_params(colors=SUBTEXT, labelsize=11)
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("홈팀 승리 확률 (%)", color=SUBTEXT, fontsize=11)
        self.ax.set_xlabel("타석 진행", color=SUBTEXT, fontsize=11)
        self.ax.axhline(50, color=BORDER, linewidth=1, linestyle="--", alpha=0.7)
        self.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    def _draw_empty_chart(self):
        self._tooltip_ann   = None
        self._tooltip_vline = None
        self.ax.cla()
        self._style_ax()
        self.ax.text(0.5, 0.5, "경기를 선택하고 [경기 분석 시작]을 클릭하세요",
                     transform=self.ax.transAxes,
                     ha="center", va="center", color=SUBTEXT, fontsize=13)
        self.canvas.draw()

    def _draw_empty_linescore(self):
        c = self.linescore_canvas
        c.delete("all")
        c.create_text(10, 40, text="─", fill=SUBTEXT,
                      font=("Arial", 12), anchor="w")

    # 캘린더 팝업

    def _open_calendar(self):
        if hasattr(self, "_cal_win") and self._cal_win and self._cal_win.winfo_exists():
            self._cal_win.destroy()
            return
        try:
            cur = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d").date()
        except ValueError:
            cur = self._kst_today()

        win = tk.Toplevel(self)
        win.title("날짜 선택")
        win.configure(bg=CARD)
        win.resizable(False, False)
        win.grab_set()
        self._cal_win = win
        win.geometry(f"+{self.winfo_x()+20}+{self.winfo_y()+150}")

        tk.Label(win, text=cur.strftime("%-m월 %-d일 (%a)"),
                 bg=CARD, fg=TEXT, font=("Arial", 17, "bold")).pack(
            anchor="w", padx=20, pady=(16, 4)
        )

        cal = Calendar(
            win,
            selectmode="day",
            year=cur.year, month=cur.month, day=cur.day,
            date_pattern="yyyy-mm-dd",
            background=CARD, foreground=TEXT,
            headersbackground=CARD, headersforeground=SUBTEXT,
            selectbackground=MINT, selectforeground=BG,
            normalbackground=CARD, normalforeground=TEXT,
            weekendbackground=CARD, weekendforeground=TEXT,
            othermonthbackground=BG, othermonthforeground=BORDER,
            bordercolor=BORDER,
            font=("Arial", 12),
        )
        cal.pack(padx=16, pady=(0, 8))

        def on_select():
            self.date_var.set(cal.get_date())
            win.destroy()
            self._fetch_games()

        bf = tk.Frame(win, bg=CARD)
        bf.pack(fill="x", padx=16, pady=(0, 16))
        tk.Button(bf, text="취소", command=win.destroy,
                  bg=CARD2, fg=SUBTEXT, font=("Arial", 12), relief="flat",
                  padx=16, pady=7, cursor="hand2").pack(side="right", padx=(4, 0))
        tk.Button(bf, text="확인", command=on_select,
                  bg=MINT, fg=BG, font=("Arial", 12, "bold"), relief="flat",
                  padx=16, pady=7, cursor="hand2").pack(side="right")

    # ═══════════════════════ 이벤트 핸들러 ════════════════════════════════

    def _fetch_games(self):
        date = self.date_var.get().strip()
        for w in self.games_inner.winfo_children():
            w.destroy()
        self._game_data = []
        self._selected_game_idx = None
        self._game_card_frames  = []
        self.status_lbl.config(text="● 로딩 중...", fg=YELLOW)
        self.update()

        def worker():
            try:
                games = get_schedule_kst(date)
                self.after(0, lambda: self._populate_game_cards(games))
            except Exception as e:
                self.after(0, lambda: (
                    messagebox.showerror("오류", f"로드 실패:\n{e}"),
                    self.status_lbl.config(text="● 로드 실패", fg=RED),
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_game_cards(self, games):
        self._game_data = games
        for w in self.games_inner.winfo_children():
            w.destroy()
        self._game_card_frames = []

        for i, g in enumerate(games):
            card = self._make_game_card(self.games_inner, g, i)
            card.pack(fill="x", pady=3, padx=2)
            self._game_card_frames.append(card)

        self.games_inner.update_idletasks()
        self.games_canvas.configure(
            scrollregion=self.games_canvas.bbox("all")
        )

        n = len(games)
        self.status_lbl.config(
            text=f"● {n}경기 로드 완료" if n else "● 경기 없음",
            fg=MINT if n else SUBTEXT,
        )

    def _make_game_card(self, parent, g, idx):
        is_final = "Final" in g.get("status", "")
        is_live  = is_game_live(g)

        card = tk.Frame(parent, bg=CARD, cursor="hand2")

        def on_click(event=None, i=idx):
            self._select_game_card(i)

        def _bind_all(w):
            w.bind("<Button-1>", on_click)
            for c in w.winfo_children():
                _bind_all(c)

        card.bind("<Button-1>", on_click)

        # 상태 + 구장 행
        top = tk.Frame(card, bg=CARD)
        top.pack(fill="x", padx=12, pady=(8, 2))

        if is_live:
            st_txt, st_fg = "● LIVE", RED
        elif is_final:
            st_txt, st_fg = "종료", SUBTEXT
        else:
            st_txt, st_fg = "예정", SUBTEXT

        tk.Label(top, text=st_txt, bg=CARD, fg=st_fg,
                 font=("Arial", 10, "bold")).pack(side="left")
        venue = g.get("venue", "")
        if venue:
            tk.Label(top, text=venue[:18], bg=CARD, fg=SUBTEXT,
                     font=("Arial", 10)).pack(side="right")

        # 팀명 + 점수 행
        mid = tk.Frame(card, bg=CARD)
        mid.pack(fill="x", padx=12, pady=(2, 8))
        mid.columnconfigure(1, weight=1)

        away_s = str(g.get("away_score", 0)) if (is_final or is_live) else "·"
        home_s = str(g.get("home_score", 0)) if (is_final or is_live) else "·"

        if is_final:
            a_win  = g.get("away_score", 0) > g.get("home_score", 0)
            h_win  = g.get("home_score", 0) > g.get("away_score", 0)
            away_col = TEXT   if a_win else SUBTEXT
            home_col = TEXT   if h_win else SUBTEXT
            away_wt  = "bold" if a_win else "normal"
            home_wt  = "bold" if h_win else "normal"
        else:
            away_col = home_col = MIDTEXT
            away_wt  = home_wt  = "normal"

        tk.Label(mid, text=team_kr(g.get("away_team", "")), bg=CARD, fg=away_col,
                 font=("Arial", 12, away_wt), anchor="w").grid(
            row=0, column=0, sticky="w", pady=2)
        tk.Label(mid, text=away_s, bg=CARD, fg=away_col,
                 font=("Arial", 17, "bold"), anchor="e",
                 width=3).grid(row=0, column=2, sticky="e")

        tk.Label(mid, text=team_kr(g.get("home_team", "")), bg=CARD, fg=home_col,
                 font=("Arial", 12, home_wt), anchor="w").grid(
            row=1, column=0, sticky="w", pady=2)
        tk.Label(mid, text=home_s, bg=CARD, fg=home_col,
                 font=("Arial", 17, "bold"), anchor="e",
                 width=3).grid(row=1, column=2, sticky="e")

        # 승자 표시
        if is_final:
            win_row = 1 if g.get("home_score", 0) > g.get("away_score", 0) else 0
            tk.Label(mid, text="◀", bg=CARD, fg=MINT,
                     font=("Arial", 10)).grid(row=win_row, column=1)

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x")
        _bind_all(card)
        return card

    _SEL_CARD = "#2e3a2e"   # 선택된 카드 배경 (어두운 민트 틴트)

    def _select_game_card(self, idx):
        self._selected_game_idx = idx
        for i, card in enumerate(self._game_card_frames):
            bg = self._SEL_CARD if i == idx else CARD
            card.config(bg=bg)
            for w in card.winfo_children():
                self._set_bg_recursive(w, bg)
        if self._game_data:
            game   = self._game_data[idx]
            status = game.get("status", "")
            self._draw_scoreboard_preview(game)
            # 경기 전이고 pregame 모듈 사용 가능이면 자동 예측
            if (
                _PREGAME_OK
                and "Final"     not in status
                and "Completed" not in status
                and not is_game_live(game)
                and game.get("home_team_id")
                and game.get("away_team_id")
            ):
                self._run_pregame_analysis(game)
            else:
                self._hide_pregame_panel()

    def _set_bg_recursive(self, widget, color):
        try:
            widget.config(bg=color)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._set_bg_recursive(child, color)

    def _analyze_game(self):
        if not self.model_ready:
            messagebox.showwarning("모델 없음",
                "train.py를 먼저 실행하세요:\n  python train.py --games 100")
            return

        if self._selected_game_idx is None or not self._game_data:
            messagebox.showinfo("안내", "먼저 경기를 선택하세요.")
            return

        game   = self._game_data[self._selected_game_idx]
        status = game.get("status", "")

        if is_game_live(game):
            if messagebox.askyesno(
                "진행 중인 경기",
                f"'{game.get('away_team')} vs {game.get('home_team')}' 은 현재 진행 중입니다.\n\n"
                "🔴 LIVE 중계 모드로 시작하시겠습니까?",
            ):
                self.current_game_pk = game["gamePk"]
                self._start_live_mode()
                return

        if "Final" not in status and "Completed" not in status and not is_game_live(game):
            if _PREGAME_OK and game.get("home_team_id") and game.get("away_team_id"):
                # 경기 전 → pregame 예측 패널로 전환
                self._switch_tab("승률차트")
                self._run_pregame_analysis(game)
            else:
                messagebox.showinfo("안내", f"아직 시작 전입니다.\n상태: {status}")
            return

        self._stop_live_mode()
        self._stop_anim()
        self.current_game_pk = game["gamePk"]
        self.status_lbl.config(text="● 데이터 로딩 중...", fg=YELLOW)
        self.update()

        def worker():
            try:
                feed   = get_game_feed(game["gamePk"])
                states = extract_game_states(feed)
                if not states:
                    self.after(0, lambda: (
                        messagebox.showerror("오류", "경기 데이터를 파싱할 수 없습니다."),
                        self.status_lbl.config(text="● 파싱 실패", fg=RED),
                    ))
                    return
                probs = self.model.predict_batch(states)
                ls    = feed.get("liveData", {}).get("linescore", {})
                self.after(0, lambda: self._load_game(game, states, probs, ls, feed))
            except Exception as e:
                self.after(0, lambda: (
                    messagebox.showerror("오류", f"분석 실패:\n{e}"),
                    self.status_lbl.config(text="● 오류", fg=RED),
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _load_game(self, game, states, probs, linescore=None, feed=None):
        # pregame 패널 숨기고 차트 올리기
        self._hide_pregame_panel()
        self.game_states  = states
        self.win_probs    = [p * 100 for p in probs]
        self._linescore   = linescore
        self._last_feed   = feed
        self.anim_idx     = 0

        self._away_team_name = game.get("away_team", "원정팀")
        self._home_team_name = game.get("home_team", "홈팀")

        self._update_scoreboard(states[-1] if states else {}, game, linescore)
        self._update_linescore(linescore, states)
        self._update_key_plays(states, probs)
        self._populate_tree(states, probs)
        self._update_player_stats(states)
        self._draw_chart(len(states))
        self._draw_summary_chart()
        self._update_extra_info(feed, game, states)
        self._update_mvp_card(states, probs)
        self.status_lbl.config(text=f"● 분석 완료 ({len(states)} 타석)", fg=MINT)

        if self._ml_debug_win is not None:
            self._ml_debug_win.set_history(states, probs)

    # ═══════════════════════ 스코어보드 ═══════════════════════════════════

    def _draw_scoreboard_preview(self, game):
        """경기 카드 선택 시 스코어보드 위젯 갱신"""
        away_team  = game.get("away_team", "원정팀")
        home_team  = game.get("home_team", "홈팀")
        away_score = game.get("away_score", 0)
        home_score = game.get("home_score", 0)
        status     = game.get("status", "")

        is_final = "Final" in status
        is_live  = is_game_live(game)

        # 색상
        if is_final:
            away_col = TEXT if away_score >= home_score else SUBTEXT
            home_col = TEXT if home_score >= away_score else SUBTEXT
        else:
            away_col = home_col = TEXT

        score_str = lambda s: str(s) if (is_final or is_live) else "─"

        # 원정팀 갱신
        self._draw_logo_canvas(self._sb_away_logo, away_team, 52)
        self._sb_away_score.config(text=score_str(away_score), fg=away_col)
        self._sb_away_name.config(text=team_kr(away_team), fg=away_col)
        self._sb_away_pitcher.config(text="")

        # 홈팀 갱신
        self._draw_logo_canvas(self._sb_home_logo, home_team, 52)
        self._sb_home_score.config(text=score_str(home_score), fg=home_col)
        self._sb_home_name.config(text=team_kr(home_team), fg=home_col)
        self._sb_home_pitcher.config(text="")

        # 상태 표시
        if is_final:
            self._sb_status_lbl.config(text="경기 종료", fg=SUBTEXT)
            self._sb_away_prob.config(text="")
            self._sb_home_prob.config(text="")
        elif is_live:
            self._sb_status_lbl.config(text="● LIVE", fg=RED)
            self._sb_away_prob.config(text="")
            self._sb_home_prob.config(text="")
        else:
            self._sb_status_lbl.config(text="경기 전", fg=SUBTEXT)
            # 승률 라벨은 _apply_pregame_result 에서 채움

    def _update_scoreboard(self, state, game, linescore=None):
        away_team  = game.get("away_team") or self._away_team_name
        home_team  = game.get("home_team") or self._home_team_name
        away_score = state.get("away_score", game.get("away_score", 0))
        home_score = state.get("home_score", game.get("home_score", 0))

        self._draw_logo_canvas(self._sb_away_logo, away_team, 52)
        self._sb_away_score.config(text=str(away_score), fg=TEXT)
        self._sb_away_name.config(text=team_kr(away_team), fg=TEXT)
        if self._away_pitcher:
            self._sb_away_pitcher.config(text=f"P: {self._away_pitcher}")

        self._draw_logo_canvas(self._sb_home_logo, home_team, 52)
        self._sb_home_score.config(text=str(home_score), fg=TEXT)
        self._sb_home_name.config(text=team_kr(home_team), fg=TEXT)
        if self._home_pitcher:
            self._sb_home_pitcher.config(text=f"P: {self._home_pitcher}")

        # 이닝 정보가 있으면 vs 대신 표시
        if linescore:
            inn = linescore.get("currentInning", "")
            half = linescore.get("inningHalf", "")
            half_kr = "초" if "Top" in str(half) else "말" if "Bottom" in str(half) else ""
            if inn:
                self._sb_vs_lbl.config(text=f"{inn}회 {half_kr}")
            else:
                self._sb_vs_lbl.config(text="vs")
        else:
            self._sb_vs_lbl.config(text="vs")

    # 이닝별 득점표

    def _update_linescore(self, linescore, states):
        c = self.linescore_canvas
        c.delete("all")
        if not linescore:
            self._draw_empty_linescore()
            return

        innings_data = linescore.get("innings", [])

        # innings_data가 없어도 game_states로 최대 이닝 추론
        max_inn_from_states = 0
        if states:
            max_inn_from_states = max((_disp_inn(s) for s in states), default=0)

        if not innings_data and max_inn_from_states == 0:
            self._draw_empty_linescore()
            return

        home_totals = linescore.get("teams", {}).get("home", {})
        away_totals = linescore.get("teams", {}).get("away", {})

        away_name = team_kr(states[0].get("away_team", "원정")) if states else "원정"
        home_name = team_kr(states[0].get("home_team", "홈")) if states else "홈"

        # 연장전 포함: 실제 이닝 수, 최소 9이닝 보장
        n_inn  = max(len(innings_data), max_inn_from_states, 9)
        col_w  = 26        # 이닝당 칸 너비 (좁게 해서 연장전도 잘 들어오게)
        name_w = 86
        pad_x  = 4
        row_h  = 24
        hdr_h  = 18

        total_w = name_w + (n_inn + 3) * col_w + pad_x * 2
        total_h = hdr_h + row_h * 2 + 4
        c.config(height=total_h)
        # scrollregion 설정 → 연장전도 스크롤로 접근 가능
        c.config(scrollregion=(0, 0, total_w, total_h))

        def cx(col):
            return name_w + col * col_w + col_w // 2

        def draw_cell(col, row, text, fg=TEXT, bold=False, special=False):
            x = cx(col)
            y = hdr_h + row * row_h + row_h // 2
            if special:
                c.create_rectangle(x - col_w//2 + 1, y - row_h//2 + 1,
                                   x + col_w//2 - 1, y + row_h//2 - 1,
                                   fill=CARD2, outline="")
            fnt = ("Arial", 11, "bold") if bold else ("Arial", 11)
            c.create_text(x, y, text=str(text), fill=fg, font=fnt, anchor="center")

        # 헤더: 이닝 번호
        for i in range(n_inn):
            # 연장전 이닝(10회 이상)은 다른 색으로 강조
            fg_inn = YELLOW if i >= 9 else SUBTEXT
            c.create_text(cx(i), hdr_h // 2, text=str(i + 1),
                          fill=fg_inn, font=("Arial", 10, "bold" if i >= 9 else "normal"),
                          anchor="center")
        for j, lbl in enumerate(["R", "H", "E"]):
            c.create_text(cx(n_inn + j), hdr_h // 2, text=lbl,
                          fill=SUBTEXT, font=("Arial", 10, "bold"), anchor="center")

        # 팀 이름
        for row, name in enumerate([away_name[:8], home_name[:8]]):
            y = hdr_h + row * row_h + row_h // 2
            c.create_text(pad_x + 4, y, text=name, fill=TEXT,
                          font=("Arial", 11), anchor="w")

        # 이닝별 득점
        for i, inn in enumerate(innings_data):
            a_r = inn.get("away", {}).get("runs", "")
            h_r = inn.get("home", {}).get("runs", "")
            a_fg = MINT if isinstance(a_r, int) and a_r > 0 else TEXT
            h_fg = MINT if isinstance(h_r, int) and h_r > 0 else TEXT
            draw_cell(i, 0, a_r if a_r != "" else "-", fg=a_fg)
            draw_cell(i, 1, h_r if h_r != "" else "-", fg=h_fg)

        # 아직 진행 안 된 이닝
        for i in range(len(innings_data), n_inn):
            draw_cell(i, 0, "-", fg=SUBTEXT)
            draw_cell(i, 1, "-", fg=SUBTEXT)

        # R H E 합계 칸
        for row, totals in enumerate([away_totals, home_totals]):
            r = totals.get("runs",   0)
            h = totals.get("hits",   0)
            e = totals.get("errors", 0)
            draw_cell(n_inn,     row, r, fg=TEXT,    bold=True, special=True)
            draw_cell(n_inn + 1, row, h, fg=SUBTEXT, special=True)
            draw_cell(n_inn + 2, row, e,
                      fg=RED if isinstance(e, int) and e > 0 else SUBTEXT,
                      special=True)

        # 구분선
        c.create_line(0, hdr_h, total_w, hdr_h, fill=BORDER, width=1)
        c.create_line(0, hdr_h + row_h, total_w, hdr_h + row_h, fill=BORDER, width=1)
        c.create_line(name_w + n_inn * col_w, hdr_h,
                      name_w + n_inn * col_w, hdr_h + row_h * 2, fill=BORDER, width=1)

        # 9이닝 경계선 (연장전 시각적 구분)
        if n_inn > 9:
            extra_x = name_w + 9 * col_w
            c.create_line(extra_x, 0, extra_x, total_h,
                          fill=YELLOW, width=1, dash=(4, 3))

    # 주요 장면

    def _update_key_plays(self, states, probs):
        for w in self.keyplay_frame.winfo_children():
            w.destroy()

        _key_events = {
            "Home Run", "Grand Slam", "Triple", "Double",
            "Sac Fly", "Walk", "Intent Walk", "Hit By Pitch",
        }

        pairs = list(zip(states, probs))

        # 일단 모든 타석의 WPA(승률 변화량)부터 구해놓기
        plays_wpa = []   # (state, prob, wpa)
        for i, (s, p) in enumerate(pairs):
            if i > 0:
                delta = p - probs[i - 1]
                wpa   = -delta if s.get("is_top") else delta
            else:
                wpa = 0.0
            plays_wpa.append((s, p, wpa))

        # 여기서 홈런/3루타 등 굵직한 이벤트만 골라냄
        key_plays = [(s, p, wpa) for s, p, wpa in plays_wpa
                     if s.get("event") in _key_events]

        if not key_plays:
            key_plays = [(s, p, wpa) for s, p, wpa in plays_wpa
                         if "Home Run" in s.get("event", "")]

        if not key_plays:
            tk.Label(self.keyplay_frame,
                     text="주요 장면 없음",
                     bg=BG, fg=SUBTEXT, font=("Arial", 12)).pack(anchor="w", pady=8)
            return

        # 끝내기처럼 마지막 이닝에 나온 주요 플레이는 빠지면 안 되니까 먼저 챙겨둠
        final_inning = max((_disp_inn(s) for s in states), default=9)
        # 마지막 이닝에 있던 주요 플레이 전부
        must_include = [(s, p, wpa) for s, p, wpa in key_plays
                        if _disp_inn(s) == final_inning]
        # 끝나기 직전 분위기도 보여줘야 해서, 마지막 10타석 안의 주요 이벤트도 같이 포함
        last_states_ids = {id(s) for s, _, _ in plays_wpa[-10:]}
        must_include += [(s, p, wpa) for s, p, wpa in key_plays
                         if id(s) in last_states_ids
                         and (s, p, wpa) not in must_include]

        # 남은 자리는 승률 변화가 큰 순서대로 채워줌
        must_ids = {id(s) for s, _, _ in must_include}
        remaining = [(s, p, wpa) for s, p, wpa in key_plays
                     if id(s) not in must_ids]
        remaining_sorted = sorted(remaining, key=lambda x: abs(x[2]), reverse=True)

        # must_include 를 앞에 두고, 나머지로 총 7개까지 채움
        top = must_include + remaining_sorted
        # 중복 제거 (id 기준)
        seen_ids = set()
        deduped = []
        for item in top:
            if id(item[0]) not in seen_ids:
                seen_ids.add(id(item[0]))
                deduped.append(item)
        top = deduped[:7]   # 최대 7개 표시

        # 시간 순서(이닝 순)로 다시 정렬
        top_sorted = sorted(top, key=lambda x: (
            _disp_inn(x[0]),
            0 if x[0].get("is_top") else 1,
        ))

        # 카드로 그리기
        for s, p, wpa in top_sorted:
            row = tk.Frame(self.keyplay_frame, bg=CARD)
            row.pack(fill="x", pady=2)

            half   = "초" if s.get("is_top") else "말"
            inn    = _disp_inn(s)
            evt_en = s.get("event", "")
            evt_kr_str = event_kr(evt_en)
            batter = s.get("batter", "")
            prob   = p * 100

            # 연장전 이닝 배지
            is_extra = isinstance(inn, int) and inn > 9
            inn_label = f"★{inn}회" if is_extra else f"{inn}회"
            inn_fg    = YELLOW if is_extra else SUBTEXT

            if evt_en in ("Home Run", "Grand Slam"):
                dot_col = RED
            elif evt_en in ("Triple", "Double"):
                dot_col = YELLOW
            else:
                dot_col = MINT

            tk.Label(row, text="●", bg=CARD, fg=dot_col,
                     font=("Arial", 10)).pack(side="left", padx=(10, 6), pady=7)

            info = tk.Frame(row, bg=CARD)
            info.pack(side="left", expand=True)
            tk.Label(info,
                     text=f"{inn_label} {half}  {batter}",
                     bg=CARD, fg=inn_fg, font=("Arial", 11)).pack(anchor="w")
            tk.Label(info, text=evt_kr_str,
                     bg=CARD, fg=dot_col,
                     font=("Arial", 12, "bold")).pack(anchor="w")

            right_f = tk.Frame(row, bg=CARD)
            right_f.pack(side="right", padx=12)
            tk.Label(right_f, text=f"홈 {prob:.1f}%",
                     bg=CARD, fg=MINT if prob >= 50 else RED,
                     font=("Arial", 13, "bold")).pack(anchor="e")
            wpa_col = MINT if wpa >= 0 else RED
            wpa_str = f"WPA {'+' if wpa >= 0 else ''}{wpa:.3f}"
            tk.Label(right_f, text=wpa_str,
                     bg=CARD, fg=wpa_col,
                     font=("Arial", 10)).pack(anchor="e")

    # ═══════════════════════ 차트 ═════════════════════════════════════════

    def _draw_chart(self, up_to=None):
        if not self.game_states:
            self._draw_empty_chart()
            return

        n     = up_to if up_to is not None else len(self.game_states)
        probs = self.win_probs[:n]
        xs    = list(range(1, n + 1))

        # ax.cla() 전에 stale 참조 먼저 무효화
        self._tooltip_ann   = None
        self._tooltip_vline = None

        self.ax.cla()
        self._style_ax()

        home_team = team_kr(self.game_states[0].get("home_team", "홈팀"))
        away_team = team_kr(self.game_states[0].get("away_team", "원정팀"))
        arr       = np.array(probs)

        self.ax.plot(xs, arr, color=MINT, linewidth=2, zorder=3)
        self.ax.fill_between(xs, arr, 50, where=(arr >= 50),
                             color=MINT, alpha=0.2, interpolate=True)
        self.ax.fill_between(xs, arr, 50, where=(arr < 50),
                             color=RED, alpha=0.2, interpolate=True)

        prev_inning = None
        for i, s in enumerate(self.game_states[:n]):
            cur = _disp_inn(s)
            if cur != prev_inning:
                self.ax.axvline(i + 1, color=BORDER, linewidth=0.8, alpha=0.6)
                self.ax.text(i + 1, 97, f"{cur}",
                             color=SUBTEXT, fontsize=9, ha="center", va="top")
                prev_inning = cur

        if probs:
            last   = probs[-1]
            dot_c  = MINT if last >= 50 else RED
            self.ax.scatter([xs[-1]], [last], color=dot_c, s=55, zorder=5)
            self.ax.text(xs[-1], last + 4, f"{last:.1f}%",
                         color=dot_c, fontsize=11, ha="center", fontweight="bold")

        away_p = mpatches.Patch(color=RED,  alpha=0.5, label=f"{away_team} 우세")
        home_p = mpatches.Patch(color=MINT, alpha=0.5, label=f"{home_team} 우세")
        self.ax.legend(handles=[home_p, away_p], loc="upper left",
                       facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=11)

        self.ax.set_xlim(0, max(len(self.game_states) + 1, 10))
        self.fig.tight_layout()
        self.canvas.draw()

    # ═══════════════════════ 타석 로그 ════════════════════════════════════

    def _switch_records_team(self, team):
        self._rec_team_var.set(team)
        self._rec_away_btn.config(
            bg=MINT if team == "away" else CARD2,
            fg=BG   if team == "away" else SUBTEXT)
        self._rec_home_btn.config(
            bg=MINT if team == "home" else CARD2,
            fg=BG   if team == "home" else SUBTEXT)
        away_kr = team_kr(self._away_team_name or "원정팀")
        home_kr = team_kr(self._home_team_name or "홈팀")
        if team == "away":
            self._rec_team_lbl.config(
                text=f"초 공격 — {away_kr} 타석")
        else:
            self._rec_team_lbl.config(
                text=f"말 공격 — {home_kr} 타석")
        # 현재 데이터로 다시 필터링
        if self.game_states and self.win_probs:
            self._populate_tree(self.game_states, self.win_probs)

    def _populate_tree(self, states, probs):
        self.tree.delete(*self.tree.get_children())
        # 팀 필터: away=초(is_top=1), home=말(is_top=0)
        team = getattr(self, "_rec_team_var", None)
        filter_top = 1 if (team is None or team.get() == "away") else 0
        for s, p in zip(states, probs):
            if s.get("is_top", 1) != filter_top:
                continue
            inn = f"{'초' if s.get('is_top') else '말'}{_disp_inn(s)}회"
            self.tree.insert("", "end", values=(
                inn,
                s.get("batter",  "-")[:16],
                s.get("pitcher", "-")[:16],
                event_kr(s.get("event", "-"))[:22],
                s.get("home_score",  0),
                s.get("away_score",  0),
                f"{p * 100:.1f}%",
            ))

    # ═══════════════════════ 개인 기록 집계 ══════════════════════════════

    def _calc_batter_stats(self, states, is_top_filter):
        from collections import defaultdict
        HIT_EVENTS   = {"Single", "Double", "Triple", "Home Run"}
        NO_AB_EVENTS = {"Walk", "Intent Walk", "Hit By Pitch",
                        "Sac Fly", "Sac Bunt", "Catcher Interference",
                        "Fan Interference", "Field Error"}
        order_map  = {}   # name → 등장 순서(1-based)
        first_inn  = {}   # name → 첫 등장 이닝
        stats = defaultdict(lambda: {
            "ab": 0, "h": 0, "dbl": 0, "tpl": 0, "hr": 0,
            "k": 0, "bb": 0, "hbp": 0, "rbi": 0, "runs": 0,
            "sb": 0, "gidp": 0, "e": 0,
            "order": 0, "first_inning": 0,
        })
        prev_score = [0, 0]   # [away, home]
        for s in states:
            if s.get("is_top") != is_top_filter:
                hs = s.get("home_score", 0); as_ = s.get("away_score", 0)
                prev_score = [as_, hs]
                continue
            name  = s.get("batter", "").strip()
            event = s.get("event",  "").strip()
            if not name or not event:
                continue
            inn = _disp_inn(s)
            if name not in order_map:
                order_map[name] = len(order_map) + 1
                stats[name]["order"] = order_map[name]
                stats[name]["first_inning"] = inn

            if event not in NO_AB_EVENTS:
                stats[name]["ab"] += 1
            if event in HIT_EVENTS:            stats[name]["h"]    += 1
            if event == "Double":              stats[name]["dbl"]  += 1
            if event == "Triple":              stats[name]["tpl"]  += 1
            if event == "Home Run":            stats[name]["hr"]   += 1
            if "Strikeout" in event:           stats[name]["k"]    += 1
            if event in ("Walk","Intent Walk"):stats[name]["bb"]   += 1
            if event == "Hit By Pitch":        stats[name]["hbp"]  += 1
            if event == "Stolen Base":         stats[name]["sb"]   += 1
            if "Grounded Into DP" in event or event == "Double Play":
                                               stats[name]["gidp"] += 1
            if event == "Field Error":         stats[name]["e"]    += 1

            hs  = s.get("home_score", 0)
            as_ = s.get("away_score", 0)
            rbi = (as_ - prev_score[0]) if is_top_filter == 1 else (hs - prev_score[1])
            if rbi > 0:
                stats[name]["rbi"] += rbi
            prev_score = [as_, hs]

        return stats

    def _build_lineup_slots(self, states, is_top_filter):
        """
        타순 슬롯(1-9) 별로 선수 목록을 반환.
        boxscore 가 있으면 실제 배팅오더 사용, 없으면 states 순서로 9개 슬롯 추정.
        반환: [{"slot":1, "players":[{"name":str,"pos":str,"first_inning":int,"stats":dict}]}]
        """
        bstats = self._calc_batter_stats(states, is_top_filter)
        side   = "away" if is_top_filter == 1 else "home"

        # boxscore 에서 실제 배팅오더 시도
        feed    = getattr(self, "_last_feed", None)
        bx_players = {}   # fullName → {slot, pos, is_starter}
        if feed:
            try:
                bx_team = (feed.get("liveData", {})
                               .get("boxscore", {})
                               .get("teams", {})
                               .get(side, {})
                               .get("players", {}))
                for pid, pdata in bx_team.items():
                    bo_raw = pdata.get("battingOrder", "")
                    if not bo_raw:
                        continue
                    try:
                        bo = int(bo_raw)
                    except ValueError:
                        continue
                    slot       = bo // 100
                    is_starter = (bo % 100 == 0)
                    fname      = pdata.get("person", {}).get("fullName", "")
                    pos        = pdata.get("position", {}).get("abbreviation", "")
                    if slot > 0 and fname:
                        bx_players[fname] = {
                            "slot": slot, "pos": pos,
                            "is_starter": is_starter, "bo": bo,
                        }
            except Exception:
                pass

        # 슬롯 구성
        slots = {i: [] for i in range(1, 10)}

        if bx_players:
            # boxscore 기반: bo 순서대로 같은 slot에 쌓음
            for name, bx in sorted(bx_players.items(),
                                   key=lambda x: x[1]["bo"]):
                slot = bx["slot"]
                if slot < 1 or slot > 9:
                    continue
                st = bstats.get(name, {
                    "ab":0,"h":0,"hr":0,"rbi":0,"runs":0,
                    "sb":0,"bb":0,"k":0,"first_inning":0,
                })
                slots[slot].append({
                    "name": name,
                    "pos":  bx["pos"],
                    "is_starter": bx["is_starter"],
                    "first_inning": st.get("first_inning", 0),
                    "stats": st,
                })
        else:
            # fallback: 등장 순서 기반
            sorted_p = sorted(bstats.items(), key=lambda x: x[1]["order"])
            for i, (name, st) in enumerate(sorted_p):
                slot = (i % 9) + 1
                slots[slot].append({
                    "name": name,
                    "pos":  "",
                    "is_starter": (i < 9),
                    "first_inning": st.get("first_inning", 0),
                    "stats": st,
                })

        # boxscore 타자 스탯으로 보정 (더 정확)
        if feed:
            try:
                bx_team = (feed.get("liveData",{})
                               .get("boxscore",{})
                               .get("teams",{})
                               .get(side,{})
                               .get("players",{}))
                for pid, pdata in bx_team.items():
                    fname = pdata.get("person",{}).get("fullName","")
                    bx_b  = pdata.get("stats",{}).get("batting",{})
                    if fname in bstats and bx_b:
                        bstats[fname]["ab"]   = bx_b.get("atBats",        bstats[fname]["ab"])
                        bstats[fname]["h"]    = bx_b.get("hits",          bstats[fname]["h"])
                        bstats[fname]["hr"]   = bx_b.get("homeRuns",      bstats[fname]["hr"])
                        bstats[fname]["rbi"]  = bx_b.get("rbi",           bstats[fname]["rbi"])
                        bstats[fname]["runs"] = bx_b.get("runs",          bstats[fname]["runs"])
                        bstats[fname]["bb"]   = bx_b.get("baseOnBalls",   bstats[fname]["bb"])
                        bstats[fname]["k"]    = bx_b.get("strikeOuts",    bstats[fname]["k"])
                        bstats[fname]["hbp"]  = bx_b.get("hitByPitch",    bstats[fname]["hbp"])
                        bstats[fname]["sb"]   = bx_b.get("stolenBases",   bstats[fname]["sb"])
                        bstats[fname]["gidp"] = bx_b.get("groundIntoDoublePlay", bstats[fname]["gidp"])
            except Exception:
                pass

        return [{"slot": s, "players": slots[s]}
                for s in range(1, 10) if slots[s]]

    def _get_pitcher_data_from_feed(self, side: str) -> tuple:
        """
        boxscore에서 투수 기록 추출.
        Returns: (decisions_dict, pitcher_list)
          decisions: {"winner":name, "loser":name, "save":name, "hold":[name,...]}
          pitcher_list: [{"name", "decision", "ip", "pitches", "bf", "h", "hr",
                          "bb", "hbp", "k", "r", "er", "balls", "strikes"}, ...]
        """
        feed = getattr(self, "_last_feed", None)
        if not feed:
            return {}, []
        try:
            live       = feed.get("liveData", {})
            bx         = live.get("boxscore", {})
            bx_team    = bx.get("teams", {}).get(side, {})
            players    = bx_team.get("players", {})
            pit_order  = bx_team.get("pitchers", [])   # [id, id, ...]
            dec_raw    = live.get("decisions", {})

            # 결정 (W/L/S)
            dec = {}
            for role in ("winner", "loser", "save"):
                p = dec_raw.get(role)
                if p:
                    dec[role] = p.get("fullName", "")
                    dec[f"{role}_id"] = p.get("id")

            pitchers = []
            for pid_int in pit_order:
                pid_str = f"ID{pid_int}"
                pdata   = players.get(pid_str, {})
                fname   = pdata.get("person", {}).get("fullName", "")
                pst     = pdata.get("stats", {}).get("pitching", {})

                # 결과 결정
                decision = ""
                if fname == dec.get("winner"):  decision = "W"
                elif fname == dec.get("loser"):  decision = "L"
                elif fname == dec.get("save"):   decision = "S"
                elif pst.get("holds",       0) > 0: decision = "H"
                elif pst.get("blownSaves",  0) > 0: decision = "BS"

                balls   = pst.get("balls",          0) or 0
                strikes = pst.get("strikes",         0) or 0
                pitches = pst.get("numberOfPitches", 0) or 0

                pitchers.append({
                    "name":     fname,
                    "decision": decision,
                    "ip":       pst.get("inningsPitched", "0.0"),
                    "pitches":  pitches,
                    "balls":    balls,
                    "strikes":  strikes,
                    "bf":       pst.get("battersFaced",  0) or 0,
                    "h":        pst.get("hits",          0) or 0,
                    "hr":       pst.get("homeRuns",      0) or 0,
                    "bb":       pst.get("baseOnBalls",   0) or 0,
                    "hbp":      pst.get("hitBatsmen",    0) or 0,
                    "k":        pst.get("strikeOuts",    0) or 0,
                    "r":        pst.get("runs",          0) or 0,
                    "er":       pst.get("earnedRuns",    0) or 0,
                })
            return dec, pitchers
        except Exception:
            return {}, []

    def _calc_pitcher_stats(self, states, is_top_filter):
        from collections import defaultdict
        HIT_EVENTS = {"Single", "Double", "Triple", "Home Run"}
        stats = defaultdict(lambda: {"outs":0,"hits":0,"runs":0,
                                      "k":0,"bb":0,"order":99})
        order_map = {}
        prev_score = {}
        for s in states:
            if s.get("is_top") != is_top_filter:
                continue
            name  = s.get("pitcher","").strip()
            event = s.get("event",  "").strip()
            if not name or not event:
                continue
            if name not in order_map:
                order_map[name] = len(order_map) + 1
                stats[name]["order"] = order_map[name]
            if event in HIT_EVENTS:
                stats[name]["hits"] += 1
            if "Strikeout" in event:
                stats[name]["k"] += 1; stats[name]["outs"] += 1
            elif any(kw in event for kw in ("out","Out","DP","Play","Bunt")):
                stats[name]["outs"] += 1
            if event in ("Walk","Intent Walk"):
                stats[name]["bb"] += 1
            hs  = s.get("home_score", 0)
            as_ = s.get("away_score", 0)
            key = (hs, as_)
            if name in prev_score and prev_score[name] != key:
                old  = prev_score[name]
                diff = (hs - old[0]) + (as_ - old[1])
                if diff > 0:
                    stats[name]["runs"] += diff
            prev_score[name] = key
        return stats

    def _update_player_stats(self, states):
        team   = self._ps_team.get()
        is_top = 1 if team == "away" else 0
        slots  = self._build_lineup_slots(states, is_top)
        # 해당 팀의 투수 (MIL 탭 → MIL 투수, SF 탭 → SF 투수)
        pit_side = team
        dec, pitchers = self._get_pitcher_data_from_feed(pit_side)

        for w in self._ps_inner.winfo_children():
            w.destroy()

        SW = 4   # stat column char width

        # ps_cv 참조 (스크롤 전파용)
        _ps_cv = self._ps_inner.master   # Canvas

        _ps_scroll = _make_wheel_scroll(_ps_cv)

        def _bind_scroll(widget):
            """위젯 및 자식 전체에 마우스 휠 이벤트 전파"""
            widget.bind("<MouseWheel>", _ps_scroll)
            for child in widget.winfo_children():
                _bind_scroll(child)

        # 스탯 숫자 셀 헬퍼
        def _stat(parent, text, fg=TEXT, bold=False, bg=CARD):
            tk.Label(parent, text=str(text), bg=bg, fg=fg,
                     font=("Arial", 11, "bold") if bold else ("Arial", 11),
                     width=SW, anchor="center").pack(side="left", ipady=5)

        # 타자 카드 한 줄
        def _batter_card(parent, slot_no, player, is_starter, bg_card):
            name      = player["name"]
            pos       = player["pos"]
            first_inn = player["first_inning"]
            st        = player["stats"]

            border_color = MINT if is_starter else "#444444"
            card = tk.Frame(parent, bg=bg_card,
                            highlightbackground=border_color,
                            highlightthickness=1)
            card.pack(fill="x", padx=(4, 6), pady=(0, 1))

            # 왼쪽 컬러 바
            tk.Frame(card, bg=MINT if is_starter else "#555555",
                     width=4).pack(side="left", fill="y")

            # 타순 번호
            tk.Label(card, text=str(slot_no) if is_starter else "─",
                     bg=bg_card,
                     fg=MINT if is_starter else SUBTEXT,
                     font=("Arial", 12, "bold"), width=2,
                     anchor="center").pack(side="left", padx=(5, 2), ipady=5)

            # 포지션
            pos_txt = pos if pos else ("PH" if not is_starter else "")
            tk.Label(card, text=pos_txt, bg=bg_card, fg=SUBTEXT,
                     font=("Arial", 10), width=3,
                     anchor="center").pack(side="left", padx=(0, 3))

            # 세로 구분선
            tk.Frame(card, bg=BORDER, width=1).pack(side="left", fill="y", pady=3)

            # 이름 영역
            name_f = tk.Frame(card, bg=bg_card)
            name_f.pack(side="left", fill="x", expand=True, padx=(8, 4))

            tk.Label(name_f, text=name[:22],
                     bg=bg_card,
                     fg=TEXT if is_starter else SUBTEXT,
                     font=("Arial", 12, "bold") if is_starter else ("Arial", 11),
                     anchor="w").pack(anchor="w", pady=(3, 0))

            if not is_starter and first_inn:
                tk.Label(name_f, text=f"{first_inn}회 교체 출전",
                         bg=bg_card, fg="#777777",
                         font=("Arial", 9)).pack(anchor="w", pady=(0, 3))
            else:
                tk.Frame(name_f, bg=bg_card, height=3).pack()

            # 타자 스탯 11개
            h  = st.get("h",  0); hr = st.get("hr", 0)
            _stat(card, st.get("ab",   0), bg=bg_card)
            _stat(card, st.get("runs", 0), bg=bg_card)
            _stat(card, h,  fg=MINT   if h  > 0 else TEXT, bold=(h  > 0), bg=bg_card)
            _stat(card, hr, fg=YELLOW if hr > 0 else TEXT, bold=(hr > 0), bg=bg_card)
            _stat(card, st.get("rbi",  0), bg=bg_card)
            _stat(card, st.get("sb",   0), bg=bg_card)
            _stat(card, st.get("bb",   0), bg=bg_card)
            _stat(card, st.get("hbp",  0), bg=bg_card)
            _stat(card, st.get("k",    0), fg=SUBTEXT if st.get("k",0)>0 else TEXT, bg=bg_card)
            _stat(card, st.get("gidp", 0), bg=bg_card)
            _stat(card, st.get("e",    0), fg=RED if st.get("e",0)>0 else TEXT, bg=bg_card)

        # 타자 섹션
        if slots:
            for slot_data in slots:
                slot_no = slot_data["slot"]
                for pi, player in enumerate(slot_data["players"]):
                    is_starter = (pi == 0)
                    bg_card    = CARD if slot_no % 2 == 0 else CARD2
                    if not is_starter:
                        # 대타: 들여쓰기 + 연결선
                        row = tk.Frame(self._ps_inner, bg=BG)
                        row.pack(fill="x")
                        tk.Label(row, text="  └", bg=BG, fg=SUBTEXT,
                                 font=("Arial", 11)).pack(side="left")
                        _batter_card(row, slot_no, player,
                                     is_starter=False, bg_card="#282828")
                    else:
                        _batter_card(self._ps_inner, slot_no, player,
                                     is_starter=True, bg_card=bg_card)
        else:
            tk.Label(self._ps_inner, text="타자 데이터 없음",
                     bg=CARD, fg=SUBTEXT, font=("Arial", 11)).pack(pady=16)

        # 모든 타자 카드에 스크롤 이벤트 전파
        _bind_scroll(self._ps_inner)

        # 투수 섹션 구분선
        div = tk.Frame(self._ps_inner, bg=BG)
        div.pack(fill="x", pady=(12, 6), padx=6)
        tk.Frame(div, bg=BORDER, height=1).pack(fill="x", pady=(0, 4))

        # 투수 타이틀 + 결과 뱃지 한 줄
        pit_title_row = tk.Frame(self._ps_inner, bg=BG)
        pit_title_row.pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(pit_title_row, text="투수", bg=BG, fg=TEXT,
                 font=("Arial", 14, "bold")).pack(side="left")
        if dec:
            parts = []
            if dec.get("winner"): parts.append(f"승  {dec['winner'].split()[-1]}")
            if dec.get("loser"):  parts.append(f"패  {dec['loser'].split()[-1]}")
            if dec.get("save"):   parts.append(f"세이브  {dec['save'].split()[-1]}")
            if parts:
                tk.Label(pit_title_row,
                         text="   ·   ".join(parts),
                         bg=BG, fg=SUBTEXT,
                         font=("Arial", 11)).pack(side="left", padx=12)

        # 투수 컬럼 헤더
        pit_hdr = tk.Frame(self._ps_inner, bg=CARD2)
        pit_hdr.pack(fill="x", padx=4, pady=(0, 2))
        tk.Label(pit_hdr, text="결과", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 10, "bold"), width=4,
                 anchor="center").pack(side="left", padx=(4, 0), pady=3)
        tk.Frame(pit_hdr, bg=BORDER, width=1).pack(side="left", fill="y", pady=2)
        tk.Label(pit_hdr, text="이름", bg=CARD2, fg=SUBTEXT,
                 font=("Arial", 10, "bold"),
                 anchor="w").pack(side="left", fill="x", expand=True, padx=8)
        for lbl in ["이닝", "투구수", "상대", "피안", "피홈", "볼넷", "사구", "삼진", "실점"]:
            tk.Label(pit_hdr, text=lbl, bg=CARD2, fg=SUBTEXT,
                     font=("Arial", 10, "bold"), width=SW,
                     anchor="center").pack(side="left", padx=1, pady=3)

        # 투수 카드 렌더링
        if pitchers:
            for pi, p in enumerate(pitchers):
                bg_p    = CARD if pi % 2 == 0 else CARD2
                dec_lbl = p["decision"]
                dec_fg  = (MINT   if dec_lbl == "W" else
                           RED    if dec_lbl == "L" else
                           YELLOW if dec_lbl in ("S", "H", "BS") else SUBTEXT)
                bar_col = (MINT   if dec_lbl == "W" else
                           RED    if dec_lbl == "L" else
                           YELLOW if dec_lbl in ("S", "H", "BS") else "#555555")

                border_col = bar_col if dec_lbl else BORDER
                card = tk.Frame(self._ps_inner, bg=bg_p,
                                highlightbackground=border_col,
                                highlightthickness=1)
                card.pack(fill="x", padx=4, pady=(0, 1))

                # 왼쪽 컬러 바
                tk.Frame(card, bg=bar_col, width=4).pack(side="left", fill="y")

                # 결과 뱃지
                tk.Label(card, text=dec_lbl if dec_lbl else "  ",
                         bg=bg_p, fg=dec_fg,
                         font=("Arial", 12, "bold"), width=3,
                         anchor="center").pack(side="left", padx=(4, 2), ipady=6)

                # 세로 구분선
                tk.Frame(card, bg=BORDER, width=1).pack(side="left", fill="y", pady=3)

                # 이름 영역
                name_f = tk.Frame(card, bg=bg_p)
                name_f.pack(side="left", fill="x", expand=True, padx=(8, 4))
                tk.Label(name_f, text=p["name"][:22],
                         bg=bg_p, fg=TEXT,
                         font=("Arial", 12, "bold"),
                         anchor="w").pack(anchor="w", pady=5)

                # 스탯 9개
                bs_txt = f"{p['pitches']}({p['balls']}-{p['strikes']})" if p["pitches"] else "─"
                _stat(card, p["ip"],  bg=bg_p)
                _stat(card, bs_txt,   bg=bg_p)
                _stat(card, p["bf"],  bg=bg_p)
                _stat(card, p["h"],   bg=bg_p)
                _stat(card, p["hr"],  fg=YELLOW if p["hr"] > 0 else TEXT,
                      bold=(p["hr"] > 0), bg=bg_p)
                _stat(card, p["bb"],  bg=bg_p)
                _stat(card, p["hbp"], bg=bg_p)
                _stat(card, p["k"],   bg=bg_p)
                _stat(card, p["r"],   fg=RED if p["r"] > 0 else TEXT,
                      bold=(p["r"] > 0), bg=bg_p)

                _bind_scroll(card)
        else:
            tk.Label(self._ps_inner,
                     text="투수 데이터 없음 (분석 완료 경기에서 표시)",
                     bg=CARD, fg=SUBTEXT, font=("Arial", 11)).pack(pady=10)

        tk.Frame(self._ps_inner, bg=BG, height=16).pack()

        # 위젯 배치 완료 후 scrollregion 강제 갱신
        self._ps_inner.update_idletasks()
        _ps_cv.configure(scrollregion=_ps_cv.bbox("all"))

    # ═══════════════════════ 애니메이션 ═══════════════════════════════════

    def _start_anim(self):
        if not self.game_states:
            messagebox.showinfo("안내", "먼저 경기를 분석하세요.")
            return
        if self.anim_idx >= len(self.game_states):
            self.anim_idx = 0
        self._switch_tab("승률 차트")
        self._anim_step()

    def _anim_step(self):
        self.anim_idx += 1
        self._draw_chart(self.anim_idx)
        if self.anim_idx < len(self.game_states):
            self.anim_job = self.after(self.anim_speed, self._anim_step)
        else:
            self.status_lbl.config(text="● 애니메이션 완료", fg=MINT)

    def _stop_anim(self):
        if self.anim_job:
            self.after_cancel(self.anim_job)
            self.anim_job = None

    def _reset_anim(self):
        self._stop_anim()
        self.anim_idx = 0
        self._draw_chart(0)

    # ═══════════════════════ ML 실시간 분석 창 ═════════════════════════════

    def _toggle_ml_debug(self):
        if self._ml_debug_win is not None:
            self._ml_debug_win.destroy()
            self._ml_debug_win = None
            return

        if not self.model_ready:
            messagebox.showwarning("모델 없음", "train.py로 모델을 먼저 학습하세요.")
            return

        def on_close():
            self._ml_debug_win = None

        self._ml_debug_win = MLDebugWindow(self, self.model, on_close=on_close)

        print(f"[ml_debug] _toggle_ml_debug: len(game_states)={len(self.game_states)}, "
              f"len(win_probs)={len(self.win_probs)}")

        if self.game_states and self.win_probs:
            probs = [p / 100 for p in self.win_probs]
            self._ml_debug_win.set_history(self.game_states, probs)

    # ═══════════════════════ LIVE 모드 ════════════════════════════════════

    def _toggle_live(self):
        if self.live_mode:
            self._stop_live_mode()
        else:
            if self._selected_game_idx is None or not self._game_data:
                messagebox.showinfo("안내", "먼저 경기를 선택하세요.")
                return
            if not self.model_ready:
                messagebox.showwarning("모델 없음", "train.py로 모델을 먼저 학습하세요.")
                return
            self.current_game_pk = self._game_data[self._selected_game_idx]["gamePk"]
            self._start_live_mode()

    def _start_live_mode(self):
        self._stop_anim()
        self.live_mode = True
        self.live_btn.config(text="⬛  LIVE 중계 정지",
                              bg=CARD2, fg=MIDTEXT,
                              activebackground=BORDER, activeforeground=TEXT)
        self.live_panel.grid()
        self.status_lbl.config(text="● LIVE 연결 중...", fg=RED)
        self._blink_live()
        self._poll_live()

    def _stop_live_mode(self):
        self.live_mode = False
        if self.live_poll_job:
            self.after_cancel(self.live_poll_job)
            self.live_poll_job = None
        self.live_btn.config(text="🔴  LIVE 중계 시작",
                              bg="#3a1212", fg="#ff8080",
                              activebackground="#4a1818", activeforeground=TEXT)
        self.live_panel.grid_remove()

    def _poll_live(self):
        if not self.live_mode or not self.current_game_pk:
            return

        def worker():
            try:
                print(f"[live] _poll_live worker 시작: game_pk={self.current_game_pk}, "
                      f"model_ready={self.model_ready}")
                feed    = get_game_feed(self.current_game_pk)
                ab_info = get_current_at_bat(feed)
                print(f"[live] get_current_at_bat: is_live={ab_info.get('is_live')}, "
                      f"is_final={ab_info.get('is_final')}, "
                      f"abstract_state={ab_info.get('abstract_state')!r}, "
                      f"inning={ab_info.get('inning')}, is_top={ab_info.get('is_top')}, "
                      f"score={ab_info.get('away_score')}:{ab_info.get('home_score')}, "
                      f"balls-strikes-outs={ab_info.get('balls')}-{ab_info.get('strikes')}-{ab_info.get('outs')}")

                states  = extract_game_states(feed)
                print(f"[live] extract_game_states 결과: len(states)={len(states)}")

                if states and self.model_ready:
                    print("[live] predict_batch() 호출 시도")
                    probs = self.model.predict_batch(states)
                else:
                    probs = []
                    print(f"[live] predict_batch() 호출 안됨 -> probs=[] "
                          f"(states_empty={not states}, model_ready={self.model_ready})")

                ls = feed.get("liveData", {}).get("linescore", {})
                print(f"[live] worker 종료: len(states)={len(states)}, len(probs)={len(probs)}")
                self.after(0, lambda: self._apply_live_update(ab_info, states, probs, ls, feed))
            except Exception:
                tb = traceback.format_exc()
                print(f"[live] _poll_live worker 예외 발생:\n{tb}")
                self.after(0, lambda: self.status_lbl.config(
                    text="● LIVE 오류 (콘솔 로그 확인)", fg=YELLOW))

        threading.Thread(target=worker, daemon=True).start()
        if self.live_mode:
            self.live_poll_job = self.after(LIVE_POLL_INTERVAL, self._poll_live)

    def _apply_live_update(self, ab_info, states, probs, linescore, feed=None):
        print(f"[live] _apply_live_update 진입: len(states)={len(states) if states else 0}, "
              f"len(probs)={len(probs) if probs else 0}, "
              f"기존 len(self.win_probs)={len(self.win_probs)}, "
              f"is_final={ab_info.get('is_final')}, is_live={ab_info.get('is_live')}")

        # 경기 종료 처리
        if ab_info.get("is_final"):
            self._stop_live_mode()
            self.status_lbl.config(text="● 경기 종료", fg=MINT)

        # 라이브 패널 항상 갱신
        if states and probs:
            last_prob = probs[-1] * 100
            self.lv_prob_lbl.config(text=f"{last_prob:.1f}%",
                                    fg=MINT if last_prob >= 50 else RED)
            print(f"[main] live win_prob updated -> {last_prob:.1f}%")
        else:
            cur_text = self.lv_prob_lbl.cget("text")
            print(f"[live] lv_prob_lbl 갱신 안됨 (states={'있음' if states else '없음'}, "
                  f"probs={'있음' if probs else '없음'}) -> 현재 표시값='{cur_text}' 유지")

        half = "초" if ab_info.get("is_top") else "말"
        self.lv_inning_lbl.config( text=f"{ab_info.get('inning','?')}회 {half}")
        self.lv_batter_lbl.config( text=f"타자: {ab_info.get('batter','─')}")
        self.lv_pitcher_lbl.config(text=f"투수: {ab_info.get('pitcher','─')}")
        self.lv_balls_lbl.config(  text=str(ab_info.get("balls",   0)))
        self.lv_strikes_lbl.config(text=str(ab_info.get("strikes", 0)))
        self.lv_outs_lbl.config(   text=str(ab_info.get("outs",    0)))
        self._draw_diamond(ab_info.get("runner_1b", 0),
                           ab_info.get("runner_2b", 0),
                           ab_info.get("runner_3b", 0))

        sel       = self._selected_game_idx
        game_meta = self._game_data[sel] if sel is not None and self._game_data else {}

        self._update_scoreboard(
            {"home_score": ab_info.get("home_score", 0),
             "away_score": ab_info.get("away_score", 0)},
            game_meta, linescore,
        )
        self._update_linescore(linescore, states)

        # 모든 탭 항상 갱신 (새 타석 여부 무관)
        if not states:
            print("[live] _apply_live_update: states 비어있음 -> "
                  "win_probs/차트/ML분석 갱신 건너뛰고 return "
                  "(점수/이닝/볼카운트/주자/스코어보드는 위에서 이미 갱신됨)")
            if ab_info.get("is_live"):
                self.status_lbl.config(
                    text=f"● LIVE 대기  |  {LIVE_POLL_INTERVAL//1000}초 후 갱신",
                    fg=YELLOW)
            return

        # probs 없으면 이전 값 재사용, 완전히 없으면 0.5 배열
        if probs:
            win_probs_pct = [p * 100 for p in probs]
            print(f"[live] win_probs_pct: predict_batch 결과 사용 (len={len(win_probs_pct)})")
        elif self.win_probs and len(self.win_probs) == len(states):
            win_probs_pct = self.win_probs
            probs         = [p / 100 for p in win_probs_pct]
            print(f"[live] win_probs_pct: 이전 값 재사용 (len={len(win_probs_pct)}) "
                  f"-> probs가 비어 predict_batch 결과가 아님!")
        else:
            win_probs_pct = [50.0] * len(states)
            probs         = [0.5]  * len(states)
            print(f"[live] win_probs_pct: 50.0% 기본값으로 채움 (len={len(win_probs_pct)}) "
                  f"-> probs도 비어있고 기존 win_probs 길이도 states와 불일치")

        # 값 자체에 None/NaN 있는지 확인
        bad_vals = [(i, v) for i, v in enumerate(win_probs_pct)
                    if v is None or (isinstance(v, float) and np.isnan(v))]
        if bad_vals:
            print(f"[live] win_probs_pct에 None/NaN 발견 ({len(bad_vals)}건, 처음5개): {bad_vals[:5]}")

        self.game_states = states
        self.win_probs   = win_probs_pct
        self._last_feed  = feed

        print(f"[live] 최종 배열 길이: states={len(states)}, probs={len(probs)}, "
              f"win_probs={len(self.win_probs)}")

        # 요약 탭
        self._draw_summary_chart()
        self._update_key_plays(states, probs)

        # 승률 차트 탭 (현재까지 전체 플레이 표시)
        print(f"[live] _draw_chart 호출: up_to={len(states)}, "
              f"chart probs len={len(self.win_probs[:len(states)])}")
        self._draw_chart(len(states))

        # 타석 기록
        self._populate_tree(states, probs)

        # 개인 기록
        self._update_player_stats(states)

        # 기타 정보
        if feed and game_meta:
            self._update_extra_info(feed, game_meta, states)

        # ML 실시간 분석 창
        print(f"[live] _ml_debug_win 갱신 시도: 창 존재={self._ml_debug_win is not None}, "
              f"len(states)={len(states)}, len(probs)={len(probs)}")
        if self._ml_debug_win is not None:
            self._ml_debug_win.set_history(states, probs)

        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_lbl.config(
            text=f"● LIVE  |  {len(states)} 타석  |  {now}", fg=RED)

    def _blink_live(self):
        if not self.live_mode:
            return
        self._blink_on = not self._blink_on
        try:
            self.lv_badge.config(fg=RED if self._blink_on else CARD2)
        except Exception:
            return
        self.after(700, self._blink_live)

    # ═══════════════════════ 순위 팝업 ════════════════════════════════════

    def _open_standings(self):
        """MLB 순위 팝업 창 열기"""
        # 이미 열려있으면 앞으로
        if hasattr(self, "_standings_win") and self._standings_win and \
                self._standings_win.winfo_exists():
            self._standings_win.lift()
            return

        win = tk.Toplevel(self)
        win.title("MLB 리그 순위")
        win.geometry("820x640")
        win.configure(bg=BG)
        win.resizable(True, True)
        self._standings_win = win

        # 헤더
        hdr = tk.Frame(win, bg=PANEL, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📊  MLB 리그 순위", bg=PANEL, fg=TEXT,
                 font=("Arial", 15, "bold")).pack(side="left", padx=16)

        # 시즌 레이블
        self._st_season_lbl = tk.Label(hdr, text="", bg=PANEL, fg=SUBTEXT,
                                        font=("Arial", 12))
        self._st_season_lbl.pack(side="left", padx=8)

        # 새로고침 버튼
        self._mkbtn(hdr, "↺ 새로고침",
                    lambda: self._fetch_standings(win), CARD2, fg=MINT, w=10).pack(
            side="right", padx=12)

        # 리그 탭 바
        tab_bar = tk.Frame(win, bg=PANEL, height=38)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)
        tk.Frame(tab_bar, bg=PANEL, width=8).pack(side="left")

        self._st_league = tk.StringVar(value="NL")
        self._st_tab_btns = {}
        for lg in ["NL", "AL"]:
            b = tk.Button(
                tab_bar, text=lg,
                bg=PANEL, fg=SUBTEXT if lg != "NL" else MINT,
                font=("Arial", 13, "bold") if lg == "NL" else ("Arial", 13),
                relief="flat", bd=0, padx=20, pady=8,
                cursor="hand2",
                activebackground=PANEL, activeforeground=MINT,
                command=lambda l=lg: self._switch_standings_league(l),
            )
            b.pack(side="left")
            self._st_tab_btns[lg] = b

        tk.Frame(win, bg=BORDER, height=1).pack(fill="x")

        # 본문 (스크롤 가능)
        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self._st_canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self._st_canvas.yview)
        self._st_canvas.configure(yscrollcommand=vsb.set)
        self._st_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._st_inner = {}
        for lg in ["NL", "AL"]:
            f = tk.Frame(self._st_canvas, bg=BG)
            self._st_inner[lg] = f

        self._st_win_id = self._st_canvas.create_window(
            (0, 0), window=self._st_inner["NL"], anchor="nw"
        )
        self._st_inner["NL"].bind("<Configure>",
            lambda e: self._st_canvas.configure(
                scrollregion=self._st_canvas.bbox("all")))
        self._st_canvas.bind("<Configure>",
            lambda e: self._st_canvas.itemconfig(self._st_win_id, width=e.width))

        # 스크롤 바인딩
        win.bind_all("<MouseWheel>", self._on_standings_scroll)
        win.bind_all("<Button-4>",   self._on_standings_scroll)
        win.bind_all("<Button-5>",   self._on_standings_scroll)
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_standings(win))

        # 로딩 시작
        self._fetch_standings(win)

    def _on_standings_scroll(self, event):
        if not hasattr(self, "_standings_win") or not self._standings_win or \
                not self._standings_win.winfo_exists():
            return
        cx = self._st_canvas.winfo_rootx()
        cy = self._st_canvas.winfo_rooty()
        cw = self._st_canvas.winfo_width()
        ch = self._st_canvas.winfo_height()
        px = self._standings_win.winfo_pointerx()
        py = self._standings_win.winfo_pointery()
        if not (cx <= px <= cx + cw and cy <= py <= cy + ch):
            return
        if event.num == 4:
            self._st_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._st_canvas.yview_scroll(1, "units")
        else:
            direction = -1 if event.delta > 0 else 1
            self._st_canvas.yview_scroll(direction, "units")
        return "break"

    def _close_standings(self, win):
        # 전역 바인딩 해제 후 닫기
        try:
            win.unbind_all("<MouseWheel>")
            win.unbind_all("<Button-4>")
            win.unbind_all("<Button-5>")
        except Exception:
            pass
        # 원래 바인딩 복원
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>",   self._on_mousewheel)
        self.bind_all("<Button-5>",   self._on_mousewheel)
        win.destroy()

    def _fetch_standings(self, win):
        """백그라운드에서 순위 데이터 로드"""
        for lg in ["NL", "AL"]:
            for w in self._st_inner[lg].winfo_children():
                w.destroy()
            tk.Label(self._st_inner[lg], text="로딩 중...",
                     bg=BG, fg=SUBTEXT, font=("Arial", 13)).pack(pady=30)

        year = self._kst_today().year

        def worker():
            try:
                data = get_standings(year)
                self.after(0, lambda: self._populate_standings(data, year))
            except Exception as e:
                self.after(0, lambda: self._standings_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _standings_error(self, msg):
        for lg in ["NL", "AL"]:
            for w in self._st_inner[lg].winfo_children():
                w.destroy()
            tk.Label(self._st_inner[lg], text=f"오류: {msg}",
                     bg=BG, fg=RED, font=("Arial", 12)).pack(pady=20)

    def _populate_standings(self, data, year):
        """순위 데이터를 팝업 창에 렌더링"""
        if not hasattr(self, "_standings_win") or not self._standings_win or \
                not self._standings_win.winfo_exists():
            return

        self._st_season_lbl.config(text=f"{year} 시즌")

        for lg in ["NL", "AL"]:
            for w in self._st_inner[lg].winfo_children():
                w.destroy()

            parent = self._st_inner[lg]
            lg_data = data.get(lg, {})

            for div_name in ["동부", "중부", "서부"]:
                teams = lg_data.get(div_name, [])
                self._st_division_block(parent, f"{lg} {div_name}", teams)

        # 현재 리그 다시 표시
        self._switch_standings_league(self._st_league.get())

    def _st_division_block(self, parent, title: str, teams: list):
        """디비전 블록 렌더링 (grid 레이아웃)"""
        block = tk.Frame(parent, bg=BG)
        block.pack(fill="x", padx=16, pady=(12, 0))

        tk.Label(block, text=title, bg=BG, fg=TEXT,
                 font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 6))

        card = tk.Frame(block, bg=CARD)
        card.pack(fill="x")
        card.columnconfigure(0, weight=1)   # 팀명 열만 늘어남

        # 컬럼 정의: (헤더, grid-column, anchor, minwidth)
        COL_DEFS = [
            # col  헤더      anchor    minw
            (0,  "팀",      "w",      180),
            (1,  "승",      "center",  36),
            (2,  "패",      "center",  36),
            (3,  "승률",    "center",  52),
            (4,  "게임차",  "center",  56),
            (5,  "홈",      "center",  62),
            (6,  "원정",    "center",  62),
            (7,  "최근10",  "center",  62),
            (8,  "연속",    "center",  50),
        ]

        # 헤더 행
        hdr = tk.Frame(card, bg=CARD2)
        hdr.pack(fill="x")
        hdr.columnconfigure(0, weight=1)
        for col_i, col_name, anchor, minw in COL_DEFS:
            tk.Label(hdr, text=col_name, bg=CARD2, fg=SUBTEXT,
                     font=("Arial", 11, "bold"),
                     anchor=anchor, padx=6, pady=5,
                     width=max(minw // 9, len(col_name) + 1),
                     ).grid(row=0, column=col_i, sticky="ew")

        # 팀 행
        for i, t in enumerate(teams):
            row_bg = CARD if i % 2 == 0 else CARD2
            row = tk.Frame(card, bg=row_bg)
            row.pack(fill="x")
            row.columnconfigure(0, weight=1)

            # 팀명 셀: 로고 canvas + 한글명 label
            name_cell = tk.Frame(row, bg=row_bg)
            name_cell.grid(row=0, column=0, sticky="ew", padx=6, pady=4)

            abbr  = _team_abbr(t["name"])
            photo, _ = self._load_team_logo(t["name"], 20)
            logo_c = tk.Canvas(name_cell, width=20, height=20, bg=row_bg,
                                highlightthickness=0)
            logo_c.pack(side="left", padx=(0, 6))
            if photo:
                logo_c.create_image(10, 10, image=photo)
            else:
                logo_c.create_text(10, 10, text=abbr[:3], fill=MINT,
                                   font=("Arial", 9, "bold"))

            kr_name = team_kr(t["name"])
            tk.Label(name_cell, text=kr_name, bg=row_bg, fg=TEXT,
                     font=("Arial", 12), anchor="w").pack(side="left")

            # 나머지 통계 열
            try:
                pct_f = float(t["pct"])
            except (ValueError, TypeError):
                pct_f = 0.0

            streak_str = str(t.get("streak", "-"))
            streak_col = (MINT if "W" in streak_str else
                          RED  if "L" in streak_str else SUBTEXT)

            stat_vals = [
                (1, str(t["wins"]),   TEXT,       "center"),
                (2, str(t["losses"]), TEXT,       "center"),
                (3, str(t["pct"]),    MINT if pct_f >= 0.5 else TEXT, "center"),
                (4, str(t["gb"]),     SUBTEXT,    "center"),
                (5, t["home"],        SUBTEXT,    "center"),
                (6, t["away"],        SUBTEXT,    "center"),
                (7, t["last10"],      TEXT,       "center"),
                (8, streak_str,       streak_col, "center"),
            ]
            for col_i, val, fg, anchor in stat_vals:
                _, _, _, minw = next(c for c in COL_DEFS if c[0] == col_i)
                tk.Label(row, text=val, bg=row_bg, fg=fg,
                         font=("Arial", 12), anchor=anchor, padx=4,
                         width=max(minw // 9, len(val) + 1),
                         ).grid(row=0, column=col_i, sticky="ew", pady=5)

        tk.Frame(block, bg=BG, height=8).pack()

    def _switch_standings_league(self, league: str):
        """AL/NL 탭 전환"""
        self._st_league.set(league)
        for lg, btn in self._st_tab_btns.items():
            if lg == league:
                btn.config(fg=MINT, font=("Arial", 13, "bold"))
            else:
                btn.config(fg=SUBTEXT, font=("Arial", 13))

        # 캔버스 window 교체
        self._st_canvas.itemconfig(self._st_win_id, window=self._st_inner[league])



        # 스크롤 리전 재계산
        self._st_inner[league].update_idletasks()
        self._st_canvas.configure(scrollregion=self._st_canvas.bbox("all"))
        self._st_canvas.yview_moveto(0)

        # Configure 바인딩 재연결
        self._st_inner[league].bind("<Configure>",
            lambda e: self._st_canvas.configure(
                scrollregion=self._st_canvas.bbox("all")))

    # 주자 다이아몬드

    def _draw_diamond(self, r1, r2, r3):
        c = self.diamond_canvas
        c.delete("all")
        bases = {"home": (35, 50), "1b": (58, 32), "2b": (35, 10), "3b": (12, 32)}
        order = ["home", "1b", "2b", "3b", "home"]
        for i in range(len(order) - 1):
            x1, y1 = bases[order[i]]
            x2, y2 = bases[order[i + 1]]
            c.create_line(x1, y1, x2, y2, fill=SUBTEXT, width=1)
        sz = 6
        for key, filled in [("home", 0), ("1b", r1), ("2b", r2), ("3b", r3)]:
            x, y   = bases[key]
            color  = MINT if filled else BORDER
            c.create_rectangle(x-sz, y-sz, x+sz, y+sz,
                                fill=color, outline=TEXT, width=1)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MLBApp()
    app.mainloop()
