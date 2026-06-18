"""로지스틱 회귀 기반 승리 확률 예측 모델"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from .features import FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "win_prob_model.pkl")


class WinProbabilityModel:
    """
    이진 분류(Binary Classification): 홈팀 승리 확률 예측
    - 입력: 이닝/아웃/점수차/주자 상태 등 경기 상황
    - 출력: P(홈팀 승리) ∈ [0, 1]
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=0.5,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self.is_trained = False

    def train(self, df):
        """DataFrame(FEATURE_COLS + home_wins 포함)으로 모델 학습"""
        X = df[FEATURE_COLS].values.astype(float)
        y = df["home_wins"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  정확도(Accuracy): {acc:.3f}")
        print(f"  브라이어 점수(Brier Score): {brier:.3f}  (낮을수록 좋음)")
        print(f"  AUC-ROC: {auc:.3f}")
        return {"accuracy": acc, "brier": brier, "auc": auc}

    def predict(self, state_dict):
        """단일 상태 dict → 홈팀 승리 확률(float)"""
        print(f"[model] predict() 호출: is_trained={self.is_trained}")
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다. train() 또는 load()를 먼저 실행하세요.")
        X = np.array([[state_dict.get(col, 0) for col in FEATURE_COLS]], dtype=float)
        print(f"[model] predict() X={X.tolist()}, NaN 포함={np.isnan(X).any()}")
        result = float(self.pipeline.predict_proba(X)[0, 1])
        print(f"[model] predict() 결과 P(home_win)={result}")
        return result

    def predict_batch(self, states):
        """상태 dict 리스트 → 확률 리스트"""
        print(f"[model] predict_batch() 호출: len(states)={len(states) if states else 0}, "
              f"is_trained={self.is_trained}")
        if not states:
            print("[model] predict_batch(): states 비어있음 -> [] 반환")
            return []
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        print("[model] predict_batch() 호출됨")
        X = np.array([[s.get(col, 0) for col in FEATURE_COLS] for s in states], dtype=float)
        print(f"[model] predict_batch() X.shape={X.shape}, NaN 포함={np.isnan(X).any()}, "
              f"마지막 행={X[-1].tolist()}")
        probs = self.pipeline.predict_proba(X)[:, 1].tolist()
        print(f"[model] predict_batch() 결과 len(probs)={len(probs)}, "
              f"마지막 prob={probs[-1] if probs else None}")
        return probs

    def save(self, path=None):
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"  모델 저장: {path}")

    def load(self, path=None):
        path = path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일 없음: {path}")
        self.pipeline = joblib.load(path)
        self.is_trained = True
        print(f"  모델 로드: {path}")
