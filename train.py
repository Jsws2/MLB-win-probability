"""
모델 학습 스크립트
사용법: python train.py [--games 200] [--start 2023-04-01] [--end 2023-09-30]
"""

import argparse
import time
import sys
from src.collector import get_historical_game_pks, get_game_feed
from src.features import build_training_df
from src.model import WinProbabilityModel


def main():
    parser = argparse.ArgumentParser(description="MLB 승리 확률 모델 학습")
    parser.add_argument("--games", type=int, default=200, help="학습에 사용할 경기 수 (기본: 200)")
    parser.add_argument("--start", default="2023-04-01", help="시작 날짜 YYYY-MM-DD")
    parser.add_argument("--end", default="2023-09-30", help="종료 날짜 YYYY-MM-DD")
    args = parser.parse_args()

    print("=" * 60)
    print("  MLB 실시간 승리 확률 예측 시스템 - 모델 학습")
    print("=" * 60)
    print(f"\n[1/4] 경기 목록 수집 ({args.start} ~ {args.end}, 최대 {args.games}경기)")

    game_pks = get_historical_game_pks(args.start, args.end, max_games=args.games)
    print(f"  → 수집된 경기: {len(game_pks)}개")

    if not game_pks:
        print("수집된 경기가 없습니다. 날짜 범위를 확인하세요.")
        sys.exit(1)

    print(f"\n[2/4] 경기 데이터 다운로드 (play-by-play)")
    feeds = []
    for i, pk in enumerate(game_pks, 1):
        try:
            feed = get_game_feed(pk)
            feeds.append(feed)
            if i % 10 == 0 or i == len(game_pks):
                print(f"  {i}/{len(game_pks)} 완료", end="\r")
            time.sleep(0.1)
        except Exception as e:
            print(f"  경기 {pk} 오류: {e}")
    print(f"\n  → 다운로드 성공: {len(feeds)}개")

    print(f"\n[3/4] 특징(feature) 추출 및 전처리")
    df = build_training_df(feeds)
    if df.empty:
        print("유효한 데이터가 없습니다.")
        sys.exit(1)

    print(f"  → 총 타석(샘플) 수: {len(df):,}개")
    print(f"  → 홈팀 승리 비율: {df['home_wins'].mean():.1%}")
    print(f"  → 사용 특징: {list(df.columns[:7])}")

    print(f"\n[4/4] 로지스틱 회귀 모델 학습")
    model = WinProbabilityModel()
    metrics = model.train(df)
    model.save()

    print("\n" + "=" * 60)
    print("  학습 완료!")
    print(f"  정확도: {metrics['accuracy']:.1%}  |  AUC: {metrics['auc']:.3f}")
    print("  이제 main.py를 실행하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
