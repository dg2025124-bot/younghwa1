import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

LOW_PRED_FLOOR = 1  # 로그 스케일 그래프 바닥에 붙일 값
LOW_PRED_THRESHOLD = 1000  # 이 값보다 예측 관객 수가 작으면 바닥에 표시

st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 일별 박스오피스 데이터와 영화 정보를 결합해 총 관객 수를 예측하는 다중 회귀 모델입니다.")


# ------------------------------------------------------------------
# 데이터 로딩
# ------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_raw_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


@st.cache_data(show_spinner="영화별 변수를 계산하는 중...")
def build_features(daily: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    daily_sorted = daily.sort_values(["영화코드", "날짜"])
    grouped = daily_sorted.groupby("영화코드")

    daily_feat = grouped.agg(
        daily_days_count=("날짜", "count"),
        max_screens=("스크린수", "max"),
        max_shows=("상영횟수", "max"),
        max_daily_audience=("일관객", "max"),
        avg_daily_audience=("일관객", "mean"),
        best_rank=("순위", "min"),
    ).reset_index()

    first_day_audience = (
        grouped.first()["일관객"].reset_index().rename(columns={"일관객": "first_day_audience"})
    )
    daily_feat = daily_feat.merge(first_day_audience, on="영화코드", how="left")
    daily_feat = daily_feat.rename(columns={"영화코드": "movieCd"})

    # 영화별 표(movies)에 있는 영화를 전부 유지하기 위해 movies를 기준으로 left join
    merged = movies.sort_values("movieCd").merge(daily_feat, on="movieCd", how="left")
    merged = merged.reset_index(drop=True)

    daily_derived_cols = [
        "daily_days_count", "max_screens", "max_shows",
        "max_daily_audience", "avg_daily_audience", "best_rank", "first_day_audience",
    ]
    movies_numeric_cols = ["first_scrn", "first_show", "first_week_audi", "days_in_top10", "peak", "total_audi"]
    for col in daily_derived_cols + movies_numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    return merged


daily_df, movies_df = load_raw_data()
merged_df = build_features(daily_df, movies_df)

# 기준 기간 (박스오피스 일별 데이터 기준)
date_series = pd.to_datetime(daily_df["날짜"], format="%Y%m%d")
period_start, period_end = date_series.min().strftime("%Y-%m-%d"), date_series.max().strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# 영화코드 순 정렬 후 10편 단위로 앞 3편 테스트 / 나머지 학습
# ------------------------------------------------------------------
merged_df = merged_df.sort_values("movieCd").reset_index(drop=True)
merged_df["is_test"] = (merged_df.index % 10) < 3
train_df = merged_df[~merged_df["is_test"]].copy()
test_df = merged_df[merged_df["is_test"]].copy()


# ------------------------------------------------------------------
# 변수 선택 (체크박스)
# ------------------------------------------------------------------
FEATURE_OPTIONS = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_week_audi": "첫 주 관객수",
    "days_in_top10": "10위권 체류일수",
    "peak": "성수기(1·7·12월) 개봉 여부",
    "daily_days_count": "박스오피스 등장 일수",
    "max_screens": "박스오피스 최고 스크린수",
    "max_shows": "박스오피스 최고 상영횟수",
    "first_day_audience": "박스오피스 첫날 일일 관객수",
    "max_daily_audience": "박스오피스 최고 일일 관객수",
    "avg_daily_audience": "박스오피스 평균 일일 관객수",
    "best_rank": "박스오피스 최고 순위(낮을수록 상위)",
}
DEFAULT_ON = {"first_scrn", "first_week_audi", "days_in_top10", "peak"}

st.sidebar.header("⚙️ 예측에 사용할 변수")
st.sidebar.caption("체크한 변수들로 다중 회귀 모델을 학습합니다.")
selected_features = []
for col, label in FEATURE_OPTIONS.items():
    checked = st.sidebar.checkbox(label, value=(col in DEFAULT_ON), key=f"feat_{col}")
    if checked:
        selected_features.append(col)

if not selected_features:
    st.warning("최소 한 개 이상의 변수를 체크해 주세요.")
    st.stop()


# ------------------------------------------------------------------
# 모델 학습 및 평가
# ------------------------------------------------------------------
X_train = train_df[selected_features].values
y_train = train_df["total_audi"].values
X_test = test_df[selected_features].values
y_test = test_df["total_audi"].values

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None)  # 음수 예측 방지

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

nonzero_mask = y_test > 0
mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100

below_threshold = y_pred < LOW_PRED_THRESHOLD
below_threshold_count = int(below_threshold.sum())


# ------------------------------------------------------------------
# 화면 표시: 요약 정보
# ------------------------------------------------------------------
st.subheader("📋 모델 학습 개요")
c1, c2, c3 = st.columns(3)
c1.metric("학습에 쓴 영화 편수", f"{len(train_df):,}편")
c2.metric("예측 점수를 측정한 영화 편수", f"{len(test_df):,}편")
c3.metric("기준 기간(박스오피스 데이터)", f"{period_start} ~ {period_end}")

st.subheader("📈 예측 성능 (테스트 영화 기준)")
m1, m2, m3 = st.columns(3)
m1.metric("결정계수 R²", f"{r2:.3f}")
m2.metric("평균 절대 오차 (MAE)", f"{mae:,.0f}명")
m3.metric("평균 절대 비율 오차 (MAPE)", f"{mape:.1f}%")

st.caption(
    f"예측된 총 관객 수가 {LOW_PRED_THRESHOLD:,}명보다 작게 나온 영화는 "
    f"**{below_threshold_count}편**이며, 그래프 바닥(로그 스케일상 {LOW_PRED_FLOOR}명 지점)에 표시됩니다."
)


# ------------------------------------------------------------------
# 산점도 (Plotly)
# ------------------------------------------------------------------
plot_y = np.where(below_threshold, LOW_PRED_FLOOR, y_pred)
plot_x = np.clip(y_test, a_min=1, a_max=None)  # 로그축 오류 방지

names = test_df["movieNm"].values

fig = go.Figure()

normal_mask = ~below_threshold
fig.add_trace(
    go.Scatter(
        x=plot_x[normal_mask],
        y=plot_y[normal_mask],
        mode="markers",
        name=f"예측 ≥ {LOW_PRED_THRESHOLD:,}명",
        text=names[normal_mask],
        marker=dict(size=9, color="#3366CC", opacity=0.75),
        hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>",
    )
)

if below_threshold_count > 0:
    fig.add_trace(
        go.Scatter(
            x=plot_x[below_threshold],
            y=plot_y[below_threshold],
            mode="markers",
            name=f"예측 < {LOW_PRED_THRESHOLD:,}명 (바닥 표시, {below_threshold_count}편)",
            text=names[below_threshold],
            marker=dict(size=10, color="#DC3912", symbol="triangle-down"),
            hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>예측: 1,000명 미만 (바닥에 표시)<extra></extra>",
        )
    )

axis_min = max(1, min(plot_x.min(), plot_y.min()))
axis_max = max(plot_x.max(), plot_y.max())
ref_line = np.array([axis_min, axis_max])
fig.add_trace(
    go.Scatter(
        x=ref_line,
        y=ref_line,
        mode="lines",
        name="예측 = 실제 (기준선)",
        line=dict(color="gray", dash="dash"),
    )
)

fig.update_xaxes(type="log", title="실제 총 관객 수 (명, 로그 스케일)")
fig.update_yaxes(type="log", title="예측 총 관객 수 (명, 로그 스케일)")
fig.update_layout(
    title="테스트 영화: 실제 총 관객 수 vs 예측 총 관객 수",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("사용한 변수 목록 보기"):
    for col in selected_features:
        st.write(f"- {FEATURE_OPTIONS[col]} ({col})")
