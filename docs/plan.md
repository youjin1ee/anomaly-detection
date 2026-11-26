# 메트릭 동적 베이스라인 사이드카 실험 계획

- 상태: v0.2, 1차 마일스톤(뼈대, 설정과 읽기, 모델 2종, 출력, 평가 하네스, 패키징) 구현 착수
- 작성일: 2026-11-25
- 성격: 범용 이상 탐지 사이드카 계획. Prometheus HTTP API 호환 저장소(Prometheus, VictoriaMetrics, Thanos, Mimir 등)가 있는 곳이면 사이드카나 단독 컨테이너로 붙이는 오픈소스 후보 프로젝트
- 표기: 수치 중 제안값은 제안값이라고 표시. 확인하지 못한 사실은 확인하지 못했다고 표시

## 배경

- 관제와 알림 규칙은 대개 고정 임계값 중심. 임계값 근거가 약하고, 급증이나 증가라는 이름이어도 실제 판정은 절대값 비교인 경우가 많음
- 평소 대비로 보는 규칙은 드물고, 떨어지는 쪽을 보는 규칙은 더 드묾. 요청량 급감, 출력 토큰 길이 급감, 학습 처리량 하락 같은 조용한 하락은 고정 임계로 놓치기 쉬움. 이 판단은 일반적인 규칙 구성에 대한 가정이며 평가 하네스에서 실측
- 대상 환경을 특정하지 않음. Prometheus HTTP API 호환 저장소에 PromQL로 읽을 수 있는 메트릭이 있으면 적용 가능

### 침묵의 장애 시나리오 (평가에 사용)

- 모델 롤아웃 후 chat template 불일치로 응답이 첫 토큰에서 EOS로 끝남
  - HTTP 200, 지연은 오히려 개선. 신호는 출력 토큰 길이 급감 하나
- 앞단 LB나 DNS 변경으로 요청이 서빙에 도달하지 않음
  - 신호는 요청량이 0 근처로 하락
- InfiniBand 링크가 400G 대신 200G로 협상되어 분산 학습 전체가 느려짐
  - GPU 사용률 100%, 에러 0. 신호는 스텝 시간 증가와 처리량(tokens/sec) 하락

### 구성안과 선택

- 세 가지 구성안: 별도 사이드카 방식, 백엔드 감지 루프 안 기준선 계산 방식, recording rule 밴드 방식
- 선택: 별도 사이드카 방식. 파이썬 사이드카가 PromQL로 메트릭을 읽고 계절성 기준선, RCF, 예측 모델로 이상 점수와 상하한 밴드를 계산해 `/metrics`로 노출
  - 사이드카 컨테이너로도, 단독 컨테이너로도 배치 가능. 저장소 쪽 변경은 `/metrics` 스크레이프 설정 추가뿐
- 나머지 두 방식은 비교 대상이자 넘어야 할 기준선으로만 사용

## 목표와 성공 기준

목표
- 사이드카 방식이 현재 고정 임계와 recording rule 방식보다 침묵의 장애 3개 시나리오를 더 빨리, 더 적은 오탐으로 잡는지 수치로 확인
- 정확도를 단계별로 올리는 과정(기준선 측정, RCF, 예측 밴드, 조합과 튜닝)과 각 단계의 근거를 기록

성공 기준 (모두 제안값. 1단계 기준선 측정 뒤 사용자와 조정)

| 항목 | 기준 | 측정 방법 |
|---|---|---|
| 시나리오 탐지 | 침묵의 장애 3개 시나리오 모두, 시나리오당 변형 20회 이상 주입 중 탐지율 90% 이상 | 합성 주입 |
| 탐지 지연 상한 | 요청량 급락 3분, 출력 토큰 길이 급감 5분, 처리량 하락 10분 이내 | 이상 시작 시각부터 첫 경보까지 |
| 오탐 상한 | 정상 구간 재생에서 시계열당 하루 0.1건 이하. 평가 시계열 20개 기준 합산 하루 2건 이하 | 이상이 없는 14일 합성과 NAB 정상 구간 |
| 기준선 대비 개선 | 기준선 두 방식 중 더 좋은 쪽 대비 이벤트 단위 F1 0.10 이상 개선, 또는 같은 재현율에서 오탐 50% 이상 감소 | 같은 하네스 |
| 고정 임계 대비 | 고정 임계로 못 잡는 시나리오(위 3개 중 추정 3개)에서 탐지율 0%에서 90% 이상 | 1단계에서 실측으로 확인 |
| 워밍업 | 학습 데이터 7일 미만에서도 값 없음을 정확히 내고, 최소 표본 충족 뒤에만 점수 출력 | 단위 시험 |
| 재시작 복구 | 재시작 뒤 5분 이내 정상 출력 재개 | 재시작 시험 |
| 부하 | 시계열 500개, 평가 주기 60초 기준 한 주기 30초 이내, 메모리 1GiB 이하 | 6단계 측정 |

## 범위와 비범위

범위
- 레포 `anomaly-detection` 안에서 끝나는 조사, 평가 하네스, 합성 데이터, 사이드카 구현, 실험 결과 기록
- 마지막 단계에서 임의의 Prometheus 호환 환경에 붙이는 통합 가이드 작성

비범위
- 특정 제품 전용 연동 코드와 제품 본체 수정
- helm 차트(1차 마일스톤 범위 밖, 이후 단계에서 검토)
- 상용 재배포 불가 라이선스(GPL, AGPL, ELv2, vmanomaly 같은 엔터프라이즈 전용) 코드 사용
- 로그 기반 이상 탐지(VictoriaLogs). 메트릭만 다룸
- 다변량 이상 탐지, 원인 분석(RCA)

## 동작 원리

### Random Cut Forest (RCF)

출처는 Guha et al. 2016 논문(참고 자료 1)과 AWS 공개 구현(참고 자료 4). 논문 본문에서 확인한 내용

- 트리 구성: 점 집합에서 각 차원 범위 길이에 비례하게 차원을 고름. 그 범위 안에서 균등 분포로 값을 골라 둘로 가름. 각 부분집합에 재귀 적용
  - Isolation Forest는 차원을 균등하게 고름. RCF는 범위에 비례. 불필요한 차원의 영향을 줄이는 차이
- 스트리밍 갱신: 트리에서 점을 삭제하거나 삽입해도 그 점 집합에서 새로 만든 것과 같은 확률의 트리가 됨. 표본 갱신에 부분선형 시간, 공간은 O(d|S|)
- 표본 추출: 저수지 표집(reservoir sampling)으로 균등 표본, 또는 최근성 가중 표본을 유지. 표집과 트리 구성이 서로 독립
- 이상 점수 CoDisp(collusive displacement): 점을 트리에서 빼면 나머지 점들이 얼마나 이동하는지(트리 높이 변화)를, 그 점을 포함하는 부분집합 중 최대값으로 정의. 점이 표본에 없어도 삽입했을 때의 값을 계산할 수 있음
- shingle: 시계열의 연속된 값 w개를 한 점으로 묶는 기법. 논문 예시는 크기 4이며 한 칸씩 밀면서 w차원 점을 만듦. 모양이 평소와 다른 구간이 이상 후보
  - 논문 논의: shingle이 너무 작으면 자연 변동에 오탐, 너무 크면 탐지가 늦거나 놓침. 표본 크기는 256, 512, 1K에서 결론이 같았음(해당 데이터셋 한정)
- 논문의 한계 취지: 점수는 상대적인 값이라 임계 결정이 별도 필요. 이 점이 아래 ThresholdedRCF 같은 후처리가 있는 이유로 추정되며 논문에서 직접 확인한 것은 아님

실험에 쓸 때의 쟁점(가설, 실험으로 확인)
- shingle은 짧은 구간의 모양만 봄. 일 단위, 주 단위 계절성은 shingle 크기 8만으로는 담기 어려울 가능성. 시각 특성(시각, 요일) 추가 입력이나 차분, 정규화로 보완할 수 있는지 확인
- 수평 이동(레벨 시프트)이 지속되면 새 수준을 정상으로 학습. 침묵의 장애는 지속되는 하락이라 학습 속도와 표본 크기가 탐지 지연을 좌우

### AWS 공개 구현

- 레포 `aws/random-cut-forest-by-aws`: Apache-2.0, 핵심 구현은 Java(릴리스 4.4.0-java, 2025-09-15). 레포 안에 Rust 구현(crate 이름 `rcf`, 버전 4.0.0)과 `python_rcf_wrapper`가 있음. 마지막 커밋은 2025-10-21 확인
- 계층: `randomcutforest-core`는 점수와 외삽 같은 원시 추정. `randomcutforest-parkservices`는 `ThresholdedRandomCutForest`(이상 여부 판정)와 `RCFCaster`(보정된 예측)로 원시 점수를 판정으로 가공 (레포 README)
- `ThresholdedRandomCutForest` 소스에서 확인한 것
  - `process(점, 타임스탬프)`가 `AnomalyDescriptor`를 돌려줌. 결측 위치 지정 오버로드가 있음
  - `setIgnoreNearExpectedFromAbove`, `FromBelow`와 각각 `ByRatio` 변형이 있음. AMP의 ignoreNearExpected 옵션과 이름이 대응
  - `AnomalyDescriptor`에 기대값(expectedValues) 배열이 있음. 기대값 개수는 최대 1
  - 생성 시 `shingleSize`, `sampleSize`, `numberOfTrees`, `outputAfter` 지정
- 확인 못 한 것: 상하한 밴드를 직접 내는 API가 있는지. 기대값과 점수는 확인했지만 밴드 형태의 출력은 소스에서 찾지 못함. `RCFCaster`가 예측 범위를 내는지는 이 조사에서 확인하지 못함
- 파이썬 사용: 공식 PyPI 패키지는 없음(PyPI `randomcutforest` 조회 404). `python_rcf_wrapper`는 JPype로 Java를 호출하는 래퍼이며 `TRCF`를 노출. 즉 JVM이 필요. Rust 구현은 파이썬 바인딩을 확인하지 못함
- 확인 못 한 것: Java jar의 배포처(Maven Central 여부)와 래퍼 설치 절차의 실제 동작. 1단계에서 확인

### rrcf (kLabUM)

- MIT, 순수 파이썬. 최신 릴리스 0.4.4(PyPI 2023-04-30), 레포 마지막 push 2024-02-24. 사실상 유지보수 정체
- 용도: AWS 구현과의 대조군, 알고리즘 동작 이해용. 운영 후보로는 비추천(순수 파이썬 성능은 이번 실험에서 측정)

### OpenSearch Anomaly Detection (참고)

- RCF로 거의 실시간 이상 탐지. 점마다 anomaly grade와 confidence를 계산. 설정이 없으면 기대값에서 20% 미만 편차는 무시하는 기본 규칙이 있음(문서 확인 범위)
- shingle 기본값과 학습 방식은 문서의 앞부분만 확인했으며 이 조사에서는 확인하지 못함

### 예측 밴드

과거 패턴으로 지금 시각의 기대 범위를 예측하고, 실제값이 범위를 벗어나면 이상으로 봄. 점수는 이탈 정도를 정규화해 만듦.

- statsforecast 2.1.1 (Apache-2.0, 2026-07-16)
  - `MSTL(season_length=[...])`로 다중 계절성 분해. 소스 확인: `season_length`는 정수 또는 리스트, 추세 예측 모델 기본값 `AutoETS(model='ZZN')`, 예측구간은 `ConformalIntervals`를 넘기는 방식
  - AutoETS, AutoARIMA의 `season_length`는 정수 하나만 받음. 다중 계절성에는 MSTL, AutoMFLES, AutoTBATS(리스트 허용)를 사용
  - `forecast(level=[...])`로 신뢰수준별 구간을 받음
  - 5분 해상도 기준 일 계절 288, 주 계절 2016 점. 해상도는 실험 변수
- Prophet 1.5.0 (MIT, 2026-10-05)
  - 구간은 추세 불확실성, 계절성 불확실성, 관측 잡음으로 구성. 기본 `interval_width` 80%. 계절성 불확실성은 `mcmc_samples` 설정 시에만 포함되고 수 분이 걸릴 수 있음(공식 문서)
  - 구간은 과거와 같은 비율로 추세가 바뀐다고 가정하므로 적중률이 맞지 않을 수 있다고 문서가 명시
- Darts 0.48.1 (Apache-2.0, 2026-10-16)
  - `darts.ad`에 스코어러(Difference, Norm, NLL 계열, Wasserstein, PyOD, k-means 등), 탐지기(IQR, Quantile, Threshold), 집계기(AND, OR, Ensemble), `ForecastingAnomalyModel`이 있음(공식 문서 확인 범위)
  - 예측 모델 위에 스코어러와 탐지기를 얹는 평가 틀로 활용할 후보
- 기준선용 단순 계절 방식: 같은 시각, 같은 요일의 과거 값에 대해 평균과 표준편차(z-score), 또는 중앙값과 MAD(median absolute deviation)로 밴드 구성
  - 레퍼런스 `grafana/promql-anomaly-detection`(Apache-2.0, v0.3.0, 2026-10-15). README 확인 범위: 단기 밴드, 장기 밴드(계절성), 마진 밴드로 구성. adaptive 전략은 평균과 표준편차, robust 전략은 중앙값과 MAD. 0 값을 결측으로 처리하므로 0으로 급감하는 경우를 못 잡고 별도 알림을 권장함. 대부분 전략이 24~26시간 데이터를 요구. 정확한 수식과 평활 파라미터는 확인하지 못함

### AMP anomaly detection

확인된 것(공식 문서)
- 2025-10-30 발표(사용자 제공 정보). RCF 기반. 밴드, 점수, 값 4개 출력(upper_band, lower_band, score, value)
- `CreateAnomalyDetector` 요청: `alias`, `configuration`, `evaluationIntervalInSeconds`(기본 60, 범위 30~86400), `labels`, `missingDataAction`(markAsAnomaly 또는 skip 중 하나), `tags`
- `RandomCutForestConfiguration`: `query`(필수, 최대 8192자, 집계 연산자로 감싸야 함. 허용: avg, count, group, max, min, quantile, stddev, stdvar, sum), `shingleSize`(기본 8, 범위 2~1024), `sampleSize`(기본 256, 범위 256~1024), `ignoreNearExpectedFromAbove`, `ignoreNearExpectedFromBelow`
- `IgnoreNearExpected`는 절대량(`amount`) 또는 비율(`ratio`) 중 하나
- 학습 데이터는 14일 이상 권장. `PreviewAnomalyDetector`로 과거 데이터에 대해 미리 확인 가능
- 미리보기 응답: 원 시계열과 함께 `lower_band`, `score`, `upper_band` 시계열을 돌려줌. 표본이 충분할 때만 반환. score는 0~1이며 1이 이상 확신. 미리보기에서는 레이블 `anomaly_detector_preview`의 값으로 구분
- 사용자 안내 문서의 설명: 계절 추세에 적응하고 결측을 완만하게 처리하며 신뢰 점수를 제공
- 모범 사례: 안정적인 메트릭부터, 집계 데이터 사용, 민감도 조정

추정(문서에 근거 없음)
- 내부가 `ThresholdedRandomCutForest`와 같은 계열일 가능성. 이름이 같은 ignoreNearExpected 옵션이 근거이나 확인한 것은 아님
- 밴드는 RCF의 기대값에 오차 폭을 더하고 뺀 값일 가능성. 계산식은 공개되어 있지 않음
- 계절성 적응이 shingle로 되는지, 별도 전처리인지 알 수 없음

확인 못 한 것
- 운영 중인 탐지기의 출력 시계열 이름과 레이블(미리보기 형식만 확인)
- 요금. 공식 가격 페이지를 이번에 확인하지 않음

## 아키텍처

```text
Prometheus 호환 API (Prometheus, VictoriaMetrics, Thanos, Mimir 등)
        |  query_range / query (PromQL, 집계된 쿼리)
        v
  사이드카 (Python)
   - 쿼리 설정 로더 (YAML)
   - 시계열별 모델 풀 (계절성 기준선, RCF, 이후 예측 밴드)
   - 워밍업과 최소 표본 가드
   - 상태 저장소 (선택)
        |  /metrics (Prometheus exposition)
        v
  스크레이프 대상 (기존 수집기가 긁어감)
```

입력
- 설정 파일의 detector 항목마다 PromQL 하나, 평가 주기, 모델 종류, 파라미터를 가짐
- 쿼리는 집계 연산자로 감싸야 함. AMP 제약과 같은 규칙을 채택해 카디널리티를 쿼리 작성자가 통제
- 읽기는 Prometheus HTTP API 호환(`/api/v1/query_range`, `/api/v1/query`)만 사용
- 데이터소스 설정: URL, 경로 접두(VictoriaMetrics 클러스터 `/select/0/prometheus` 같은 테넌트 경로), 인증 헤더(Bearer, basic), 임의 헤더(Mimir `X-Scope-OrgID` 등), TLS 옵션

출력 (AMP 형태에 맞춤)
- 시계열 4개: `anomaly_value`, `anomaly_upper_band`, `anomaly_lower_band`, `anomaly_score`(0~1). 기대값 `anomaly_expected`를 추가. 레이블은 `detector`와 입력 쿼리의 결과 레이블을 유지
- 보조 시계열: `anomaly_detector_ready`(점수 출력 중이면 1, 워밍업이나 값 없음이면 0), 관리 중 시계열 수, 평가 지연, 쿼리 실패 수, 빈 결과 수, 카디널리티 제한으로 버린 시계열 수, 백필 상태
- 알림 규칙과의 대응: 걸림은 score가 임계 이상(`for`로 지속 조건 표현), 정상은 점수 출력 중 임계 미만, 값 없음은 점수 시계열 자체가 없음(워밍업과 입력 결측). 값 없음 알림은 `absent_over_time(anomaly_score[...])`로 구성

시계열당 모델과 카디널리티 제한
- 입력 쿼리가 돌려주는 시계열마다 모델 인스턴스 하나
- detector당 최대 시계열 수 상한(제안값 100)과 전체 상한(제안값 500)을 두고, 넘으면 초과분을 버리고 버린 수를 지표로 노출. 조용히 버리지 않음
- 시계열이 사라지면 유휴 시간 뒤 모델 폐기

워밍업
- 시작 시 `query_range`로 과거 데이터를 한 번에 읽어 모델에 먹임(backfill)
- 최소 표본에 못 미치면 score, 밴드를 내지 않음(값 없음). `anomaly_detector_ready`가 0으로 워밍업 표시
- AMP도 표본이 충분할 때만 시계열을 반환하는 형태와 일치

재시작 상태 복구
- 기본: 무상태. 재시작하면 TSDB에서 다시 backfill. TSDB가 이미 이력을 가지고 있어 별도 저장소가 필요 없음
- 선택: 주기적 스냅샷(파일 또는 PVC). backfill 시간이 너무 길 때만 사용. Java RCF에는 `serialization` 모듈이 있어 후보. 어느 쪽이 나은지는 6단계에서 측정해 결정

방향 설정
- detector별 `direction`: above, below, both. 침묵의 장애는 below가 핵심이라 기본값을 both로 두고 시나리오별로 지정
- 방향에 맞지 않는 이탈은 점수를 0으로 억제

저트래픽 보호
- 최소 표본: 학습 표본이 하한 미만이면 값 없음
- 절대 하한: 기대값이 설정한 하한보다 작으면 하락 방향 경보 억제. 단, 기대값이 충분히 컸던 상태에서 0 근처로 떨어지는 경우(요청량 시나리오)는 경보 대상. 이 두 조건의 구분이 설계 핵심
- 허용 폭: AMP의 ignoreNearExpected와 같은 절대량(amount)과 비율(ratio) 옵션

## 데이터

공개 데이터셋
- NAB(Numenta Anomaly Benchmark): 레포 MIT 라이선스(Copyright 2014-2024 Numenta). 라벨된 시계열 58개, 실제 데이터 다수(AWS CloudWatch, 광고, 트래픽, 트윗 등) 포함. 데이터 개별 라이선스는 README에서 별도 명시를 찾지 못함. 확인 내용은 `docs/results/m2-nab.md`
- NAB 점수 프로파일: Standard, Reward Low FP, Reward Low FN. RCF는 NAB 점수표에 AWS 독점 구현으로 등재(Standard 51.7, Reward Low FP 38.4, Reward Low FN 59.7. 점수표는 NAB v1.0 기준). 우리 구현의 결과와 비교할 때 구현이 다르다는 점을 같이 기록
- NAB는 정상, 이상 시계열이 섞인 범용 벤치마크라 침묵의 장애 시나리오를 대신하지는 못함. 알고리즘 일반 성능 확인용

합성 생성기
- 기본 신호: 일 계절성과 주 계절성, 완만한 추세, 잡음, 간헐 결측을 조합한 요청량, 평균 출력 토큰 길이, 처리량
- 주입 시나리오 (시나리오마다 시작 시각, 깊이, 지속, 감소 속도를 바꿔 변형 20회 이상)
  - 요청량 급락: 0 근처로 하락과 부분 하락(정상 대비 10~50% 수준)
  - 출력 토큰 길이 급감: 요청량과 지연은 정상 유지, 평균 길이만 1~3 토큰 수준으로 하락
  - 처리량 하락: 약 50% 계단 하락(400G에서 200G 협상 가정), GPU 사용률은 정상 유지
- 보조 이상: 단발 스파이크, 점진 드리프트, 분산 증가, 계절 위상 이동, 결측 구간. 오탐과 민감도 평가용
- 정상 구간: 이상 없는 14일 이상 생성. 같은 시드 재현을 위해 시드를 기록

실제 메트릭
- 실제 환경 메트릭으로 재생할 수 있는지는 열린 질문(결정 필요 절 참고)
- 서빙 엔진과 프록시 메트릭 이름은 환경마다 달라 이 문서에서 가정하지 않음. 출력 토큰 길이와 학습 처리량에 쓸 실제 메트릭 이름은 적용 환경에서 확인 필요

## 실험 단계

각 단계는 산출물과 완료 조건을 가짐. 완료는 실행한 검증 결과로만 판정

- [ ] 1단계 자료 정리
  - 산출물: `docs/research/` 아래 RCF, 예측 밴드, AMP, 라이선스, 평가 방법 정리. 이 계획의 확인 못 한 항목 해소 목록
  - 해소할 항목: Java RCF jar 배포처와 래퍼 설치 절차, Rust 구현의 파이썬 연동 가능성, ThresholdedRCF의 밴드 출력 가능성, NAB 데이터 라이선스, AMP 운영 출력 이름과 요금
  - 완료 조건: 모든 항목에 출처 URL 또는 직접 실행 결과가 붙고, 확인 불가 항목은 그렇게 표시
- [x] 2단계 평가 하네스와 합성 데이터 (1차 마일스톤에서 구현. NAB 로더와 NAB 점수는 2차 마일스톤에서 구현, `docs/results/m2-nab.md`)
  - 산출물: 합성 생성기(침묵의 장애 3개 시나리오 주입), 라벨 형식, 지표 계산기(이벤트 단위 precision, recall, F1, 탐지 지연, 하루당 오탐 수, NAB 점수), NAB 로더
  - 완료 조건: 같은 시드에서 같은 데이터와 라벨이 재현되고, 기준 탐지기 더미(항상 정상, 항상 이상)로 지표가 기대값과 일치
- [ ] 3단계 기준선 측정 (1차 마일스톤에서 고정 임계와 계절성 기준선은 측정. NAB는 2차 마일스톤에서 측정, `docs/results/m2-nab.md`)
  - 산출물: 고정 임계 방식(절대값 임계)과 recording rule 계열 방식(계절성 중앙값과 MAD) 결과표
  - recording rule 방식은 하네스에서 같은 수식으로 구현하고, 표본 몇 개는 실제 PromQL로도 계산해 일치를 확인
  - 완료 조건: 3개 시나리오와 NAB에 대한 기준선 지표 확정. 성공 기준 표의 제안값을 이 결과로 재조정해 사용자 확인
- [ ] 4단계 RCF (1차 마일스톤에서 rrcf 기반 기본 구현과 기본 파라미터 측정, 민감도 실험은 남음)
  - 산출물: rrcf 기반 탐지기(구현됨), 선택적 extra로 Java RCF 래퍼 대조 구현, shingle, sample, 트리 수, 입력 특성(시각, 차분, 정규화) 영향 실험 결과
  - 완료 조건: 기준선 대비 지표 비교표와 파라미터 민감도 그래프. 레벨 시프트 지속 시 탐지 지연 측정
- [ ] 5단계 예측 밴드
  - 산출물: statsforecast MSTL, Prophet, Darts 스코어러 기반 탐지기와 비교표. 계절 길이, 신뢰수준, 재학습 주기 영향
  - 완료 조건: 4단계와 같은 하네스에서 같은 지표로 비교. ARM64 설치와 추론 시간 기록
- [ ] 6단계 조합과 튜닝
  - 산출물: RCF와 예측 밴드 조합 방식(점수 결합, 밴드 교집합 등)과 튜닝(shingle, sample, 허용 폭, 방향, 절대 하한) 결과, 최종 설정 후보
  - 완료 조건: 성공 기준 표의 시나리오 탐지, 탐지 지연, 오탐 상한을 만족하거나, 만족하지 못한 항목과 이유를 기록
  - 튜닝 과적합 방지: 튜닝용 시드와 최종 평가용 시드를 분리
- [ ] 7단계 사이드카 패키징과 실환경 재생(replay) (1차 마일스톤에서 이미지, compose, 쿠버네티스 예시 작성. 재시작 복구와 부하 측정은 남음)
  - 산출물: 사이드카 컨테이너 이미지, `/metrics` 출력, 합성 데이터를 VictoriaMetrics와 Prometheus 양쪽에 적재해 사이드카가 둘 다 읽는지 확인한 결과, 워밍업, 재시작 복구, 카디널리티 상한, 부하 측정
  - 완료 조건: 성공 기준 표의 워밍업, 재시작 복구, 부하 항목 충족
- [ ] 8단계 통합 가이드
  - 산출물: `docs/integration-guide.md`. 임의의 Prometheus 호환 환경에 붙이는 방법. 사이드카와 단독 배치 선택 기준, 테넌트 경로와 헤더, 인증과 TLS, 스크레이프 설정과 알림 규칙 연결, 지속 조건(N분 중 M분)을 `for`와 `count_over_time`으로 표현하는 방법, 폐쇄망 배포(이미지 반입, JVM 미포함), 보존 기간 제약과 워밍업 시간
  - 완료 조건: 가이드 작성과 compose 예시 두 종(VictoriaMetrics, Prometheus)에서 따라 해 본 결과 기록

## 1차 마일스톤 결과 요약

상세는 `docs/results/m1-baseline.md`. 합성 데이터 기준이며 실환경 결과가 아님

- 구현: 설정 로더, Prometheus API 클라이언트(백필과 즉시 쿼리), 모델 3종(계절성 기준선, RCF, 정적 임계), 출력 메트릭과 상태 엔드포인트, 평가 하네스, Dockerfile, compose와 쿠버네티스 예시
- 계절성 기준선: 시나리오 4개 행 모두 탐지율 100%, 정상 구간 오탐 0건, 탐지 지연 중앙값 2.0에서 2.5분
- RCF(rrcf): 탐지율 85%에서 100%, 오탐 하루당 시계열 0.01에서 0.09건. 처리량 시나리오가 성공 기준 90%에 못 미침
- 고정 임계: 처리량 하락과 스텝 시간 증가를 한 건도 못 잡음(0%)
- 미측정: 재시작 복구 시간, 500시리즈 부하. NAB는 `docs/results/m2-nab.md`
- 남은 쟁점: 계절성 기준선이 합성 가정에 유리한 점, RCF 민감도 실험, 예측 밴드 모델, 실데이터 재생

## 실행 환경

- 무거운 작업(docker, 학습 실험, 부하 측정)은 원격 ARM64 장비에서 실행. 로컬은 편집과 조율만
- 금지: 컨테이너 런타임 환경 삭제, `docker prune`(system, volume, image 등 모든 종류), 기존 클러스터 삭제. 정리가 필요하면 이 실험이 만든 리소스에 이름을 붙여 그것만 삭제
- 원격 장비의 작업 디렉터리와 컨테이너, 클러스터 이름에 `dbl-` 접두사를 붙여 다른 작업과 구분
- 결과 데이터가 커지면 로컬에 보관하지 않고 별도 아카이브 위치에 둠. 위치는 결정 필요에서 확인
- ARM64 라이브러리 설치 가능 여부 (PyPI 파일 목록 조회. 실제 설치 성공은 1단계와 4~5단계에서 원격 ARM64 장비로 확인)
  - statsforecast 2.1.1: aarch64 wheel 있음. numba 0.68.1도 aarch64 wheel 있음
  - Prophet 1.5.0: aarch64 wheel 있음. cmdstan 동봉 여부는 확인하지 못함. cmdstanpy 1.3.2는 순수 파이썬
  - River 0.27.0: aarch64 wheel 있음. Darts 0.48.1: 순수 파이썬 wheel. PyOD 3.6.9: 순수 파이썬 wheel
  - Java RCF: JVM이 ARM64에서 동작해야 함. JPype 설치 확인 필요

## 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| 지속되는 하락을 모델이 새 정상으로 학습 | 침묵의 장애 탐지 지연이 길어짐 | 표본 크기와 최근성 조절, 장기 기준선 병용, 탐지 지연을 성공 기준에 포함 |
| JVM 의존(JPype 기반 RCF) | 이미지 크기, 폐쇄망 배포, 장애 지점 증가 | 기본은 순수 파이썬 rrcf, Java RCF는 선택적 extra로만 검토 |
| 계절성을 RCF 단독으로 못 담음 | 일간 주기 오탐 | 시각 특성 입력, 예측 밴드와 조합 |
| ARM64 설치 실패(Prophet cmdstan, numba) | 일정 지연 | 1단계에서 원격 ARM64 장비 설치 사전 확인, 실패 시 해당 라이브러리를 후보에서 제외하고 기록 |
| 합성 데이터에서만 잘 동작 | 실환경 일반화 불확실 | NAB 병행, 실제 메트릭 export를 열린 결정으로 올림, 튜닝과 최종 시드 분리 |
| 저트래픽 구간 오탐 | 신뢰 하락 | 최소 표본, 절대 하한, 허용 폭 옵션, 하락 방향 규칙 분리 |
| 카디널리티 폭증 | 메모리와 평가 시간 초과 | 집계 쿼리 강제, 상한, 버린 수 지표 노출 |
| 학습 이력 부족(저장소 보존 기간) | 워밍업이 길어짐, AMP 권장 14일 미달 | 보존 기간 확인, 7일 미만에서도 동작하는 모드 시험(성공 기준의 워밍업 항목) |
| 라이선스 | 상용 재배포 불가 | 후보는 Apache-2.0, MIT, BSD만. 레포에 NOTICE와 의존성 라이선스 목록 유지 |
| AMP 구현 추정이 틀림 | 같은 출력 형태여도 동작이 다름 | 추정과 확인을 구분해 기록하고, 출력 형태만 맞춤. 동작 일치를 목표로 두지 않음 |
| 평가 지표 해석 차이 | 결과 비교 불가 | 이벤트 단위 정의와 라벨 창 규칙을 2단계에서 문서로 고정 |

## 기술 결정

구현 수준 선택은 작성자가 정하고 근거만 기록. 아키텍처 변경은 사용자 확인 대상

| 결정 사항 | 선택 | 근거 |
|---|---|---|
| 사이드카 언어 | Python 3.12 | Darts 0.48.1, River 0.27.0이 3.11 이상 요구. 후보 라이브러리가 모두 파이썬 |
| RCF 기본 구현 | rrcf 0.4.4 (변경됨, v0.2) | MIT, 순수 파이썬, aarch64 설치와 동작을 직접 확인. JVM 의존은 이미지를 무겁게 하고 폐쇄망 배포 부담을 키움. 유지보수 정체는 알려진 약점이며 `setuptools<81` 고정이 필요(`pkg_resources` 의존). 세부 근거는 `docs/design.md` |
| RCF 선택적 extra | AWS Java RCF를 `python_rcf_wrapper`로 호출 | AMP와 같은 계열일 가능성, ThresholdedRCF의 기대값과 ignoreNearExpected 기능. 모델 레지스트리로 끼울 수 있게만 두고 구현은 이후 단계 |
| RCF 후속 후보 | Rust `rcf` 4.0.0 | JVM 제거 가능성. 파이썬 바인딩이 없어 필요 시 PyO3 래핑 비용을 평가 |
| 예측 밴드 주력 | statsforecast 2.1.1 MSTL | Apache-2.0, 다중 계절성 지원 소스 확인, ARM64 wheel 있음 |
| 예측 밴드 비교 | Prophet 1.5.0, Darts 0.48.1 | MIT, Apache-2.0. 구간 방식과 평가 틀이 달라 비교 가치 있음 |
| 스트리밍 비교 후보 | River 0.27.0 HalfSpaceTrees | BSD-3, 스트리밍 전용 |
| 보조 | PyOD 3.6.9 | BSD-2. 스코어러 비교용 |
| 제외 | vmanomaly, GPL, AGPL, ELv2 계열 | 상용 재배포 불가 또는 제약. vmanomaly는 엔터프라이즈 라이선스 필요(공식 문서) |
| 상태 저장 | 기본 무상태(재시작 시 backfill) | TSDB가 이력 보유. 스냅샷은 측정 후 필요 시 추가 |
| 모델 단위 | 시계열마다 인스턴스 하나 | AMP도 쿼리 결과 시계열 기준으로 점수를 냄(문서 확인 범위). 구현 단순 |
| 출력 형태 | AMP와 같은 4개 시계열 | 비교와 대체 용이. 이름은 `anomaly_score`, `anomaly_upper_band`, `anomaly_lower_band`, `anomaly_value`(변경됨, v0.2) |
| 시계열 데이터 포맷 | 하네스는 시드 기반 생성(파일 저장 없음), 설정은 YAML | 같은 시드면 같은 데이터가 재현되어 파일이 필요 없음. 실데이터 재생이 필요해지면 Parquet 검토 |
| 모델 인터페이스 | `update(timestamp, value)` 하나, 모델 레지스트리 | 계절성, RCF, 정적 임계가 같은 파이프라인(방향, 허용 폭, 하한, 워밍업)을 공유. 예측 밴드 모델은 `register_model`로 끼움 |
| 실행 모델 | 탐지기마다 스레드, 표준 라이브러리 HTTP 서버 | 의존성 최소화. 시계열 수백 개 규모에서 충분하다고 판단하며 부하 측정에서 재확인 |
| 평가 시계열 해상도 | 60초 | 성공 기준의 탐지 지연 3분을 측정하려면 1분 해상도가 필요 |
| 의존성 관리 | `uv`와 잠금 파일 | 재현성. 잠금 파일은 `uv.lock` |

## 결정 필요

사용자가 정할 항목
- GitHub 레포 이름과 공개 여부
  - 결정됨: 이름 `anomaly-detection`, 비공개. 오픈소스 공개 시 문서에 내부 세부가 없는지 공개 전 재검토
- 라이선스 파일
  - 미결정. LICENSE 파일은 만들지 않았고 의존성은 MIT, BSD, Apache-2.0만 사용
- 실제 환경 메트릭으로 재생 가능한지
  - 내용: 실제 운영 메트릭을 export해 재생할 수 있는지, 익명화가 필요한지
  - 영향: 실데이터 없이는 합성 데이터만으로 평가하게 되어 일반화 근거가 약함
- AMP를 실제 AWS에서 돌려 비교할지
  - 영향: 같은 합성 데이터로 AMP 탐지기를 돌리면 직접 비교가 가능하지만 AWS 계정, 비용, 데이터 반출이 필요. 요금은 이번 조사에서 확인하지 못함
  - 안 하면 공식 문서 기준의 출력 형태 일치만 확인
- 원격 ARM64 장비 사용 범위: 사용 가능한 자원과 기간, 결과 데이터를 보관할 아카이브 위치
- 성공 기준의 제안값 확정은 3단계 기준선 결과가 나온 뒤에 재논의

## 확인하지 못한 사항

- AMP 운영 탐지기의 출력 시계열 이름과 내부 알고리즘 세부, 요금
- ThresholdedRCF 또는 RCFCaster가 상하한 밴드를 직접 출력하는지
- Java RCF jar의 배포 경로와 `python_rcf_wrapper` 설치 절차의 실제 동작(1차 마일스톤에서도 확인하지 않음. rrcf를 기본으로 정해 우선순위 낮음)
- Rust RCF의 파이썬 연동 가능성
- NAB 데이터 파일 개별 라이선스
- OpenSearch Anomaly Detection의 shingle 기본값과 학습 방식(문서 일부만 읽음)
- grafana/promql-anomaly-detection의 정확한 수식과 평활 파라미터
- Prophet 1.5.0 aarch64 wheel의 cmdstan 동봉 여부
- 출력 토큰 길이와 학습 처리량에 쓸 실제 메트릭 이름(적용 환경 기준)
- 적용 환경 저장소의 보존 기간
- 논문의 CoDisp 정의 세부식은 본문 일부만 확인. 위 요약은 정의와 알고리즘 설명 수준

## 참고 자료

RCF
1. Guha et al., Robust Random Cut Forest Based Anomaly Detection On Streams, ICML 2016: https://proceedings.mlr.press/v48/guha16.html (PDF: https://proceedings.mlr.press/v48/guha16.pdf)
2. AWS random-cut-forest-by-aws (Apache-2.0): https://github.com/aws/random-cut-forest-by-aws
3. Python 래퍼 README: https://github.com/aws/random-cut-forest-by-aws/tree/main/python_rcf_wrapper
4. Rust 구현: https://github.com/aws/random-cut-forest-by-aws/tree/main/Rust
5. rrcf (MIT): https://github.com/kLabUM/rrcf
6. OpenSearch Anomaly Detection: https://docs.opensearch.org/docs/latest/observing-your-data/ad/index/

AMP
7. Anomaly detection 사용자 안내: https://docs.aws.amazon.com/prometheus/latest/userguide/prometheus-anomaly-detection.html
8. PreviewAnomalyDetector 안내: https://docs.aws.amazon.com/prometheus/latest/userguide/anomaly-detection-api.html
9. RandomCutForestConfiguration: https://docs.aws.amazon.com/prometheus/latest/APIReference/API_RandomCutForestConfiguration.html
10. CreateAnomalyDetector: https://docs.aws.amazon.com/prometheus/latest/APIReference/API_CreateAnomalyDetector.html
11. AnomalyDetectorMissingDataAction: https://docs.aws.amazon.com/prometheus/latest/APIReference/API_AnomalyDetectorMissingDataAction.html
12. IgnoreNearExpected: https://docs.aws.amazon.com/prometheus/latest/APIReference/API_IgnoreNearExpected.html

예측 밴드와 라이브러리
13. statsforecast (Apache-2.0): https://github.com/Nixtla/statsforecast
14. statsforecast 모델 문서: https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html
15. Prophet (MIT): https://github.com/facebook/prophet
16. Prophet 불확실성 구간: https://meta.github.io/prophet/docs/uncertainty_intervals.html
17. Darts (Apache-2.0): https://github.com/unit8co/darts
18. Darts anomaly detection API: https://unit8co.github.io/darts/generated_api/darts.ad.html
19. River (BSD-3): https://github.com/online-ml/river
20. PyOD (BSD-2): https://github.com/yzhao062/pyod
21. grafana/promql-anomaly-detection (Apache-2.0): https://github.com/grafana/promql-anomaly-detection
22. VictoriaMetrics anomaly detection 개요(엔터프라이즈 라이선스 필요): https://docs.victoriametrics.com/anomaly-detection/

평가
23. NAB (MIT): https://github.com/numenta/NAB

라이브러리 버전 확인(2026-11-25 기준): `gh api repos/<owner>/<repo>`와 PyPI JSON으로 확인

## 변경 이력

- v0.2 (2026-11-25)
  - 특정 제품 연결 전제를 걷어내고 임의의 Prometheus 호환 환경에 붙이는 범용 사이드카로 방향 변경
  - 8단계를 대상 제품 연결 제안에서 통합 가이드로 변경
  - 출력 이름 접두사를 `dbl_`에서 `anomaly_`로 변경
  - RCF 기본 구현을 Java 래퍼에서 rrcf로 변경, Java RCF는 선택적 extra로 격하
  - 1차 마일스톤 범위와 구현 결과를 반영. 결과는 `docs/results/`
- v0.1 (2026-11-25): 초안 작성
