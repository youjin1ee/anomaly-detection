# 3차 마일스톤 AWS Random Cut Forest PoC 결과

- 실행일: 2026-11-26
- 대상: `feat/m3-poc-awsrcf` 브랜치(`feat/m2-integrate`에서 분기). 어댑터 `models/awsrcf.py`, Java 브리지 `sidecar/awsrcf/AwsRcfBridge.java`
- 라이브러리: AWS Random Cut Forest 4.4.0 (Apache-2.0), `randomcutforest-core`와 `randomcutforest-parkservices`, Maven Central의 `software.amazon.randomcutforest`
- JVM: Eclipse Temurin 21.0.13 (linux aarch64)
- 데이터: NAB 레포 커밋 `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`, 시계열 58개, 레코드 365558개, 라벨 윈도우 116개. m2 결과와 같은 데이터와 채점기
- 실행 환경: ARM64 원격 장비(20코어). 코어 2개에 고정(`taskset`), `nice 19`, 프로세스 주소 공간 8GB 상한, JVM 힙 `-Xmx1g`(NAB 실행)와 `-Xmx2g` 이하, 병렬 GC 스레드 2. GPU 미사용. 다른 작업이 같은 장비에서 돌고 있어 부하 평균이 20에서 33(코어 20개)이었음. 시간과 지연 수치는 참고용
- 방법 정의: `docs/evaluation.md`의 NAB 절, 연동 설계는 `docs/design.md`의 AWS RCF 절

## 요약

- NAB 점수표의 AWS RCF 설정을 NAB 레포 `random_cut_forest.sql`에서 찾아 재현함. 100 트리, 표본 256, 시간 감쇠 100000 레코드, shingle 1, 점수를 8.0(log2 256)으로 나눔
  - Java 4.4.0 코어로 재현한 결과 standard 52.3, reward_low_FP_rate 39.4, reward_low_FN_rate 60.7. 공식 표기는 51.7, 38.4, 59.7. 시드 3개 평균은 53.0, 39.6, 61.1
- 같은 라이브러리에 우리 파라미터 규칙(shingle 8, 트리 50, 입력 정규화, 임계값 대비 점수)을 쓰면 standard 31.0(시드 3개 평균 30.2). rrcf 기반 rcf는 21.7(평균 20.4). 계절성 기준선은 0.0
  - 같은 알고리즘의 점수 차이에서 설정 쪽이 가장 크게 보임(52.3 대 31.0). 라이브러리 쪽 차이는 우리 설정 기준 약 9점(31.0 대 21.7). shingle 길이와 점수 환산을 따로 떼어 측정하지는 않음
- 밴드 출력 있음. `RCFCaster`가 다음 표본의 예측값과 하한 상한을 내고 어댑터 `caster` 모드가 이를 `expected`, `lower`, `upper`로 전달함. `ThresholdedRandomCutForest`는 밴드 없음
- 연동은 Java 자식 프로세스와 표준 입출력 줄 프로토콜로 결정. JPype와 1스텝 지연 차이는 p50 기준 0.04ms
- 운영 비용(공유 장비, 코어 2개): 시계열 500개 한 라운드 0.40에서 0.60초, 힙 사용 446에서 575MB(전체 GC 뒤). 1분 주기에 들어옴. rrcf는 같은 조건 추정값이 6.6초와 5에서 6GB
- 기본 점수 환산(임계값에서 0.5)은 알림용으로 너무 민감함. 점수 0.5 이상 3연속 규칙에서 오탐이 1217건(시계열당 하루 0.60건)이고 고정 임계값 0.5의 NAB 점수는 -548.7
- 장애 주입 데이터(1차 20건): 탐지율 awsrcf 0.35, awsrcf-trcf 0.30, awsrcf-nab 0.05로 fixed 0.60, seasonal 0.65에 못 미침. rrcf 0.15보다는 awsrcf 계열 두 설정이 높고 awsrcf-nab이 낮음. 지연 5건(2차)은 모든 모델이 0건. 첫 이벤트 뒤 점수가 낮아지는 rrcf의 형태가 AWS RCF에서도 일부 보임
- QA 환경 실데이터 오탐: 하루 0.1건 기준은 fixed(0.010)와 awsrcf-nab(0.033)만 충족. awsrcf는 보호 설정 켬 1.044로 기준의 10배, seasonal(1.361)과 비슷한 크기

## 확인한 사실

### 밴드와 기대값

- `RandomCutForest`(core): 표본 점수만 냄. 임계값과 밴드 없음
- `ThresholdedRandomCutForest`(parkservices): RCF 점수, anomaly grade(정상 0, 최대 1), 라이브러리가 유지하는 임계값을 냄. 기대값은 이상으로 판정한 표본에서만 채워짐. 밴드 없음. 소스(`AnomalyDescriptor`, `RCFComputeDescriptor`)와 실행으로 확인
- `RCFCaster`(parkservices): `ThresholdedRandomCutForest`를 상속하고 `forecastHorizon` 만큼의 예측과 `lower`, `upper`를 단계마다 냄. 예측 오차를 쌓아 보정함(`SIMPLE`, `MINIMAL`, `NONE`). 실행으로 확인(스모크: 표본 1500에서 expected 95.06, lower 91.64, upper 97.47)
- 밴드는 보정된 예측 구간이라 계절성 기준선의 중앙값 기준 밴드와 정의가 다름. 밴드 품질(실제 값이 구간에 든 비율)은 이번에 평가하지 않음

### 파이썬 연동 가능성

- Rust 구현은 `Cargo.toml`의 패키지 `rcf` 4.0.0, 라이브러리 이름 `rcflib`. 파이썬 바인딩 설정과 PyPI 패키지 모두 없음
- 상류 레포의 `python_rcf_wrapper`는 JPype로 Java 클래스를 부르는 래퍼이며 4.0.0-SNAPSHOT jar를 동봉함. 패키지가 아님
- Maven Central 4.4.0 jar로 ARM64 Temurin 21에서 실행됨. amd64는 실행하지 않음

### NAB 공식 설정 출처

- 출처: NAB 레포 `nab/detectors/random_cut_forest/random_cut_forest.sql`과 같은 폴더의 `random_cut_forest.py`, `README.md`. master를 읽었고 고정 커밋 `ea702d7`과 같은 내용인지는 대조하지 않음
- 이 점수는 AWS Kinesis Analytics의 `RANDOM_CUT_FOREST` SQL 함수로 낸 것이며 Java 라이브러리가 아님
- 파라미터: `numberOfTrees` 100, `subSampleSize` 256, `timeDecay` 100000, `shingleSize` 1. 점수는 0에서 log2(subSampleSize) 범위라 8.0으로 나눠 0에서 1로 맞춤
- 파일마다 스트림 애플리케이션을 멈췄다 시작해 알고리즘을 초기화함. 입력은 타임스탬프와 값 두 열이고 숫자 열인 값만 쓰였을 것으로 추정하나 확인하지 못함
- SQL 주석에 shingleSize 4, 24, 48과 트리 100, 200을 시험해 봤으나 NAB 점수가 오르지 않았다고 적혀 있음
- 이 레포의 해석: `timeDecay` 100000을 Java의 감쇠율 1e-5로 옮김(레코드 100000개 시간 척도라는 해석). Kinesis 쪽 정의를 문서로 확인하지는 못함

## 연동 방식

| 방식 | 판단 |
|---|---|
| JPype로 jar 직접 호출 | 네이티브 확장 의존성 추가, JVM 충돌이나 메모리 부족이 사이드카를 같이 죽임, 힙이 파이썬과 한 프로세스 한도를 나눔. 1스텝 지연은 자식 프로세스와 거의 같음(아래) |
| Rust 구현의 파이썬 바인딩 | 없음. 직접 만들어 두 아키텍처로 빌드해야 함. 하지 않음 |
| Java 자식 프로세스와 표준 입출력(선택) | 파이썬 의존성 없음, JVM 힙 한도를 따로 걸 수 있음, JVM 하나가 파이썬 프로세스의 모든 시계열을 맡음, 실행에는 `java`와 jar 6개만 필요 |

- 선택 사유: 의존성 부담이 가장 작고 장애 범위가 분리되며 지연 손해가 측정 오차 수준
- 선택적 extra 처리: 추가되는 것은 jar와 JVM이고 파이썬 패키지는 없어 파이썬 extra를 두지 않음. 기본 설치는 바뀌지 않고 `import`는 Java를 건드리지 않음. `ANOMALY_AWSRCF_HOME` 없이 모델을 만들면 변수 이름을 알려 주는 오류가 남
- 브리지 jar는 JDK로 `scripts/build-awsrcf-bridge.sh`가 컴파일함. 레포에 jar와 JDK는 넣지 않음
- 실행에 필요한 JDK 모듈은 `java.base` 하나(jlink 이미지로 스모크 확인)

## 어댑터와 점수 매핑

| 모드 | 내부 | 점수 | expected와 밴드 |
|---|---|---|---|
| `rcf` | 코어 `RandomCutForest` | 원점수를 `score_div`(기본 log2 표본 크기)로 나눠 0에서 1로 자름 | 밴드 없음, expected는 최근 60개 중앙값 |
| `trcf` | `ThresholdedRandomCutForest` | `logistic((raw / threshold - 1) / 0.25)`, 라이브러리 임계값에서 0.5 | 밴드 없음, expected는 최근 중앙값 |
| `caster`(기본) | `RCFCaster`, 예측 horizon 1, 보정 `SIMPLE` | `trcf`와 같음 | 직전 표본 뒤에 만든 1스텝 예측이 다음 표본의 expected, lower, upper |

- `score_from: grade`이면 `trcf`, `caster`의 점수를 라이브러리의 anomaly grade로 바꿈
- 첫 예측이 나오기 전에는 밴드가 `None`이고 expected는 최근 중앙값
- 방향, 허용 오차, 저트래픽 하한은 기존대로 `SeriesPipeline`에서 처리
- 임계값 대비 점수 환산은 NAB 점수를 보기 전에 정한 고정 규칙이며 NAB 결과에 맞춰 고치지 않음

## NAB 평가

### 평가 조건

- 모델
  - `awsrcf-nab`: 위 공식 설정. 모드 `rcf`, shingle 1, 트리 100, 표본 256, 감쇠 1e-5, 점수를 8.0으로 나눔. 파일별 규칙 없음
  - `awsrcf`: 모드 `caster`. 표본 크기와 `reference_window`는 rrcf `rcf`와 같은 규칙(워밍업 끝에서 숲이 차도록), 나머지는 어댑터 기본값(shingle 8, 트리 50, 점수는 임계값 대비)
  - `awsrcf-trcf`: `awsrcf`에서 모드만 `trcf`(밴드 없음)
  - `awsrcf-grade`: `awsrcf`에서 점수만 anomaly grade. NAB 점수를 본 뒤에 추가한 대안 환산이며 아래 해석에서 구분해 다룸
  - `rcf`: rrcf 기반(m2와 같은 코드, m2 점수 파일을 그대로 사용). `seasonal`: 계절성 기준선(m2와 같음)
- 파일별 튜닝 없음. 파라미터는 샘플링 간격과 워밍업 길이에서만 정함
- 시드: `awsrcf-nab`, `awsrcf-trcf`, `awsrcf`는 시드 0, 1, 2. 주 결과표는 시드 0. `awsrcf-grade`는 시드 0만
- NAB 점수 임계값은 모델과 프로필마다 코퍼스에서 최적인 값 하나(공식 방식). 평가 데이터로 맞춘 값이라 낙관적. 고정 임계값 0.5의 점수도 같이 기록
- `+for3`은 알림 규칙의 `for: 3`을 입힌 변형(최근 3개 레코드 점수의 최솟값)

### 공식 설정 재현

| 프로필 | 공식 표기 | 이 레포 awsrcf-nab(시드 0) | 차이 | 시드 3개 평균 | 시드 범위 |
|---|---|---|---|---|---|
| standard | 51.7 | 52.3 | +0.6 | 53.0 | 52.3 에서 53.5 |
| reward_low_FP_rate | 38.4 | 39.4 | +1.0 | 39.6 | 39.4 에서 40.1 |
| reward_low_FN_rate | 59.7 | 60.7 | +1.0 | 61.1 | 60.7 에서 61.5 |

- 판정: 재현됨(1점 안팎). 차이가 남는 이유로 라이브러리 구현 버전(Kinesis 함수 대 Java 4.4.0), 입력 처리, 시드를 생각할 수 있으나 어느 쪽인지 가르지 않음
- 공식 점수 CSV를 우리 채점기에 넣은 검증은 m2에서 완료(randomCutForest standard 51.7171). 이번에는 모델 쪽만 재현함

### 코퍼스 점수, 모델과 프로필마다 최적 임계값 하나 (시드 0)

| model | standard | threshold | reward_low_FP_rate | threshold | reward_low_FN_rate | threshold |
|---|---|---|---|---|---|---|
| awsrcf-nab | 52.3 | 0.4295 | 39.4 | 0.5370 | 60.7 | 0.4295 |
| awsrcf-nab+for3 | 35.1 | 0.2579 | 29.4 | 0.3344 | 40.1 | 0.2579 |
| awsrcf-trcf | 32.1 | 0.9978 | 22.4 | 0.9998 | 38.6 | 0.9978 |
| awsrcf-trcf+for3 | 32.2 | 0.9947 | 23.3 | 0.9990 | 39.8 | 0.9891 |
| awsrcf | 31.0 | 0.9978 | 21.5 | 0.9998 | 37.9 | 0.9978 |
| awsrcf+for3 | 31.7 | 0.9947 | 21.8 | 0.9996 | 38.4 | 0.9947 |
| awsrcf-grade | 21.2 | 1.0000 | 0.0 | 1.1000 | 41.4 | 1.0000 |
| awsrcf-grade+for3 | 18.1 | 0.8125 | 7.5 | 1.0000 | 23.8 | 0.5625 |
| rcf | 21.7 | 0.6607 | 10.8 | 0.8976 | 27.7 | 0.6607 |
| rcf+for3 | 23.8 | 0.5015 | 15.6 | 0.5960 | 28.7 | 0.3235 |
| seasonal | 0.0 | 1.1000 | 0.0 | 1.1000 | 0.0 | 1.1000 |
| seasonal+for3 | 0.0 | 1.1000 | 0.0 | 1.1000 | 0.0 | 1.1000 |

### 시드별 점수

| model | profile | seed 0 | seed 1 | seed 2 | mean | min | max |
|---|---|---|---|---|---|---|---|
| awsrcf-nab | standard | 52.3 | 53.3 | 53.5 | 53.0 | 52.3 | 53.5 |
| awsrcf-nab | reward_low_FP_rate | 39.4 | 39.4 | 40.1 | 39.6 | 39.4 | 40.1 |
| awsrcf-nab | reward_low_FN_rate | 60.7 | 61.1 | 61.5 | 61.1 | 60.7 | 61.5 |
| awsrcf-nab+for3 | standard | 35.1 | 34.6 | 34.7 | 34.8 | 34.6 | 35.1 |
| awsrcf-nab+for3 | reward_low_FP_rate | 29.4 | 30.1 | 29.6 | 29.7 | 29.4 | 30.1 |
| awsrcf-nab+for3 | reward_low_FN_rate | 40.1 | 39.5 | 38.9 | 39.5 | 38.9 | 40.1 |
| awsrcf-trcf | standard | 32.1 | 30.7 | 30.5 | 31.1 | 30.5 | 32.1 |
| awsrcf-trcf | reward_low_FP_rate | 22.4 | 23.5 | 24.4 | 23.5 | 22.4 | 24.4 |
| awsrcf-trcf | reward_low_FN_rate | 38.6 | 37.4 | 36.3 | 37.5 | 36.3 | 38.6 |
| awsrcf-trcf+for3 | standard | 32.2 | 32.8 | 29.1 | 31.4 | 29.1 | 32.8 |
| awsrcf-trcf+for3 | reward_low_FP_rate | 23.3 | 24.7 | 24.1 | 24.0 | 23.3 | 24.7 |
| awsrcf-trcf+for3 | reward_low_FN_rate | 39.8 | 38.9 | 38.7 | 39.1 | 38.7 | 39.8 |
| awsrcf | standard | 31.0 | 29.6 | 30.0 | 30.2 | 29.6 | 31.0 |
| awsrcf | reward_low_FP_rate | 21.5 | 22.8 | 23.5 | 22.6 | 21.5 | 23.5 |
| awsrcf | reward_low_FN_rate | 37.9 | 36.7 | 35.6 | 36.7 | 35.6 | 37.9 |
| awsrcf+for3 | standard | 31.7 | 32.3 | 28.2 | 30.7 | 28.2 | 32.3 |
| awsrcf+for3 | reward_low_FP_rate | 21.8 | 23.3 | 22.8 | 22.6 | 21.8 | 23.3 |
| awsrcf+for3 | reward_low_FN_rate | 38.4 | 38.8 | 36.7 | 37.9 | 36.7 | 38.8 |

### 고정 임계값 0.5의 점수 (시드 0)

| model | standard | reward_low_FP_rate | reward_low_FN_rate |
|---|---|---|---|
| awsrcf-nab | 49.2 | 36.9 | 54.9 |
| awsrcf-nab+for3 | 19.7 | 18.8 | 20.6 |
| awsrcf-trcf | -499.7 | -1088.3 | -302.4 |
| awsrcf-trcf+for3 | -289.4 | -663.7 | -163.1 |
| awsrcf | -548.7 | -1187.1 | -334.8 |
| awsrcf+for3 | -350.2 | -787.2 | -203.0 |
| awsrcf-grade | -5.9 | -93.3 | 24.8 |
| awsrcf-grade+for3 | 16.0 | -2.0 | 23.0 |
| rcf | 16.4 | -6.9 | 25.3 |
| rcf+for3 | 23.8 | 14.8 | 28.0 |
| seasonal | -905.9 | -1901.9 | -572.9 |
| seasonal+for3 | -269.2 | -602.6 | -157.1 |

### 카테고리별 NAB 점수, standard (시드 0, 탐지가 있는 코퍼스 임계값)

| model | artificialNoAnomaly | artificialWithAnomaly | realAdExchange | realAWSCloudwatch | realKnownCause | realTraffic | realTweets |
|---|---|---|---|---|---|---|---|
| awsrcf-nab | n/a | 29.0 | 62.4 | 58.0 | 43.1 | 69.4 | 45.3 |
| awsrcf-nab+for3 | n/a | 46.8 | 0.0 | 35.9 | 36.9 | 44.3 | 42.2 |
| awsrcf-trcf | n/a | 31.8 | 53.1 | 33.8 | 23.5 | 30.8 | 27.2 |
| awsrcf-trcf+for3 | n/a | 0.0 | 52.3 | 32.9 | 25.1 | 36.4 | 31.1 |
| awsrcf | n/a | 31.8 | 53.1 | 33.6 | 21.2 | 30.8 | 25.0 |
| awsrcf+for3 | n/a | 31.8 | 52.3 | 32.4 | 21.9 | 36.4 | 25.9 |
| awsrcf-grade | n/a | 1.8 | 70.0 | 53.9 | -22.6 | 44.4 | -3.0 |
| awsrcf-grade+for3 | n/a | 20.8 | -0.4 | 28.8 | -15.8 | 37.2 | 32.4 |
| rcf | n/a | 35.8 | 20.3 | 17.8 | 34.0 | 36.6 | 11.1 |
| rcf+for3 | n/a | 11.5 | 13.2 | 22.2 | 36.3 | 32.5 | 21.2 |
| seasonal | n/a | -269.6 | 38.1 | -318.0 | -185.5 | 3.1 | -1110.3 |
| seasonal+for3 | n/a | 7.5 | 0.0 | -146.8 | -101.0 | -0.8 | -185.0 |

reward_low_FP_rate, reward_low_FN_rate의 카테고리별 표는 같은 보고서에 있으며 이 문서에는 싣지 않음(`anomaly-detection-nab report` 출력).

### 카테고리별 이벤트 지표 (시드 0, 점수 0.5 이상 3개 연속)

윈도우는 참 이벤트, 탐지 지연은 윈도우 시작 기준. 정의는 m2 결과 문서와 같음

| category | model | windows | detected | recall | alerts | false alarms | precision | F1 | delay median (min) | delay p90 (min) | false alarms per day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| artificialNoAnomaly | awsrcf-nab | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialNoAnomaly | awsrcf-trcf | 0 | 0 | n/a | 79 | 79 | 0.00 | n/a | n/a | n/a | 1.33 |
| artificialNoAnomaly | awsrcf | 0 | 0 | n/a | 79 | 79 | 0.00 | n/a | n/a | n/a | 1.33 |
| artificialNoAnomaly | awsrcf-grade | 0 | 0 | n/a | 18 | 18 | 0.00 | n/a | n/a | n/a | 0.30 |
| artificialNoAnomaly | rcf | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialNoAnomaly | seasonal | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialWithAnomaly | awsrcf-nab | 6 | 1 | 0.17 | 1 | 0 | 1.00 | 0.29 | 1015.0 | 1015.0 | 0.00 |
| artificialWithAnomaly | awsrcf-trcf | 6 | 5 | 0.83 | 77 | 62 | 0.19 | 0.32 | 120.0 | 494.0 | 0.87 |
| artificialWithAnomaly | awsrcf | 6 | 6 | 1.00 | 127 | 107 | 0.16 | 0.27 | 122.5 | 537.5 | 1.50 |
| artificialWithAnomaly | awsrcf-grade | 6 | 3 | 0.50 | 13 | 8 | 0.38 | 0.43 | 1020.0 | 1336.0 | 0.11 |
| artificialWithAnomaly | rcf | 6 | 1 | 0.17 | 3 | 2 | 0.33 | 0.22 | 1015.0 | 1015.0 | 0.03 |
| artificialWithAnomaly | seasonal | 6 | 5 | 0.83 | 17 | 6 | 0.65 | 0.73 | 1030.0 | 1101.0 | 0.08 |
| realAdExchange | awsrcf-nab | 14 | 0 | 0.00 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| realAdExchange | awsrcf-trcf | 14 | 13 | 0.93 | 69 | 53 | 0.23 | 0.37 | 1620.0 | 3108.0 | 0.15 |
| realAdExchange | awsrcf | 14 | 13 | 0.93 | 69 | 53 | 0.23 | 0.37 | 1620.0 | 3108.0 | 0.15 |
| realAdExchange | awsrcf-grade | 14 | 1 | 0.07 | 2 | 1 | 0.50 | 0.12 | 240.0 | 240.0 | 0.00 |
| realAdExchange | rcf | 14 | 2 | 0.14 | 2 | 0 | 1.00 | 0.25 | 3330.0 | 4602.0 | 0.00 |
| realAdExchange | seasonal | 14 | 3 | 0.21 | 19 | 16 | 0.16 | 0.18 | 2460.0 | 2508.0 | 0.05 |
| realAWSCloudwatch | awsrcf-nab | 30 | 8 | 0.27 | 8 | 0 | 1.00 | 0.42 | 557.5 | 1085.5 | 0.00 |
| realAWSCloudwatch | awsrcf-trcf | 30 | 23 | 0.77 | 235 | 193 | 0.18 | 0.29 | 425.0 | 597.0 | 0.97 |
| realAWSCloudwatch | awsrcf | 30 | 24 | 0.80 | 241 | 198 | 0.18 | 0.29 | 437.5 | 590.5 | 0.99 |
| realAWSCloudwatch | awsrcf-grade | 30 | 10 | 0.33 | 19 | 9 | 0.53 | 0.41 | 510.0 | 1048.5 | 0.05 |
| realAWSCloudwatch | rcf | 30 | 9 | 0.30 | 18 | 7 | 0.61 | 0.40 | 510.0 | 987.0 | 0.04 |
| realAWSCloudwatch | seasonal | 30 | 16 | 0.53 | 179 | 152 | 0.15 | 0.24 | 380.0 | 1022.5 | 0.76 |
| realKnownCause | awsrcf-nab | 19 | 2 | 0.11 | 2 | 0 | 1.00 | 0.19 | 867.5 | 1285.5 | 0.00 |
| realKnownCause | awsrcf-trcf | 19 | 17 | 0.89 | 217 | 172 | 0.21 | 0.34 | 1745.0 | 5472.0 | 0.26 |
| realKnownCause | awsrcf | 19 | 17 | 0.89 | 223 | 177 | 0.21 | 0.34 | 1745.0 | 5472.0 | 0.26 |
| realKnownCause | awsrcf-grade | 19 | 7 | 0.37 | 50 | 39 | 0.22 | 0.28 | 2400.0 | 5445.0 | 0.06 |
| realKnownCause | rcf | 19 | 10 | 0.53 | 16 | 6 | 0.62 | 0.57 | 1845.0 | 7719.0 | 0.01 |
| realKnownCause | seasonal | 19 | 14 | 0.74 | 117 | 93 | 0.21 | 0.32 | 1230.0 | 3024.0 | 0.14 |
| realTraffic | awsrcf-nab | 14 | 1 | 0.07 | 2 | 1 | 0.50 | 0.12 | 440.0 | 440.0 | 0.01 |
| realTraffic | awsrcf-trcf | 14 | 13 | 0.93 | 78 | 60 | 0.23 | 0.37 | 240.0 | 701.0 | 0.38 |
| realTraffic | awsrcf | 14 | 13 | 0.93 | 78 | 60 | 0.23 | 0.37 | 240.0 | 701.0 | 0.38 |
| realTraffic | awsrcf-grade | 14 | 6 | 0.43 | 11 | 5 | 0.55 | 0.48 | 432.5 | 726.0 | 0.03 |
| realTraffic | rcf | 14 | 5 | 0.36 | 6 | 1 | 0.83 | 0.50 | 440.0 | 1353.0 | 0.01 |
| realTraffic | seasonal | 14 | 8 | 0.57 | 28 | 16 | 0.43 | 0.49 | 442.5 | 2251.9 | 0.10 |
| realTweets | awsrcf-nab | 33 | 14 | 0.42 | 22 | 6 | 0.73 | 0.54 | 992.5 | 1789.5 | 0.01 |
| realTweets | awsrcf-trcf | 33 | 33 | 1.00 | 603 | 506 | 0.16 | 0.28 | 785.0 | 1093.0 | 0.96 |
| realTweets | awsrcf | 33 | 33 | 1.00 | 647 | 543 | 0.16 | 0.28 | 660.0 | 1093.0 | 1.03 |
| realTweets | awsrcf-grade | 33 | 16 | 0.48 | 50 | 32 | 0.36 | 0.41 | 967.5 | 1985.0 | 0.06 |
| realTweets | rcf | 33 | 15 | 0.45 | 48 | 32 | 0.33 | 0.38 | 995.0 | 1991.0 | 0.06 |
| realTweets | seasonal | 33 | 29 | 0.88 | 447 | 369 | 0.17 | 0.29 | 585.0 | 1038.0 | 0.70 |
| all | awsrcf-nab | 116 | 26 | 0.22 | 35 | 7 | 0.80 | 0.35 | 990.0 | 1367.5 | 0.00 |
| all | awsrcf-trcf | 116 | 104 | 0.90 | 1358 | 1125 | 0.17 | 0.29 | 550.0 | 2442.0 | 0.55 |
| all | awsrcf | 116 | 106 | 0.91 | 1464 | 1217 | 0.17 | 0.28 | 510.0 | 2430.0 | 0.60 |
| all | awsrcf-grade | 116 | 43 | 0.37 | 163 | 112 | 0.31 | 0.34 | 910.0 | 2158.0 | 0.05 |
| all | rcf | 116 | 42 | 0.36 | 93 | 48 | 0.48 | 0.41 | 980.0 | 3052.0 | 0.02 |
| all | seasonal | 116 | 75 | 0.65 | 807 | 652 | 0.19 | 0.30 | 585.0 | 1983.0 | 0.32 |

### 비용 (NAB 실행, 파이프 왕복 포함)

| model | microseconds per record (single process) |
|---|---|
| awsrcf-nab | 618 |
| awsrcf-trcf | 545 |
| awsrcf | 1010 |
| awsrcf-grade | 1102 |
| rcf | 14052 |
| seasonal | 83 |

### 해석

- 공식 설정 재현: awsrcf-nab이 52.3, 39.4, 60.7로 공식 표기 51.7, 38.4, 59.7에 근접함. 시드 3개 평균이 공식 표기보다 조금 높음
- 설정 영향: 같은 라이브러리에서 공식 설정(52.3)과 우리 규칙(awsrcf 31.0, awsrcf-trcf 32.1)의 차이가 21점 이상. 두 설정은 shingle(1 대 8), 입력 정규화, 점수 환산, 트리 수, 표본 크기가 모두 달라 어느 항목이 원인인지는 모름
- 구현 영향: 우리 규칙 기준으로 rrcf(rcf 21.7, 3시드 평균 20.4)와 AWS 라이브러리(awsrcf 31.0, 3시드 평균 30.2)가 약 9점 차이. rrcf를 shingle 1 설정으로 돌려 보지는 않아 구현 차이와 설정 차이를 완전히 떼지는 못함
- `for3`: awsrcf-nab은 52.3에서 35.1로 내려감. 공식 설정의 점수는 순간적으로 치솟는 형태라 3연속 조건과 맞지 않는 것으로 추정하나 확인하지는 않음. 우리 규칙의 awsrcf는 31.0에서 31.7로 거의 같음
- 알림 규칙(0.5, 3연속) 기준 이벤트 지표
  - awsrcf-nab: 재현율 0.22, precision 0.80, 오탐 7건, F1 0.35. 점수가 보수적으로 나와 놓치는 윈도우가 많음
  - awsrcf, awsrcf-trcf: 재현율 0.91과 0.90, 오탐 1217건과 1125건(하루 0.60건과 0.55건), F1 0.28과 0.29. 환산이 임계값에서 0.5를 주기 때문에 점수가 레코드마다 알림 임계에 가까워 너무 민감함. 이 환산으로 고정 임계값 0.5의 NAB 점수는 -548.7과 -499.7이고 최적 임계값은 0.9978
  - awsrcf-grade: 재현율 0.37, precision 0.31, 오탐 112건(하루 0.05건), F1 0.34
  - rcf(rrcf): F1 0.41이 6개 모델 중 가장 높음. 이 알림 규칙에서는 AWS 라이브러리가 rrcf보다 나은 점이 없음
  - 계절성 기준선: 재현율 0.65, 오탐 652건, F1 0.30
- awsrcf-grade의 NAB 점수는 standard 21.2, reward_low_FP_rate 0.0, reward_low_FN_rate 41.4로 불균일하고 카테고리 편차가 큼(realKnownCause -22.6, realTweets -3.0, realAdExchange 70.0). NAB 점수를 보고 추가한 환산이라 개선으로 읽지 않음
- 최적 임계값 점수 기준 순위(standard): awsrcf-nab 52.3, awsrcf-trcf 32.1, awsrcf 31.0, rcf 21.7, seasonal 0.0
- 카테고리 점수 편차: 카테고리당 윈도우가 6개에서 33개라 일반화하지 않음

## 운영 현실성

측정 조건: ARM64 원격 장비, 코어 2개 고정, nice 19, 부하 평균 20에서 33(코어 20개)인 공유 장비. 합성 시계열(사인과 잡음, 1분 간격), 표본 크기 256, 트리는 rcf 100과 trcf, caster 50, shingle은 rcf 1과 trcf, caster 8. `scripts/bench_awsrcf.py`와 `scripts/bench_awsrcf_jpype.py`

### 1스텝 지연, 시계열 1개

| 모드 | 실행 | p50 ms | p99 ms | max ms |
|---|---|---|---|---|
| rcf | A | 0.316 | 3.589 | 12.907 |
| rcf | B | 0.728 | 4.414 | 6.345 |
| trcf | A | 0.278 | 2.639 | 10.500 |
| trcf | B | 0.667 | 4.943 | 6.775 |
| caster | A | 0.418 | 3.905 | 13.089 |
| caster | B | 0.948 | 5.293 | 13.148 |
| rrcf(rcf, shingle 8, 트리 40, 표본 256) | A | 12.702 | 16.287 | 20.915 |
| JPype 직접 호출(trcf와 같은 설정) | 별도 | 0.237 | 1.949 | 없음 |

- 실행 A: 워밍업 400, 측정 3000스텝. 실행 B: 워밍업 300, 측정 500스텝, 새 JVM. 같은 측정이 두 배 가까이 흔들려 공유 장비 부하 영향이 큼
- JPype와 자식 프로세스의 p50 차이는 0.04ms(실행 A의 trcf 0.278 대 0.237)
- 한 스텝에 들어간 시간은 JIT 워밍업과 GC 포함. max는 한 번씩 나온 값

### 시계열 500개, 한 스텝(모든 시계열에 표본 하나씩)

| 모드 | 워밍업 표본 | 라운드 p50 초 | 라운드 max 초 | 시계열 생성 초 | 힙 사용 MB(전체 GC 뒤) | JVM RSS MB(-Xmx1g) |
|---|---|---|---|---|---|---|
| rcf | 300 | 0.482 | 0.694 | 0.546 | 446 | 877 |
| trcf | 300 | 0.565 | 0.625 | 0.837 | 574 | 1007 |
| caster | 300 | 0.568 | 0.594 | 0.902 | 575 | 1011 |
| rcf(실행 A) | 400 | 0.441 | 0.645 | 0.572 | 630(GC 전) | 781 |
| trcf(실행 A) | 400 | 0.400 | 0.557 | 0.801 | 799(GC 전) | 1007 |
| caster(실행 A) | 400 | 0.599 | 0.680 | 0.881 | 692(GC 전) | 1002 |
| rrcf 시계열 20개(실행 A) | 600 | 0.265 | 0.265 | 0.091 | 해당 없음 | 파이썬 RSS 248 |

- JVM RSS는 `-Xmx1g`에 걸려 있어 점유량보다는 상한에 가까운 값. 시계열당 힙은 전체 GC 뒤 기준 약 0.9에서 1.15MB
- 500개 라운드 0.4에서 0.6초는 1분 간격의 1퍼센트 안팎. 한 시계열 약 1.2ms(직접 측정 없이 500개에서 선형으로 늘린 추정)
- rrcf의 500개 값은 20개 측정을 선형으로 늘린 추정. 라운드 약 6.6초. 20개에서 파이썬 RSS 248MB라 시계열당 약 10에서 12MB이고 500개면 5에서 6GB
- 시작 시 백필 비용: 시계열 500개에 표본 300개를 넣는 데 117에서 129초(표본당 약 0.8ms). 1일 백필(1분 간격 1440표본)을 500개에 하면 직접 재지 않은 추정치로 약 600초. 상태를 저장하지 않으므로 재시작마다 반복됨

### JVM 기동과 이미지 증가분

- JVM 기동(프로세스 시작에서 첫 응답까지): 실행 A 5회 0.084에서 0.100초, 실행 B의 caster 5회 0.101에서 0.108초. JPype는 JVM 로드에 0.371초. 첫 시계열 생성 시 클래스 로딩은 포함하지 않음
- 이미지 증가분(디렉터리 크기 측정, 이미지를 빌드해 잰 값은 아님)
  - jar 6개(코어, parkservices, jackson 3종, 브리지): 2,680KB
  - `jlink`로 `java.base`만 담은 런타임: 39,528KB(약 38.6MB). 스모크로 세 모드 실행 확인
  - Temurin 21 JRE 전체: 161,740KB(타르볼 압축 51.3MB, 압축을 푼 디렉터리를 맥에서 측정)
  - Temurin 21 JDK: 354,752KB(컴파일할 때만 필요, 이미지에는 불필요)
  - 증가분 추정: jlink 런타임을 쓰면 약 42MB, JRE 전체를 쓰면 약 164MB. 파이썬 slim 기본 이미지의 크기는 이번에 재지 않음
- 런타임 메모리: 시계열 수에 비례(위 표). 힙 한도 `-Xmx`로 제한. 힙 4GB면 한 JVM의 한도는 시계열 약 3500개로 추정. 1.15MB를 곱해 낸 값이며 직접 측정은 없음

### 1분 주기 판정

- 시계열 500개 이하: 가능. 라운드 0.4에서 0.6초로 1분의 1퍼센트 안팎, 공유 장비의 코어 2개에서 잰 값
- 병목은 시간이 아닌 메모리. 시계열당 약 1.15MB라 힙이 먼저 참
- 재시작 후 백필이 가장 큰 비용. 500개에 1일 백필이면 약 10분 동안 정상 점수가 나오지 않음(추정)
- 이번 PoC 범위 밖: amd64, 부하 중인 실제 스토어와의 연동, 장시간 안정성(수 시간 이상 실행)

## 장애 주입 데이터

장비에서 실제 LLM 서빙 엔진에 압축 일 주기 부하(1일 24분)를 걸고 장애를 예약 주입해 얻은 데이터. 정답 라벨이 있음. 실험 구성과 신호 정의는 `examples/inject-lab/README.md`. 재생 도구와 평가 규칙(점수 0.5 이상 3샘플 연속, 시간은 압축 일 주기를 86400초로 되돌려 모델에 입력)은 장애 주입 실험 브랜치의 `anomaly_detection.lab.replay`를 그대로 가져와 `python -m anomaly_detection.lab.replay`로 실행함. 모델 파라미터는 NAB 평가와 달리 파일별 규칙 없이 기본값

- 1차 실행: 4개 시나리오 각 5건, 20건. 2차 실행: 지연 증가 5건(주입 강도를 높인 실행)
- 모델: `fixed`, `seasonal`, `rcf`(rrcf)는 기본값. `awsrcf-nab`은 NAB 점수표 설정(shingle 1, 트리 100, 표본 256, 감쇠 1e-5, 점수를 8.0으로 나눔). `awsrcf`는 모드 `caster`, `awsrcf-trcf`는 모드 `trcf`로 어댑터 기본값(shingle 8, 트리 50, 표본 256)
- 기존 모델 3개의 수치는 `docs/results/m2-inject-lab.md`의 값과 같게 재현됨(fixed 탐지율 0.60, seasonal 0.65, rcf 0.15)
- 시드는 모델마다 1개, 이벤트는 시나리오당 5건이라 탐지율의 신뢰 구간이 매우 넓음

### 1차 실행, 시나리오 합산 (신호 4개, 오탐은 신호당 하루당)

| model | events | detected | recall | alerts | false alarms | precision | F1 | delay median (s) | false alarms per signal per lab day |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 20 | 12 | 0.60 | 30 | 18 | 0.40 | 0.48 | 58 | 0.51 |
| seasonal | 20 | 13 | 0.65 | 34 | 21 | 0.38 | 0.48 | 70 | 0.60 |
| rcf | 20 | 3 | 0.15 | 6 | 3 | 0.50 | 0.23 | 65 | 0.09 |
| awsrcf-nab | 20 | 1 | 0.05 | 4 | 3 | 0.25 | 0.08 | 30 | 0.09 |
| awsrcf | 20 | 7 | 0.35 | 24 | 17 | 0.29 | 0.32 | 60 | 0.48 |
| awsrcf-trcf | 20 | 6 | 0.30 | 16 | 10 | 0.38 | 0.33 | 58 | 0.28 |

### 1차 실행, 시나리오별

| scenario | signal | model | events | detected | recall | alerts | false alarms | of which during other injections | precision | F1 | delay median (s) | delay p90 (s) | false alarms per lab day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latency_rise | client_latency | fixed | 5 | 0 | 0.00 | 15 | 15 | 4 | 0.00 | 0.00 | n/a | n/a | 1.71 |
| latency_rise | client_latency | seasonal | 5 | 1 | 0.20 | 13 | 12 | 3 | 0.08 | 0.11 | 110 | 110 | 1.36 |
| latency_rise | client_latency | rcf | 5 | 1 | 0.20 | 3 | 2 | 1 | 0.33 | 0.25 | 100 | 100 | 0.23 |
| latency_rise | client_latency | awsrcf-nab | 5 | 0 | 0.00 | 2 | 2 | 1 | 0.00 | 0.00 | n/a | n/a | 0.23 |
| latency_rise | client_latency | awsrcf | 5 | 3 | 0.60 | 6 | 3 | 1 | 0.50 | 0.55 | 85 | 97 | 0.34 |
| latency_rise | client_latency | awsrcf-trcf | 5 | 2 | 0.40 | 4 | 2 | 0 | 0.50 | 0.44 | 72 | 82 | 0.23 |
| output_length_drop | output_tokens | fixed | 5 | 5 | 1.00 | 6 | 1 | 1 | 0.83 | 0.91 | 55 | 58 | 0.11 |
| output_length_drop | output_tokens | seasonal | 5 | 5 | 1.00 | 7 | 2 | 2 | 0.71 | 0.83 | 45 | 53 | 0.23 |
| output_length_drop | output_tokens | rcf | 5 | 1 | 0.20 | 1 | 0 | 0 | 1.00 | 0.33 | 25 | 25 | 0.00 |
| output_length_drop | output_tokens | awsrcf-nab | 5 | 1 | 0.20 | 1 | 0 | 0 | 1.00 | 0.33 | 30 | 30 | 0.00 |
| output_length_drop | output_tokens | awsrcf | 5 | 1 | 0.20 | 7 | 6 | 0 | 0.14 | 0.17 | 25 | 25 | 0.68 |
| output_length_drop | output_tokens | awsrcf-trcf | 5 | 1 | 0.20 | 3 | 2 | 0 | 0.33 | 0.25 | 25 | 25 | 0.23 |
| request_drop | request_rate | fixed | 5 | 5 | 1.00 | 5 | 0 | 0 | 1.00 | 1.00 | 80 | 92 | 0.00 |
| request_drop | request_rate | seasonal | 5 | 5 | 1.00 | 5 | 0 | 0 | 1.00 | 1.00 | 80 | 143 | 0.00 |
| request_drop | request_rate | rcf | 5 | 1 | 0.20 | 1 | 0 | 0 | 1.00 | 0.33 | 65 | 65 | 0.00 |
| request_drop | request_rate | awsrcf-nab | 5 | 0 | 0.00 | 0 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| request_drop | request_rate | awsrcf | 5 | 3 | 0.60 | 7 | 4 | 1 | 0.43 | 0.50 | 55 | 71 | 0.45 |
| request_drop | request_rate | awsrcf-trcf | 5 | 3 | 0.60 | 6 | 3 | 1 | 0.50 | 0.55 | 55 | 71 | 0.34 |
| throughput_drop | decode_speed | fixed | 5 | 2 | 0.40 | 4 | 2 | 1 | 0.50 | 0.44 | 30 | 30 | 0.23 |
| throughput_drop | decode_speed | seasonal | 5 | 2 | 0.40 | 9 | 7 | 3 | 0.22 | 0.29 | 112 | 202 | 0.80 |
| throughput_drop | decode_speed | rcf | 5 | 0 | 0.00 | 1 | 1 | 0 | 0.00 | 0.00 | n/a | n/a | 0.11 |
| throughput_drop | decode_speed | awsrcf-nab | 5 | 0 | 0.00 | 1 | 1 | 0 | 0.00 | 0.00 | n/a | n/a | 0.11 |
| throughput_drop | decode_speed | awsrcf | 5 | 0 | 0.00 | 4 | 4 | 1 | 0.00 | 0.00 | n/a | n/a | 0.45 |
| throughput_drop | decode_speed | awsrcf-trcf | 5 | 0 | 0.00 | 3 | 3 | 1 | 0.00 | 0.00 | n/a | n/a | 0.34 |

### 2차 실행, 지연 증가만 (주입 강도 높임)

| scenario | signal | model | events | detected | recall | alerts | false alarms | of which during other injections | precision | F1 | delay median (s) | delay p90 (s) | false alarms per lab day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latency_rise | client_latency | fixed | 5 | 0 | 0.00 | 0 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| latency_rise | client_latency | seasonal | 5 | 0 | 0.00 | 0 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| latency_rise | client_latency | rcf | 5 | 0 | 0.00 | 0 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| latency_rise | client_latency | awsrcf-nab | 5 | 0 | 0.00 | 0 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| latency_rise | client_latency | awsrcf | 5 | 0 | 0.00 | 1 | 1 | 0 | 0.00 | 0.00 | n/a | n/a | 0.41 |
| latency_rise | client_latency | awsrcf-trcf | 5 | 0 | 0.00 | 0 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |

2차 실행의 신호 4개 합산 표(이벤트가 없는 세 신호의 오탐 포함):

| model | events | detected | recall | alerts | false alarms | precision | F1 | delay median (s) | false alarms per signal per lab day |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 5 | 0 | 0.00 | 0 | 0 | n/a | 0.00 | n/a | 0.00 |
| seasonal | 5 | 0 | 0.00 | 0 | 0 | n/a | 0.00 | n/a | 0.00 |
| rcf | 5 | 0 | 0.00 | 0 | 0 | n/a | 0.00 | n/a | 0.00 |
| awsrcf-nab | 5 | 0 | 0.00 | 0 | 0 | n/a | 0.00 | n/a | 0.00 |
| awsrcf | 5 | 0 | 0.00 | 8 | 8 | 0.00 | 0.00 | n/a | 0.82 |
| awsrcf-trcf | 5 | 0 | 0.00 | 4 | 4 | 0.00 | 0.00 | n/a | 0.41 |

### 이벤트별 최고 점수, 새 정상 학습 여부 확인 (1차 실행)

이벤트를 시간 순으로 놓고 각 이벤트 구간(끝 5샘플 뒤까지)의 최고 점수를 적음. 마지막 열은 어떤 이벤트에도 속하지 않는 평가 구간 점수의 99분위수. `scripts/inject_event_scores.py` 출력

| signal | model | event 1 | event 2 | event 3 | event 4 | event 5 | quiet p99 |
|---|---|---|---|---|---|---|---|
| request_rate | rcf | 1.00 | 0.11 | 0.06 | 0.03 | 0.05 | 0.10 |
| request_rate | awsrcf-nab | 0.45 | 0.25 | 0.23 | 0.22 | 0.21 | 0.15 |
| request_rate | awsrcf | 0.99 | 0.67 | 0.97 | 0.33 | 0.50 | 0.62 |
| request_rate | awsrcf-trcf | 0.99 | 0.67 | 0.97 | 0.33 | 0.50 | 0.58 |
| output_tokens | rcf | 1.00 | 0.07 | 0.02 | 0.02 | 0.02 | 0.04 |
| output_tokens | awsrcf-nab | 0.60 | 0.28 | 0.24 | 0.23 | 0.22 | 0.15 |
| output_tokens | awsrcf | 1.00 | 0.36 | 0.31 | 0.21 | 0.25 | 0.60 |
| output_tokens | awsrcf-trcf | 1.00 | 0.38 | 0.31 | 0.22 | 0.25 | 0.44 |
| decode_speed | rcf | 0.12 | 0.03 | 0.08 | 0.02 | 0.18 | 0.12 |
| decode_speed | awsrcf-nab | 0.44 | 0.21 | 0.17 | 0.23 | 0.29 | 0.36 |
| decode_speed | awsrcf | 0.91 | 0.13 | 0.32 | 0.12 | 0.19 | 0.95 |
| decode_speed | awsrcf-trcf | 0.91 | 0.13 | 0.32 | 0.12 | 0.17 | 0.95 |
| client_latency | rcf | 0.40 | 1.00 | 0.02 | 0.02 | 0.02 | 0.09 |
| client_latency | awsrcf-nab | 0.28 | 0.47 | 0.15 | 0.09 | 0.49 | 0.39 |
| client_latency | awsrcf | 0.99 | 0.99 | 0.00 | 0.05 | 0.74 | 0.80 |
| client_latency | awsrcf-trcf | 0.99 | 0.99 | 0.05 | 0.05 | 0.74 | 0.80 |

### 해석

- 합산 탐지율: awsrcf 0.35, awsrcf-trcf 0.30, awsrcf-nab 0.05. 기존 모델은 fixed 0.60, seasonal 0.65, rcf 0.15. AWS RCF 어느 설정도 fixed와 seasonal에 못 미침. rrcf보다는 awsrcf와 awsrcf-trcf가 높고 awsrcf-nab은 낮음
- 오탐: awsrcf가 하루당 신호당 0.48건으로 rrcf(0.09)의 5배쯤이고 seasonal(0.60)에 가까움. awsrcf-trcf는 0.28. awsrcf-nab은 0.09로 rrcf와 같지만 탐지가 1건뿐이라 알림 자체가 적은 결과
- F1: fixed 0.48, seasonal 0.48, awsrcf-trcf 0.33, awsrcf 0.32, rrcf 0.23, awsrcf-nab 0.08
- awsrcf-nab이 조용한 이유: 점수를 8.0으로 나눈 값이라 이 데이터에서 이벤트 최고 점수가 0.2에서 0.6 범위이고 알림 임계 0.5에 거의 닿지 않음. NAB 평가에서 이 설정의 최적 임계값은 0.43이었으나 임계값을 이 데이터에 맞춰 바꾸지는 않음
- 시나리오별
  - request_drop(신호가 0.03배로 급락): fixed, seasonal 5건 중 5건. awsrcf, awsrcf-trcf 3건, rrcf 1건, awsrcf-nab 0건
  - output_length_drop(0.02배): fixed, seasonal 5건. awsrcf 계열 세 설정과 rrcf는 모두 1건
  - throughput_drop(0.38배에서 0.96배, 건마다 들쭉날쭉): fixed, seasonal 2건. AWS RCF 세 설정과 rrcf는 0건
  - latency_rise(1차, 신호가 거의 안 움직임): awsrcf 3건, awsrcf-trcf 2건, seasonal과 rrcf 1건, fixed와 awsrcf-nab 0건. 오탐이 같이 많아 우연 가능성을 배제하지 못함(awsrcf는 알림 6건 중 3건이 오탐)
- 2차 실행(지연 5건): 모든 모델이 0건 탐지로 기존 문서와 같은 결론. awsrcf는 알림 8건이 모두 오탐(신호 4개 합산, 하루당 신호당 0.82건), awsrcf-trcf는 4건. 지연 신호만 보면 awsrcf가 1건
- 새 정상 학습 문제: rrcf에서 관찰된 형태(첫 이벤트만 높고 이후 낮음)가 AWS RCF에서도 일부 보임. 원인이 장애가 새 정상으로 학습되는 것인지는 확인하지 못함
  - rrcf: request_rate 최고 점수가 1.00, 0.11, 0.06, 0.03, 0.05. 이후 이벤트가 평소 구간 99분위수(0.10) 아래
  - awsrcf-nab: request_rate 0.45, 0.25, 0.23, 0.22, 0.21. 첫 이벤트의 절반 수준으로 내려가지만 평소 99분위수(0.15)보다는 높게 유지. output_tokens도 0.60에서 0.28, 0.24, 0.23, 0.22로 같은 형태
  - awsrcf: output_tokens는 1.00 뒤에 0.36, 0.31, 0.21, 0.25로 평소 99분위수(0.60) 아래로 떨어져 rrcf와 비슷한 형태. request_rate는 0.99, 0.67, 0.97, 0.33, 0.50으로 오르내림. 평소 99분위수가 0.62라 이후 이벤트가 평소 변동과 구분되지 않음
  - decode_speed와 client_latency는 이벤트 건마다 신호 크기가 달라 이 형태 판단에 쓰지 않음
  - 이벤트가 10분 간격(압축 일 주기의 0.4일)이고 표본 256개가 약 21분이라 학습 창에 직전 이벤트가 들어 있을 가능성이 있으나 창 크기를 바꿔 확인하지 않음
- NAB에서 awsrcf(31.0)가 rrcf(21.7)보다 높았던 순서는 이 데이터의 합산 F1(0.32 대 0.23)에서도 같은 방향. 다만 이벤트가 20건이고 탐지율이 낮아 순위를 가리지 않음

## QA 환경 실데이터 오탐 시험

기존 재생 평가(`docs/results/m2-real-replay.md`)와 같은 데이터, 같은 도구(`anomaly-detection-replay`)로 AWS RCF 세 설정을 추가해 돌림. 데이터는 저장소와 맥에 없고 수치만 기록함

- 데이터: LLM 서빙 클러스터 테스트 환경 8일치 Prometheus 범위 질의 34개, 시계열 203개, 60초 간격. 이상 라벨이 없어 모든 알림은 오탐 후보. 실사용 트래픽이 아니라 오탐 내성의 하한을 보는 용도
- 절차: 앞 2일 워밍업, 방향 both, 알림 규칙은 점수 0.5 이상 3샘플 연속, 보호 설정 끔과 켬(`min_expected` 0.05, 허용 오차 0.05, 허용 비율 0.05). 기준선은 시계열당 하루 0.1건
- 모델: 기존 `fixed`, `seasonal`, `rcf`는 이전 실행의 점수 캐시를 그대로 써서 수치가 이전 문서와 같음(0.010, 3.463과 1.361, 0.173과 0.110). AWS RCF 세 설정은 이번에 새로 돌림. 시드는 시계열 이름의 해시
- 실행: 8 프로세스 병렬, 프로세스마다 JVM 하나(힙 1GB), 재생 소요 110초

| model | guards | series | scored | failed | alerts | per series per day | median | p90 | max | series over 0.1/day | gap_resume | idle_to_burst | burst_end | other_up | other_down | no_history |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | off | 203 | 84 | 0 | 5 | 0.010 | 0.000 | 0.000 | 0.17 | 5 | 0 | 0 | 0 | 5 | 0 | 0 |
| fixed | on | 203 | 84 | 0 | 5 | 0.010 | 0.000 | 0.000 | 0.17 | 5 | 0 | 0 | 0 | 5 | 0 | 0 |
| seasonal | off | 203 | 132 | 0 | 2620 | 3.463 | 0.000 | 9.924 | 25.27 | 65 | 169 | 559 | 20 | 1738 | 118 | 16 |
| seasonal | on | 203 | 132 | 0 | 1030 | 1.361 | 0.000 | 4.485 | 18.58 | 61 | 164 | 295 | 4 | 499 | 55 | 13 |
| rcf | off | 203 | 102 | 0 | 105 | 0.173 | 0.000 | 0.502 | 1.00 | 46 | 7 | 19 | 1 | 62 | 14 | 2 |
| rcf | on | 203 | 102 | 0 | 67 | 0.110 | 0.000 | 0.335 | 0.84 | 39 | 4 | 15 | 0 | 44 | 4 | 0 |
| awsrcf-nab | off | 203 | 173 | 0 | 28 | 0.033 | 0.000 | 0.000 | 0.67 | 15 | 1 | 3 | 0 | 24 | 0 | 0 |
| awsrcf-nab | on | 203 | 173 | 0 | 28 | 0.033 | 0.000 | 0.000 | 0.67 | 15 | 1 | 3 | 0 | 24 | 0 | 0 |
| awsrcf | off | 203 | 173 | 0 | 2393 | 2.797 | 0.000 | 7.999 | 15.73 | 81 | 416 | 885 | 6 | 772 | 252 | 62 |
| awsrcf | on | 203 | 173 | 0 | 893 | 1.044 | 0.000 | 2.845 | 8.70 | 77 | 204 | 243 | 4 | 349 | 60 | 33 |
| awsrcf-trcf | off | 203 | 173 | 0 | 2068 | 2.417 | 0.000 | 7.331 | 14.56 | 78 | 405 | 645 | 6 | 755 | 205 | 52 |
| awsrcf-trcf | on | 203 | 173 | 0 | 1082 | 1.264 | 0.000 | 3.481 | 10.38 | 75 | 286 | 211 | 5 | 427 | 118 | 35 |

### 해석

- 기준선 0.1/day를 지키는 모델은 fixed(0.010)와 awsrcf-nab(0.033). 나머지는 모두 초과
- awsrcf-nab: 보호 설정 끔과 켬이 같음(알림 28건). 점수가 낮게 나오는 설정이라 보호 설정이 걸러낼 알림이 없음. 시계열 15개가 하루 0.1건을 넘고 최대 0.67. 대부분이 other_up(24건)
- awsrcf(caster): 끔 2.797, 켬 1.044. seasonal(끔 3.463, 켬 1.361)과 같은 크기이고 rrcf(0.173, 0.110)의 10배쯤. 시계열 77개가 하루 0.1건을 넘음(켬). 임계값에서 0.5를 주는 기본 점수 환산이 알림용으로는 너무 민감하다는 NAB 평가 결과와 같은 방향
- awsrcf-trcf: 끔 2.417, 켬 1.264. 보호 설정이 알림을 줄이는 비율은 awsrcf보다 작음(48퍼센트 대 63퍼센트)
- 점수를 낸 시계열 수가 모델마다 다름(fixed 84, seasonal 132, rcf 102, AWS RCF 173). 하루당 평균은 점수를 낸 시계열 안에서만 계산하므로 모델 간 분모가 같지 않음. AWS RCF가 더 많은 시계열에 점수를 내는 이유는 확인하지 않음
- 재생 중 모델이 중단된 시계열(`failed`)은 0건. rrcf에서 있었던 재귀 한도 초과 문제는 AWS RCF에서 나타나지 않음
- 알림 맥락은 기존 문서와 같은 분류(gap_resume, idle_to_burst, burst_end, other_up, other_down, no_history)이며 awsrcf, awsrcf-trcf에서도 other_up과 idle_to_burst가 다수
- 약한 라벨(테스트 실행 흔적)과의 겹침과 알림 시각의 원인은 이번에 확인하지 않음
- AWS RCF 설정에 파일별 규칙이 없고(표본 256, 기본값) 점수 환산도 하나만 시험한 결과. 환산을 바꾸면 달라질 수 있으나 이 데이터에서 고르지 않음

## 한계와 확인하지 못한 것

- 공식 설정 재현은 Kinesis 함수와 Java 4.4.0 코어의 차이, `timeDecay`의 단위 해석, 시드 영향을 가르지 않은 상태
- rrcf를 공식 설정(shingle 1, 트리 100, 표본 256)으로 돌려 구현 영향과 설정 영향을 떼는 측정을 하지 않음
- 점수 환산 상수(0.25, 임계값 기준)는 하나만 시험함. 다른 환산은 `awsrcf-grade` 외에 측정하지 않음
- 밴드 품질(포함 비율, 폭)과 expected의 정확도는 미평가
- 지연과 메모리는 코어 2개의 부하 중인 공유 장비 값이고 실행 A와 B가 두 배 가까이 다름. 시계열 수 확장 수치는 선형 추정
- 단일 JVM에서 시계열을 하나씩 처리함. 병렬 처리와 여러 자식 프로세스는 시험하지 않음
- amd64 실행과 이미지 빌드, 장시간 안정성은 미확인
- NAB는 침묵의 장애 시나리오를 대신하지 못함. 일반 성능 확인용
- 장애 주입 평가는 시나리오당 이벤트 5건, 모델당 시드 1개. 표본 크기 256과 모델 기본값만 시험했고 임계값이나 점수 환산을 이 데이터에 맞춰 바꾸지 않음
- 새 정상 학습 여부는 이벤트별 최고 점수의 형태만 봄. 학습 창 크기나 시간 감쇠를 바꿔 원인을 가르지 않음
- QA 환경 시험에서 AWS RCF가 점수를 낸 시계열 수(173)가 다른 모델과 달라 평균의 분모가 같지 않음

## 재현 명령

```sh
# jar와 JDK는 레포 밖에 둔다. 빌드 스크립트가 awsrcf-bridge.jar를 같은 폴더에 만든다.
scripts/build-awsrcf-bridge.sh $ANOMALY_AWSRCF_HOME
export ANOMALY_AWSRCF_HOME=... ANOMALY_AWSRCF_JAVA=.../bin/java
export ANOMALY_AWSRCF_JAVA_OPTS="-Xmx1g -XX:+UseParallelGC -XX:ParallelGCThreads=2"
uv run anomaly-detection-nab run --nab-dir /path/to/nab --models awsrcf-nab,awsrcf-trcf,awsrcf,awsrcf-grade \
    --scores-dir out/scores --workers 2 --mem-limit-gb 4
uv run python scripts/bench_awsrcf.py --series 500 --rounds 30
python -m anomaly_detection.lab.replay --signals run1/signals.csv --labels run1/labels.json \
    --models fixed,seasonal,rcf,awsrcf-nab,awsrcf,awsrcf-trcf --step 5
python scripts/inject_event_scores.py --signals run1/signals.csv --labels run1/labels.json
uv run anomaly-detection-replay dumps/ --models fixed,seasonal,rcf,awsrcf-nab,awsrcf,awsrcf-trcf \
    --workers 8 --cache-dir cache --out replay.md
```
