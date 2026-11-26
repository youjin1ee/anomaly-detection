# 2차 마일스톤 NAB 평가 결과

- 실행일: 2026-11-26
- 대상: `feat/m2-nab-eval` 브랜치, 모델은 1차 마일스톤과 같은 코드(seasonal, rcf, rcf-time)와 비교용 고정 한계(fixed)
- 데이터: NAB 레포 커밋 `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`(2024-12-03), 시계열 58개, 레코드 365558개, 라벨 윈도우 116개
- 실행 환경: ARM64 원격 장비, 6 프로세스 병렬, 프로세스당 주소 공간 상한 4GB. 다른 작업과 CPU를 나눠 써서 부하 평균이 30에서 68(코어 20개)이었음. 실행 시간과 샘플당 비용은 참고용
- 방법 정의: `docs/evaluation.md`의 NAB 절
- 명령: `anomaly-detection-nab run`, `report`, `seeds`, `verify`

## 요약

- seasonal과 fixed는 NAB 점수가 null 탐지기와 같은 0점. 어떤 임계값을 골라도 탐지를 하나라도 내면 null보다 나빠서 최적 임계값이 탐지 없음(1.1)으로 나옴
- rcf와 rcf-time은 3개 시드 평균 standard 20.4점. `for: 3` 규칙을 거친 점수(rcf+for3, rcf-time+for3)는 평균 23.8점과 23.3점
- 공개 점수표에서는 Bayesian Changepoint(17.7), EXPoSE(16.4), Random(11.0)보다 위, Etsy Skyline(35.7)보다 아래
- 이벤트 단위로는 seasonal이 윈도우 116개 중 75개를 잡았지만 오탐 알림이 652건이라 precision 0.19. rcf는 42개를 잡고 오탐 48건으로 precision 0.48, F1 0.41(시드 0)
- 채점 재구현은 NAB 레포의 공식 결과 CSV로 검증. 4개 탐지기 x 3개 프로필 12개 조합에서 공식 임계값 기준 점수 차이가 최대 4.5e-11
- 1차 마일스톤 합성 데이터에서 seasonal은 F1 1.00이었음. 같은 모델이 NAB에서는 F1 0.30. 합성 결과는 깨끗한 신호와 큰 하락에 한정된 결과이고 NAB 같은 거칠고 희소한 실제 데이터의 일반 성능은 아님

## 데이터와 라이선스

- 레포에는 데이터를 넣지 않음. `anomaly-detection-nab fetch --dest DIR`가 위 커밋의 `data/`, `labels/`, `LICENSE.txt`만 내려받음. 이미 받은 NAB 체크아웃이 있으면 `--nab-dir`로 경로만 지정
- NAB 레포 라이선스는 MIT(Copyright 2014-2024 Numenta Inc.). 확인한 근거
  - 레포 루트 `LICENSE.txt`와 `setup.py`의 `license="MIT"`, `.zenodo.json`의 license MIT
  - 마지막 커밋 제목이 `chore: MIT License (#408)`(2024-12-03). GitHub 레포 메타데이터의 SPDX 값도 MIT
- 데이터 파일에 대한 별도 라이선스 문구는 찾지 못함
  - `README.md`와 `data/README.md`에 라이선스 언급이 없고 데이터 디렉터리에 별도 라이선스 파일이 없음
  - 데이터 출처는 AWS CloudWatch, 광고 거래소, 미네소타 교통국, NYC 택시 위원회, 트위터 등으로 `data/README.md`에 적혀 있음. 원 출처별 이용 조건은 확인하지 못함
  - 레포 MIT 라이선스가 데이터 파일까지 포괄하는지는 판단하지 않음. 이 레포에 데이터를 넣지 않는 이유
- 이 결과 문서는 점수와 집계 수치만 담고 NAB 데이터 값은 담지 않음

## 채점 재구현 대조

- 구현: `harness/nab_score.py`. NAB 논문과 공개 문서의 방법을 바탕으로 직접 작성. NAB 코드를 복사하지 않음
- 대조 방법: NAB 레포에 들어 있는 탐지기별 결과 CSV(레코드별 anomaly_score)를 같은 입력으로 넣어 점수를 비교. `anomaly-detection-nab verify`
  - 대상: null, random, numenta, randomCutForest 4개 탐지기. 공식 값은 `results/final_results.json`(정규화 점수)과 `config/thresholds.json`(임계값)
  - null은 0점, numenta standard 70.1011(README 표기 70.5 에서 69.7 범위 안), randomCutForest standard 51.7171(README 표기 51.7)
- 결과
  - 공식 임계값에서 우리 점수와 공식 점수의 차이는 12개 조합 모두 4.54e-11 이하
  - 우리 최적화가 고른 임계값은 공식 `thresholds.json`과 소수점 3째 자리 이하에서 다른 조합이 있고, random standard(17.66 대 16.83), random reward_low_FP(7.48 대 5.76), numenta reward_low_FP(63.21 대 63.12)에서는 우리 쪽 점수가 더 높음
  - 이 차이를 확인하려고 NAB의 `sweeper.py`를 작업 폴더 밖 임시 위치에 복사해 같은 CSV로 두 조합(numenta reward_low_FP, random standard)을 돌림. 공식 코드가 고른 임계값은 0.543099와 0.998405로 우리 값과 같았음. `thresholds.json` 값이 현재 코드의 결과와 다른 이유는 조사하지 않았고, 이전 버전 최적화기의 결과일 가능성만 추정. 임시 복사본은 레포에 넣지 않음
- 대조하지 못한 것
  - NAB의 `run.py` 전체 파이프라인을 실행하지는 않음. 탐지기 실행 단계는 우리 모델로 대체하고 채점과 정규화만 검증
  - 점수표의 나머지 탐지기(ARTime, HTM 등)는 결과 CSV를 받지 않아 재계산하지 않음
  - 카테고리별 점수는 NAB가 정의하지 않은 값. 코퍼스 전체 임계값을 쓰고 카테고리의 null과 perfect로 정규화한 우리 정의이며 공식 대조 대상이 아님

## 평가 조건

- 파라미터 규칙(`docs/evaluation.md`에 상세): 샘플링 간격(첫 100개 타임스탬프의 간격 중앙값)과 워밍업 길이(`min(floor(0.15 * n), 750)`)만으로 정함. 값, 라벨, 결과는 쓰지 않았고 파일별, 카테고리별 조정은 없음
  - seasonal: 풀링 폭 `slot_window`를 5분 데이터 기준 8로 두고 30분과 1시간 데이터는 1로 줄임
  - rcf, rcf-time: `sample_size`를 워밍업 끝에서 숲이 차도록 정함, 정규화 창은 하루 분량(288에서 1440 사이)
  - fixed: 워밍업 구간의 최소와 최대에서 범위의 절반만큼 바깥을 한계로 둠. 1차의 최소 절반, 최대 1.5배 규칙은 0과 음수 값에서 깨져서 바꿈. 이 점수는 한계 안은 0, 밖은 1
- NAB 점수: 임계값은 모델과 프로필마다 코퍼스 전체에서 점수가 최대인 값 하나(공식 `--optimize` 방식). 이 임계값은 평가 데이터로 맞춘 값이라 낙관적. 그래서 고정 임계값 0.5의 점수와 탐지가 있는 임계값 중 최고 점수도 함께 기록
- `+for3` 행: 모델 점수에 알림 규칙의 `for: 3`을 입힌 점수(최근 3개 레코드 점수의 최솟값). NAB 점수는 레코드마다 매겨서 긴 이탈 구간의 모든 레코드가 거짓 양성이 되는데, 알림 규칙은 한 번만 울리는 차이를 보기 위한 변형. 원본 점수와 같은 실행에서 파생
- 이벤트 지표: 점수 0.5 이상이 3개 레코드 연속이면 알림. 윈도우를 참 이벤트로 보고 워밍업 구간에서 끝나는 윈도우는 제외. 탐지 지연은 윈도우 시작부터 잰 값. NAB 윈도우는 라벨된 이상 앞뒤로 넓어서(파일 길이의 약 10퍼센트를 이상 개수로 나눈 폭) 지연에는 이상 시작 전 시간이 들어 있음. 합성 시나리오의 지연과 비교할 수 없음
- 시드: rcf와 rcf-time은 시드 0, 1, 2로 실행. 아래 주 결과표는 시드 0이고 시드별 점수와 평균은 따로 둠. 나머지 모델은 무작위 요소가 없음
- 타임스탬프가 이전보다 늦지 않은 레코드(중복과 역행 몇 건)는 파이프라인이 건너뛰어 탐지 없음으로 처리

## 결과표

수치는 도구 출력 그대로. `+for3`은 위 설명의 변형. artificialNoAnomaly는 윈도우가 없어 NAB 점수를 정의할 수 없음(n/a). 이 카테고리에서는 이벤트 표의 오탐 알림이 4개 모델 모두 0건

### 코퍼스 점수, 모델과 프로필마다 최적 임계값 하나 (시드 0)

| model | standard | threshold | reward_low_FP_rate | threshold | reward_low_FN_rate | threshold |
|---|---|---|---|---|---|---|
| fixed | 0.0 | 1.1000 | 0.0 | 1.1000 | 0.0 | 1.1000 |
| fixed+for3 | 0.0 | 1.1000 | 0.0 | 1.1000 | 0.0 | 1.1000 |
| seasonal | 0.0 | 1.1000 | 0.0 | 1.1000 | 0.0 | 1.1000 |
| seasonal+for3 | 0.0 | 1.1000 | 0.0 | 1.1000 | 0.0 | 1.1000 |
| rcf | 21.7 | 0.6607 | 10.8 | 0.8976 | 27.7 | 0.6607 |
| rcf+for3 | 23.8 | 0.5015 | 15.6 | 0.5960 | 28.7 | 0.3235 |
| rcf-time | 18.9 | 0.8000 | 12.5 | 0.9526 | 26.4 | 0.5176 |
| rcf-time+for3 | 24.0 | 0.3304 | 12.8 | 0.7900 | 30.6 | 0.2834 |

### 탐지가 있는 임계값 중 최고 점수 (시드 0)

| model | standard | threshold | reward_low_FP_rate | threshold | reward_low_FN_rate | threshold |
|---|---|---|---|---|---|---|
| fixed | -57.1 | 1.0000 | -161.5 | 1.0000 | -21.1 | 1.0000 |
| fixed+for3 | -62.8 | 1.0000 | -146.3 | 1.0000 | -34.4 | 1.0000 |
| seasonal | -437.5 | 1.0000 | -917.5 | 1.0000 | -276.7 | 1.0000 |
| seasonal+for3 | -106.8 | 1.0000 | -238.0 | 1.0000 | -62.5 | 1.0000 |
| rcf | 21.7 | 0.6607 | 10.8 | 0.8976 | 27.7 | 0.6607 |
| rcf+for3 | 23.8 | 0.5015 | 15.6 | 0.5960 | 28.7 | 0.3235 |
| rcf-time | 18.9 | 0.8000 | 12.5 | 0.9526 | 26.4 | 0.5176 |
| rcf-time+for3 | 24.0 | 0.3304 | 12.8 | 0.7900 | 30.6 | 0.2834 |

### 고정 임계값 0.5의 점수 (시드 0)

| model | standard | reward_low_FP_rate | reward_low_FN_rate |
|---|---|---|---|
| fixed | -57.1 | -161.5 | -21.1 |
| fixed+for3 | -62.8 | -146.3 | -34.4 |
| seasonal | -905.9 | -1901.9 | -572.9 |
| seasonal+for3 | -269.2 | -602.6 | -157.1 |
| rcf | 16.4 | -6.9 | 25.3 |
| rcf+for3 | 23.8 | 14.8 | 28.0 |
| rcf-time | 16.9 | -6.5 | 25.7 |
| rcf-time+for3 | 18.1 | 8.7 | 22.1 |

### rcf 계열의 시드별 점수

| model | profile | seed 0 | seed 1 | seed 2 | mean | min | max |
|---|---|---|---|---|---|---|---|
| rcf | standard | 21.7 | 19.9 | 19.7 | 20.4 | 19.7 | 21.7 |
| rcf | reward_low_FP_rate | 10.8 | 14.3 | 14.4 | 13.2 | 10.8 | 14.4 |
| rcf | reward_low_FN_rate | 27.7 | 25.6 | 26.9 | 26.7 | 25.6 | 27.7 |
| rcf+for3 | standard | 23.8 | 21.9 | 25.7 | 23.8 | 21.9 | 25.7 |
| rcf+for3 | reward_low_FP_rate | 15.6 | 14.2 | 14.3 | 14.7 | 14.2 | 15.6 |
| rcf+for3 | reward_low_FN_rate | 28.7 | 29.2 | 30.9 | 29.6 | 28.7 | 30.9 |
| rcf-time | standard | 18.9 | 20.0 | 22.3 | 20.4 | 18.9 | 22.3 |
| rcf-time | reward_low_FP_rate | 12.5 | 12.2 | 13.7 | 12.8 | 12.2 | 13.7 |
| rcf-time | reward_low_FN_rate | 26.4 | 26.4 | 26.9 | 26.6 | 26.4 | 26.9 |
| rcf-time+for3 | standard | 24.0 | 21.9 | 24.0 | 23.3 | 21.9 | 24.0 |
| rcf-time+for3 | reward_low_FP_rate | 12.8 | 14.2 | 16.6 | 14.5 | 12.8 | 16.6 |
| rcf-time+for3 | reward_low_FN_rate | 30.6 | 27.4 | 28.6 | 28.9 | 27.4 | 30.6 |

### 공개 점수표와 나란히 (시드 0)

출처: NAB 레포 README의 Scoreboard 절(커밋 `ea702d7`). 공개 탐지기 점수는 각 저자가 공식 도구로 낸 값이고, 이 표는 같은 조건의 통제된 비교가 아님. rcf 행은 rrcf 구현이며 점수표의 Random Cut Forest(AWS 독점 구현)와 다름. 점수표의 Numenta HTM 행은 README가 범위(70.5 에서 69.7)로 적은 값 중 윗값

| detector | standard | reward_low_FP_rate | reward_low_FN_rate | source |
|---|---|---|---|---|
| Perfect | 100.0 | 100.0 | 100.0 | NAB README scoreboard |
| ARTime | 74.9 | 65.1 | 80.4 | NAB README scoreboard |
| Numenta HTM (NuPIC 1.0, range upper end) | 70.5 | 62.6 | 75.2 | NAB README scoreboard |
| CAD OSE | 69.9 | 67.0 | 73.2 | NAB README scoreboard |
| earthgecko Skyline | 58.2 | 46.2 | 63.9 | NAB README scoreboard |
| KNN CAD | 58.0 | 43.4 | 64.8 | NAB README scoreboard |
| Relative Entropy | 54.6 | 47.6 | 58.8 | NAB README scoreboard |
| Random Cut Forest (AWS proprietary implementation) | 51.7 | 38.4 | 59.7 | NAB README scoreboard |
| Twitter ADVec v1.0.0 | 47.1 | 33.6 | 53.5 | NAB README scoreboard |
| Windowed Gaussian | 39.6 | 20.9 | 47.4 | NAB README scoreboard |
| Etsy Skyline | 35.7 | 27.1 | 44.5 | NAB README scoreboard |
| this repo: rcf-time+for3 | 24.0 | 12.8 | 30.6 | this run |
| this repo: rcf+for3 | 23.8 | 15.6 | 28.7 | this run |
| this repo: rcf | 21.7 | 10.8 | 27.7 | this run |
| this repo: rcf-time | 18.9 | 12.5 | 26.4 | this run |
| Bayesian Changepoint | 17.7 | 3.2 | 32.2 | NAB README scoreboard |
| EXPoSE | 16.4 | 3.2 | 26.9 | NAB README scoreboard |
| Random (mean over seeds) | 11.0 | 1.2 | 19.5 | NAB README scoreboard |
| Null | 0.0 | 0.0 | 0.0 | NAB README scoreboard |
| this repo: fixed | 0.0 | 0.0 | 0.0 | this run |
| this repo: fixed+for3 | 0.0 | 0.0 | 0.0 | this run |
| this repo: seasonal | 0.0 | 0.0 | 0.0 | this run |
| this repo: seasonal+for3 | 0.0 | 0.0 | 0.0 | this run |

### 카테고리별 NAB 점수, standard (시드 0, 탐지가 있는 코퍼스 임계값)

| model | artificialNoAnomaly | artificialWithAnomaly | realAdExchange | realAWSCloudwatch | realKnownCause | realTraffic | realTweets |
|---|---|---|---|---|---|---|---|
| fixed | n/a | 15.5 | 53.7 | -285.2 | 27.1 | 44.5 | -1.6 |
| fixed+for3 | n/a | 15.4 | 0.0 | -262.2 | 18.3 | 12.5 | -0.8 |
| seasonal | n/a | -269.6 | 38.1 | -318.0 | -185.5 | 3.1 | -1110.3 |
| seasonal+for3 | n/a | 7.5 | 0.0 | -146.8 | -101.0 | -0.8 | -185.0 |
| rcf | n/a | 35.8 | 20.3 | 17.8 | 34.0 | 36.6 | 11.1 |
| rcf+for3 | n/a | 11.5 | 13.2 | 22.2 | 36.3 | 32.5 | 21.2 |
| rcf-time | n/a | 8.9 | 13.3 | 20.6 | 34.6 | 25.8 | 10.0 |
| rcf-time+for3 | n/a | 22.5 | 32.5 | 18.7 | 28.5 | 39.3 | 16.4 |

### 카테고리별 NAB 점수, reward_low_FP_rate (시드 0)

| model | artificialNoAnomaly | artificialWithAnomaly | realAdExchange | realAWSCloudwatch | realKnownCause | realTraffic | realTweets |
|---|---|---|---|---|---|---|---|
| fixed | n/a | 15.5 | 48.3 | -611.0 | 25.9 | 41.8 | -68.3 |
| fixed+for3 | n/a | 15.4 | 0.0 | -546.0 | 18.3 | 11.7 | -36.2 |
| seasonal | n/a | -572.5 | 31.1 | -676.1 | -389.8 | 0.4 | -2295.3 |
| seasonal+for3 | n/a | -1.4 | 0.0 | -319.7 | -210.9 | -1.6 | -423.6 |
| rcf | n/a | 6.9 | 13.6 | 18.5 | 19.8 | 18.3 | -4.2 |
| rcf+for3 | n/a | 7.6 | 13.2 | 18.5 | 27.6 | 26.4 | 4.1 |
| rcf-time | n/a | 6.8 | 6.7 | 18.7 | 22.4 | 12.5 | 4.8 |
| rcf-time+for3 | n/a | 7.6 | 6.6 | 19.4 | 26.7 | 6.7 | 5.1 |

### 카테고리별 NAB 점수, reward_low_FN_rate (시드 0)

| model | artificialNoAnomaly | artificialWithAnomaly | realAdExchange | realAWSCloudwatch | realKnownCause | realTraffic | realTweets |
|---|---|---|---|---|---|---|---|
| fixed | n/a | 15.9 | 57.3 | -175.7 | 28.6 | 46.3 | 22.2 |
| fixed+for3 | n/a | 15.8 | 0.0 | -167.0 | 19.2 | 13.1 | 11.6 |
| seasonal | n/a | -168.6 | 39.7 | -197.6 | -116.6 | 4.5 | -714.0 |
| seasonal+for3 | n/a | 10.5 | 0.0 | -91.0 | -63.8 | -0.5 | -102.1 |
| rcf | n/a | 40.5 | 20.7 | 21.9 | 40.2 | 38.7 | 22.5 |
| rcf+for3 | n/a | 25.1 | 26.5 | 26.0 | 39.3 | 33.1 | 24.9 |
| rcf-time | n/a | 22.1 | 27.2 | 25.9 | 38.4 | 39.3 | 16.2 |
| rcf-time+for3 | n/a | 25.1 | 40.3 | 28.3 | 32.5 | 39.7 | 24.6 |

### 카테고리별 이벤트 지표 (시드 0, 점수 0.5 이상 3개 연속)

| category | model | windows | detected | recall | alerts | false alarms | precision | F1 | delay median (min) | delay p90 (min) | false alarms per day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| artificialNoAnomaly | fixed | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialNoAnomaly | seasonal | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialNoAnomaly | rcf | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialNoAnomaly | rcf-time | 0 | 0 | n/a | 0 | 0 | n/a | n/a | n/a | n/a | 0.00 |
| artificialWithAnomaly | fixed | 6 | 1 | 0.17 | 1 | 0 | 1.00 | 0.29 | 1050.0 | 1050.0 | 0.00 |
| artificialWithAnomaly | seasonal | 6 | 5 | 0.83 | 17 | 6 | 0.65 | 0.73 | 1030.0 | 1101.0 | 0.08 |
| artificialWithAnomaly | rcf | 6 | 1 | 0.17 | 3 | 2 | 0.33 | 0.22 | 1015.0 | 1015.0 | 0.03 |
| artificialWithAnomaly | rcf-time | 6 | 1 | 0.17 | 4 | 3 | 0.25 | 0.20 | 1015.0 | 1015.0 | 0.04 |
| realAdExchange | fixed | 14 | 0 | 0.00 | 0 | 0 | n/a | 0.00 | n/a | n/a | 0.00 |
| realAdExchange | seasonal | 14 | 3 | 0.21 | 19 | 16 | 0.16 | 0.18 | 2460.0 | 2508.0 | 0.05 |
| realAdExchange | rcf | 14 | 2 | 0.14 | 2 | 0 | 1.00 | 0.25 | 3330.0 | 4602.0 | 0.00 |
| realAdExchange | rcf-time | 14 | 2 | 0.14 | 2 | 0 | 1.00 | 0.25 | 3330.0 | 4602.0 | 0.00 |
| realAWSCloudwatch | fixed | 30 | 6 | 0.20 | 6 | 0 | 1.00 | 0.33 | 765.0 | 1122.5 | 0.00 |
| realAWSCloudwatch | seasonal | 30 | 16 | 0.53 | 179 | 152 | 0.15 | 0.24 | 380.0 | 1022.5 | 0.76 |
| realAWSCloudwatch | rcf | 30 | 9 | 0.30 | 18 | 7 | 0.61 | 0.40 | 510.0 | 987.0 | 0.04 |
| realAWSCloudwatch | rcf-time | 30 | 9 | 0.30 | 20 | 9 | 0.55 | 0.39 | 510.0 | 987.0 | 0.05 |
| realKnownCause | fixed | 19 | 4 | 0.21 | 4 | 0 | 1.00 | 0.35 | 1010.0 | 7953.0 | 0.00 |
| realKnownCause | seasonal | 19 | 14 | 0.74 | 117 | 93 | 0.21 | 0.32 | 1230.0 | 3024.0 | 0.14 |
| realKnownCause | rcf | 19 | 10 | 0.53 | 16 | 6 | 0.62 | 0.57 | 1845.0 | 7719.0 | 0.01 |
| realKnownCause | rcf-time | 19 | 9 | 0.47 | 19 | 9 | 0.53 | 0.50 | 2710.0 | 6912.0 | 0.01 |
| realTraffic | fixed | 14 | 2 | 0.14 | 3 | 1 | 0.67 | 0.24 | 70.0 | 78.0 | 0.01 |
| realTraffic | seasonal | 14 | 8 | 0.57 | 28 | 16 | 0.43 | 0.49 | 442.5 | 2251.9 | 0.10 |
| realTraffic | rcf | 14 | 5 | 0.36 | 6 | 1 | 0.83 | 0.50 | 440.0 | 1353.0 | 0.01 |
| realTraffic | rcf-time | 14 | 3 | 0.21 | 3 | 0 | 1.00 | 0.35 | 440.0 | 700.0 | 0.00 |
| realTweets | fixed | 33 | 12 | 0.36 | 73 | 49 | 0.33 | 0.35 | 847.5 | 1296.5 | 0.09 |
| realTweets | seasonal | 33 | 29 | 0.88 | 447 | 369 | 0.17 | 0.29 | 585.0 | 1038.0 | 0.70 |
| realTweets | rcf | 33 | 15 | 0.45 | 48 | 32 | 0.33 | 0.38 | 995.0 | 1991.0 | 0.06 |
| realTweets | rcf-time | 33 | 11 | 0.33 | 47 | 35 | 0.26 | 0.29 | 995.0 | 1985.0 | 0.07 |
| all | fixed | 116 | 25 | 0.22 | 87 | 50 | 0.43 | 0.29 | 800.0 | 1322.0 | 0.02 |
| all | seasonal | 116 | 75 | 0.65 | 807 | 652 | 0.19 | 0.30 | 585.0 | 1983.0 | 0.32 |
| all | rcf | 116 | 42 | 0.36 | 93 | 48 | 0.48 | 0.41 | 980.0 | 3052.0 | 0.02 |
| all | rcf-time | 116 | 35 | 0.30 | 95 | 56 | 0.41 | 0.35 | 980.0 | 3288.0 | 0.03 |

### 비용

부하가 높은 공유 장비에서 잰 값이라 참고용

| model | microseconds per record (single process) |
|---|---|
| fixed | 6 |
| seasonal | 83 |
| rcf | 14052 |
| rcf-time | 24450 |

### 채점 대조 전체 표

| detector | profile | official score | ours at the official threshold | difference | ours at our best threshold | our threshold | official threshold |
|---|---|---|---|---|---|---|---|
| null | standard | 0.0000 | 0.0000 | +0.00e+00 | 0.0000 | 1.100000 | 0.625000 |
| null | reward_low_FP_rate | 0.0000 | 0.0000 | +0.00e+00 | 0.0000 | 1.100000 | 0.625000 |
| null | reward_low_FN_rate | 0.0000 | 0.0000 | +0.00e+00 | 0.0000 | 1.100000 | 0.625000 |
| random | standard | 16.8318 | 16.8318 | -1.81e-11 | 17.6554 | 0.998405 | 0.998450 |
| random | reward_low_FP_rate | 5.7637 | 5.7637 | +2.38e-11 | 7.4800 | 0.999595 | 0.999512 |
| random | reward_low_FN_rate | 25.8764 | 25.8764 | +4.54e-11 | 27.5749 | 0.998405 | 0.998450 |
| numenta | standard | 70.1011 | 70.1011 | +1.62e-11 | 70.1011 | 0.543099 | 0.542188 |
| numenta | reward_low_FP_rate | 63.1168 | 63.1168 | +1.04e-11 | 63.2092 | 0.543099 | 0.575196 |
| numenta | reward_low_FN_rate | 74.3202 | 74.3202 | +1.08e-11 | 74.3202 | 0.543099 | 0.542188 |
| randomCutForest | standard | 51.7171 | 51.7171 | +1.36e-12 | 51.7171 | 0.454019 | 0.453918 |
| randomCutForest | reward_low_FP_rate | 38.3608 | 38.3608 | +7.50e-12 | 38.3608 | 0.575366 | 0.575196 |
| randomCutForest | reward_low_FN_rate | 59.7488 | 59.7488 | -9.24e-12 | 59.7488 | 0.448061 | 0.448048 |

## 해석

- seasonal이 null 이하인 이유
  - 워밍업 이후 채점 대상 레코드 332842개 중 점수 0.5 이상이 8.7퍼센트, 정확히 1.0이 3.8퍼센트. rcf는 각각 0.25퍼센트와 0퍼센트
  - 그중 윈도우 밖 거짓 양성 레코드가 22154개. NAB는 거짓 양성 레코드마다 감점하므로 긴 이탈 구간 전체가 비용
  - 상위 10개 파일이 그 71퍼센트를 차지. 큰 파일은 `cpu_utilization_asg_misconfiguration`(3116개), `Twitter_volume_CVS`(3048개), `Twitter_volume_PFE`(2877개), `Twitter_volume_UPS`(1445개)
  - CVS와 PFE는 값의 77퍼센트와 56퍼센트가 0인 희소한 계수 데이터. 중앙값과 MAD가 0에 가까우면 사소한 값도 큰 편차로 보일 수 있다는 가설이 있으나 확인하지 않음. `cpu_utilization_asg_misconfiguration`에는 0이 없어서 이 가설로는 설명되지 않음. 이 파일의 원인은 조사하지 않음
  - 점수가 1.0으로 포화한 레코드가 많아 최적화기가 그 사이를 가를 수 없음. 모델을 바꿔 개선하는 일은 이번 범위가 아니어서 하지 않음
- fixed: 윈도우 116개 중 25개를 잡고(recall 0.22) 오탐 알림이 50건. 코퍼스 점수는 어느 임계값에서도 null 이하. 합성 시나리오에서 고정 한계가 놓치던 하락형 이상을 못 잡는 점과 같은 방향이지만 원인을 파일별로 나눠 보지는 않음
- rcf 계열: 점수 0.5 이상 레코드가 0.25퍼센트로 적고 포화한 레코드가 없음. 임계값으로 순위를 가를 수 있는 분포와 양수 점수가 같이 나타남. 이 분포가 점수 차이의 원인인지는 확인하지 않음(같이 관찰한 것뿐)
- `for: 3`을 입히면 seasonal은 크게 개선(-437.5에서 -106.8)되지만 여전히 null 아래. rcf는 평균 20.4에서 23.8, rcf-time은 20.4에서 23.3으로 오름. 시드 범위(rcf 19.7 에서 21.7, rcf+for3 21.9 에서 25.7)가 겹쳐서 개선이 시드 변동보다 크다고 말하기 어려움
- 카테고리별(시드 0, standard): rcf는 realKnownCause 34.0, realTraffic 36.6, realTweets 11.1. artificialWithAnomaly는 rcf 35.8, rcf-time 8.9로 같은 계열 안에서도 크게 갈림. 카테고리당 윈도우가 6개에서 33개라 카테고리별 차이를 일반화하지 않음

## 한계와 확인하지 못한 것

- NAB 코퍼스 임계값은 평가 데이터로 맞춘 값이고 공식 정의도 그러함. 새 데이터에서 같은 점수를 기대할 수 없음
- 점수표의 다른 탐지기 값은 재현하지 않았고 README 표기를 옮김. NAB v1.0 기준이라고 README가 밝힌 값이 섞여 있음
- rcf는 시드 3개만 돌림. 범위가 넓은 항목(reward_low_FP_rate는 10.8 에서 14.4)이 있어 소수점 이하 차이를 해석하지 않음. seasonal, fixed는 무작위 요소가 없음
- 파라미터 규칙은 NAB 해상도(5분, 10분, 30분, 1시간)에 맞춘 한 가지 선택. 다른 규칙으로 결과가 얼마나 달라지는지는 측정하지 않음. 데이터별 튜닝을 하지 않았다는 점은 지켰으나 규칙 자체도 NAB 점수를 본 뒤 바꾸지 않았음
- NAB 데이터 파일의 원 출처별 이용 조건과 레포 MIT 라이선스의 데이터 적용 범위는 확인하지 못함
- 이벤트 지표의 탐지 지연은 윈도우 시작 기준이라 합성 시나리오 결과와 비교할 수 없음
- 실행 중 다른 작업과 CPU를 공유해서 시간과 비용 수치가 흔들림
- NAB는 침묵의 장애 시나리오를 대신하지 못함. 일반 성능 확인용이고 1차 합성 시나리오 결과를 대체하지 않음
