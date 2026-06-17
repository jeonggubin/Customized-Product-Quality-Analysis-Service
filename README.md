# Team: sweet-smart-factory
# 맞춤형 제품 품질 분석 서비스

### 🛠 개발 배경
<table align="center">
  <tr>
    <td align="center" valign="top">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EC%9D%B8%EA%B0%84%EC%9D%98%20%EC%9C%A1%EC%95%88%20%EA%B2%80%EC%82%AC%20%EC%98%A4%EB%A5%98%20%EB%B6%84%EC%84%9D.png">
        <img src="https://github.com/user-attachments/assets/f58c0498-0ed4-496a-b1de-fa32ddb04d43" height="280" alt="작업 시간에 따른 검사 정확도 변화">
      </a>
    </td>
    <td align="center" valign="top">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EA%B3%A0%EC%9A%A9%EB%85%B8%EB%8F%99%EB%B6%80%20%EC%A4%91%EC%86%8C%EA%B8%B0%EC%97%85%20%EB%B9%88%EC%9D%BC%EC%9E%90%EB%A6%AC%20%EC%8B%A4%ED%83%9C%EC%A1%B0%EC%82%AC.png">
        <img src="https://github.com/user-attachments/assets/13cfb3db-373f-4b7b-9793-688f73ec4797" height="280" alt="국내 주요 산업별 빈일자리 비중">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>작업 시간에 따른 검사 정확도 변화</b></td>
    <td align="center"><b>국내 주요 산업별 빈일자리 비중</b></td>
  </tr>
</table>

현재 중소 제조업계는 **인력 부족**과 **수작업 품질 검사의 한계**라는 이중고를 겪고 있습니다.
* **수작업 검사의 한계:** 육안 검사는 작업 시간이 늘어날수록 피로도가 누적되어, 60분 경과 시 **검사 정확도는 약 60%로 급감**하고 **오진율은 40%까지 폭증**합니다. *(출처: Applied Ergonomics 연구 논문 '인간의 육안 검사 오류 분석')*
* **심각한 구인난:** 국내 주요 산업 중 중소 제조업의 빈일자리 비중은 58.9%로 가장 압도적이며, 이로 인해 숙련된 검사 인력 확보가 매우 어려운 실정입니다. *(출처: 고용노동부, '중소기업 빈일자리 실태조사')*
* **AI 자동화의 필요성:** 스마트 공장 구축 시 기업들이 가장 필요로 하는 분야 1위가 AI 머신 비전 불량 검출(33.9%)인 만큼, 본 프로젝트는 작업자의 피로도와 관계없이 높은 검사 정확도를 유지하고 공정 효율을 극대화하기 위해 기획되었습니다.*(출처: 중소기업중앙회, ‘스마트 공장 구축 AI 도입 의견조사‘)*

### 💎 기존 서비스 차별점
### 1. 멀티모달 데이터 교차 분석 (Vision + Sensor Fusion)
* **단편적 판정의 한계 극복:** 기존의 외관 검사 시스템은 불량 유무만 판정할 뿐 원인을 알 수 없었습니다. 본 서비스는 비전 AI로 탐지된 불량 시점의 이미지와 하드웨어 센서(로스팅 연기 ADC, 콘칭 점도 등) 데이터를 분석하여 불량의 근본 원인을 역추적합니다.
* **공정 데이터 ML 파이프라인:** 6개 공정의 8개 센서 데이터를 48차원의 Wide Pivot 형태로 전처리하고, Random Forest 및 LightGBM 등 7종의 머신러닝 모델을 자동 비교 분석하여 불량 유형별 영향도를 정밀하게 도출합니다.

### 2. RAG 기반의 현장 맞춤형 자연어 조치 가이드
* **LLM 지능형 질의응답 시스템:** 복잡한 수치 센서 데이터나 로그를 작업자가 해석할 필요 없이, 로컬 LLM(Qwen3:8B)과 고성능 임베딩 모델(BGE-M3) 기반의 RAG 파이프라인을 구축했습니다.
* **직관적인 조치 가이드:** 현장 작업자가 불량 원인이나 대처 방안을 채팅으로 물어보면, 공정 매뉴얼 데이터를 기반으로 "즉각적인 자연어 조치 가이드" 및 요약 보고서를 제공하여 공정 다운타임을 최소화합니다.

### 3. 사용자 맞춤형 식품 공정 커스텀 전환 기능 (High Extensibility) 
* **라벨링 데이터 기반의 도메인 확장:** 초콜릿 공정 시스템에 국한되지 않고, 사용자가 운영하는 **자체 식품 생산 라인의 라벨링된 이미지 데이터셋(ZIP)** 및 센서 데이터를 대시보드에 업로드하여 즉각적인 커스텀 모델 학습 및 서비스 전환이 가능합니다.

### 📝 한 줄 요약
**비전 AI 불량 검출과 ML 센서 분석 기반의 RAG 챗봇 연계 및 사용자 맞춤형 데이터 확장 기능을 갖춘 지능형 스마트 팩토리 플랫폼**

---

## 📅 프로젝트 개요
- **프로젝트 명:** 맞춤형 제품 품질 분석 서비스
- **수행 기간:** 2026.05.27 ~ 2026.06.15
- **주요 기능**
  - **1:** 실시간 비전 AI 불량 검사
  - **2:** 공정 센서 데이터 ML 예측 파이프라인
  - **3:** RAG 기반 AI 챗봇 및 품질 분석 보고서
  - **4:** Streamlit 기반 스마트 공장 통합 웹 대시보드
  - **5:** 안정적인 데이터 적재 및 멀티 라인 확장성 (MariaDB)

---

## 🛠 기술 스택
| 분류 | 기술 Stack |
| :--- | :--- |
| **Languages** | C, Python |
| **Frameworks/ Libraries** | YOLOv5nu, YOLOv8n, YOLO11n, PyTorch, OpenCV, Scikit-learn, Streamlit |
| **Communication** | Serial(UART), Wi-Fi |
| **Database** | MariaDB |
| **Hardware/OS** | STM32F446, Servo_Motor(MG996R), Raspberry Pi 5, Webcam, Linux (Raspberry Pi OS) |
| **RAG / LLM** | Qwen3:8B, BGE-M3, Ollama |

<img src="https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=C&logoColor=white" /> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white" /> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white" /> <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" /> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white" /> <img src="https://img.shields.io/badge/YOLOv5nu-00FFFF?style=flat-square&logo=PyTorch&logoColor=black" /> <img src="https://img.shields.io/badge/YOLOv8n-00FFFF?style=flat-square&logo=PyTorch&logoColor=black" /> <img src="https://img.shields.io/badge/YOLO11n-00FFFF?style=flat-square&logo=PyTorch&logoColor=black" /> <img src="https://img.shields.io/badge/Qwen3--8B-551A8B?style=flat-square&logo=render&logoColor=white" /> <img src="https://img.shields.io/badge/BGE--M3-4B0082?style=flat-square&logo=databricks&logoColor=white" /> <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=Ollama&logoColor=white" /> <img src="https://img.shields.io/badge/Serial(UART)-FF6F61?style=flat-square&logo=diagrams.net&logoColor=white" /> <img src="https://img.shields.io/badge/MariaDB-003545?style=flat-square&logo=MariaDB&logoColor=white" /> <img src="https://img.shields.io/badge/Linux%20(RPi%20OS)-FCC624?style=flat-square&logo=Linux&logoColor=black" />

---

## 🧩 시스템 흐름도
<table align="center" width="100%">
  <tr>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EC%84%9C%EB%B9%84%EC%8A%A4%ED%9D%90%EB%A6%84%EB%8F%84.jpg">
        <img src="https://github.com/user-attachments/assets/c3399c21-8b63-4f24-b0fa-320d1f1c1f31" width="100%" alt="서비스 시스템 흐름도">
      </a>
    </td>
  </tr>
</table>

---

## 📂 디렉토리 구조

```text
├── 💻 streamlit_server/          # 중앙 관리 서버 및 웹 대시보드
│   ├── chocolate.py              # 메인 대시보드 및 비전/공정 AI 파이프라인 연동
│   └── apple.py                  # 확장 상용화 테스트(사과 품질 분석) 파트
│
├── 🍓 raspberry_pi/              # 엣지 디바이스 및 AI 카메라 제어
│   ├── rpi_client.py             # 라즈베리파이 메인 클라이언트 구동 스크립트
│   ├── servo.py                  # 불량 제품 분류 서보모터 제어
|   └── serial_db_logger.py       # 시리얼 수집 데이터 DB 적재 로거
│
├── ⚡ stm32_mcu/                 # 하드웨어 센서 제어
│
└── 📊 docs/                      # 데이터베이스 및 가이드 문서
    └── dbdiagram.io              # MariaDB 관계형 데이터베이스 스키마 설계 파일
```
---

## 🔍 상세 기능 설명
### 1. 실시간 데이터 전처리 및 ML 기반 공정 불량 예측 파이프라인
<table align="center" width="100%">
  <tr>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EB%B6%88%EB%9F%89%EC%98%88%EC%B8%A1%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8.jpg">
        <img src="https://github.com/user-attachments/assets/7d4b764b-d892-4b4c-84dd-112dde4ca802" width="100%" alt="공정 센서 데이터 ML 예측 파이프라인">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding-top: 15px;">
      <b>📊 ML 기반 공정 데이터 분석 및 불량 예측 파이프라인 (Process Data ML Prediction Pipeline)</b>
    </td>
  </tr>
</table>


* **데이터 포맷 변환 (Pivot Processing):** 제품 1개당 6개의 공정(로스팅, 분쇄, 콘칭, 템퍼링, 몰딩, 냉각)으로 나뉘어 들어오는 시계열 Long Format 데이터를 하나의 행(Wide Format)으로 변환하는 자동화 피벗 파이프라인을 구축했습니다.
* **데이터 스케일링 및 불균형 해소:** 센서별 데이터 편차를 제어하기 위해 0~1 범위로 스케일을 통일(Scaling)하고, 정상 데이터 대비 극소수인 불량 데이터를 보강하기 위해 AI 기반 불량 데이터 증강 알고리즘을 적용하여 모델의 편향(Bias)을 방지했습니다.
* **최적 모델 자동 선택:** Random Forest, XGBoost, LightGBM 등 총 7종의 머신러닝 모델을 동시 학습시킨 후, `macro-F1 score`가 가장 높은 최적의 모델을 시스템이 자동 선택하여 추론(Inference)에 활용합니다.

### 2. 비전 AI 알고리즘 (사용자 맞춤형 자동 학습 프로세스)
사용자의 제조 환경에 맞춰 '외관 분류(Classification)'와 '객체 탐지(Detection)' 중 원하는 학습 방식을 선택하면, 이미지 업로드부터 모델별 성능 비교 및 경량화(ONNX)까지의 전 과정이 자동으로 수행되는 맞춤형 AI 모델 생성 파이프라인을 제공합니다.
<table align="center" width="100%">
  <tr>
    <td align="center" colspan="3" width="50%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EB%B9%84%EC%A0%84%EB%AA%A8%EB%8D%B81.jpg">
        <img src="https://github.com/user-attachments/assets/b6791ec4-30ed-48bb-b55a-63b7ac84e716" width="100%" alt="YOLO 기반 객체 탐지 화면">
      </a>
    </td>
    <td align="center" colspan="3" width="50%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EB%B9%84%EC%A0%84%EB%AA%A8%EB%8D%B82.jpg">
        <img src="https://github.com/user-attachments/assets/3e26b5f4-b2ca-457b-86f1-77c6d680af41" width="100%" alt="경량 백본 기반 외관 분류 화면">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="3" style="padding-top: 8px; padding-bottom: 35px;">
      <b>🔍 Detection Mode</b>
    </td>
    <td align="center" colspan="3" style="padding-top: 8px; padding-bottom: 35px;">
      <b>📊 Classification Mode</b>
    </td>
  </tr>

  <tr>
    <td align="center" colspan="2" width="33.3%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/nodefects.png">
        <img src="https://github.com/user-attachments/assets/d54de863-3fc4-4944-bece-728bade27881" width="100%" alt="배경흰색 정상">
      </a>
    </td>
    <td align="center" colspan="2" width="33.3%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/crack.png">
        <img src="https://github.com/user-attachments/assets/162d2bb2-ba5a-4426-abf5-c25bfebc255a" width="100%" alt="크랙흰색">
      </a>
    </td>
    <td align="center" colspan="2" width="33.3%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/crack_bloom.png">
        <img src="https://github.com/user-attachments/assets/b344d64e-8252-4c5e-960d-42c71ad2e3b1" width="100%" alt="크랙불름흰색">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2" style="padding-top: 8px;">
      <b>🟢 정상 제품 (NoDefects)</b>
    </td>
    <td align="center" colspan="2" style="padding-top: 8px;">
      <b>🔴 균열 제품 (Crack)</b>
    </td>
    <td align="center" colspan="2" style="padding-top: 8px;">
      <b>🟡 백화/복합 불량 (Crack_Bloom)</b>
    </td>
  </tr>
</table>

### 3. RAG 기반 지능형 제조 공정 진단 AI 챗봇
<table align="center" width="100%">
  <tr>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/ai%20%EC%B1%97%EB%B4%87.jpg">
        <img src="https://github.com/user-attachments/assets/110ef421-9b9d-42e8-86c7-73d6b90ea3df" width="100%" alt="RAG 기반 AI 챗봇 대시보드 화면">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding-top: 15px;">
      <b>🤖 RAG 기반 지능형 공정 진단 AI 챗봇 (Intelligent Process Diagnosis AI Chatbot)</b>
    </td>
  </tr>
</table>

* **자연어 가이드 가동:** 현장 작업자가 웹 대시보드를 통해 불량 이력이나 조치 방법을 자연어로 질문하면, 공정 매뉴얼 데이터베이스를 기반으로 실시간 대처 가이드를 제공합니다.
* **로컬 LLM 아키텍처:** 데이터 외부 유출 방지 및 보안을 위해 로컬 환경에서 `BGE-M3` 임베딩 모델로 수만 건의 문서를 벡터화한 후, 오프라인 거대 언어 모델인 `Qwen3:8B`를 활용해 맥락을 파악하고 답변을 생성하는 완전 폐쇄형 RAG 파이프라인을 구현했습니다.

### 4. 통합 품질 분석 보고서 및 도메인 상용화 확장성 (사과 품질 분석)
* **공정 센서 영향도 분석:** 비전 검사 결과 불량으로 판정된 제품의 이력을 바탕으로, 어떤 공정의 어떤 센서 변수(예: 로스팅 연기 ADC, 분쇄 온도 등)가 불량 발생에 가장 큰 영향을 미쳤는지 특성 중요도(Feature Importance) 순위를 매겨 시각화 보고서를 발행합니다.
* **크로스 도메인 상용화 검증:** 초콜릿 공정 플랫폼에서 확보한 파이프라인 범용성을 바탕으로, 신선도 및 부패 여부를 추적하는 **'사과 품질 분석 공정'** 시스템으로의 기술 전이(Transfer) 및 상용화 확장성 검증을 마쳤습니다.

### 5. 에지 하드웨어 연동 및 분산 데이터베이스 설계(확장성)
* **공정 라인별 독립 데이터 수집:** Raspberry Pi 5 및 STM32(MCU) 보드를 물리 팩토리 인터페이스와 연동하여 온습도, 조도, 공기질 등 3종의 환경 센서 데이터를 실시간 피딩받습니다.
* **MariaDB 아키텍처 스키마:** 각 공정 라인(Line A-1, Line A-2 등)에서 수집된 대용량 시계열 데이터가 꼬이지 않고 현재 타임스탬프와 매핑되어 무결성을 유지하도록 관계형 데이터베이스 구조를 구축했습니다.
<table align="center" width="100%">
  <tr>
    <td align="center" width="50%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%ED%9A%8C%EB%A1%9C%EB%8F%84.png">
        <img src="https://github.com/user-attachments/assets/2976970a-066d-4674-a5d3-0c6c8642e1e8" width="100%" alt="하드웨어 회로도">
      </a>
    </td>
    <td align="center" width="50%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%ED%95%80%EB%A7%B5.png">
        <img src="https://github.com/user-attachments/assets/e1052976-064c-4929-85be-648b9b3fa6ee" width="100%" alt="하드웨어 핀맵">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding-top: 8px; padding-bottom: 30px;">
      <b>⚙️ 하드웨어 회로도 (Hardware Circuit)</b>
    </td>
    <td align="center" style="padding-top: 8px; padding-bottom: 30px;">
      <b>📌 MCU 핀맵 구성 (Pin Mapping)</b>
    </td>
  </tr>

  <tr>
    <td align="center" width="50%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/db%EC%8A%A4%ED%82%A4%EB%A7%88.png">
        <img src="https://github.com/user-attachments/assets/18f54fc9-be91-43c1-a32e-caf7bf6fe048" width="100%" alt="DB 스키마">
      </a>
    </td>
    <td align="center" width="50%" style="padding: 5px; valign: top;">
      <a href="https://github.com/jeonggubin/Customized-Product-Quality-Analysis-Service/blob/main/images/%EB%9D%BC%EC%9D%B8%EB%B3%84%EB%A7%88%EB%A6%AC%EC%95%84db.png">
        <img src="https://github.com/user-attachments/assets/3362bc59-9e2f-4f76-83ce-1cd049c92764" width="100%" alt="라인별 마리아DB 데이터">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding-top: 8px;">
      <b>🗄️ 데이터베이스 구조 (DB Schema)</b>
    </td>
    <td align="center" style="padding-top: 8px;">
      <b>📈 라인별 수집 데이터 (MariaDB Logs)</b>
    </td>
  </tr>
</table>


---

## 🎬 시연 영상

---

## ⚠️ 보완점 및 향후 과제
* **공정 설비 제어 완전 자동화**
  비전 AI 및 ML의 최종 진단 결과 데이터가 무선 통신을 거쳐 STM32 MCU로 자동 피드백되도록 구현하고, 불량 감지 시 서보모터가 설비를 스스로 제어하는 완전 자동화 공정을 구축할 예정이다.
* **설비 예지 정비 인프라 도입**
  시계열로 누적되는 센서 빅데이터를 딥러닝 기반 모델로 장기 학습하여, 설비의 고장 발생 시점을 미리 예측하고 최적의 정비 일정을 대시보드에 자동 제안하는 기능을 추가할 예정이다.
* **XAI 기반의 불량 원인 분석 시각화**
  설명 가능한 AI(XAI) 기법을 전처리 파이프라인에 도입하여, AI가 판단한 불량 원인의 수학적 근거를 작업자가 직관적으로 파악할 수 있도록 고도화할 예정이다.
* **다중 생산 라인 및 실데이터 확장성 개선**
  현재 라인별로 정립된 MariaDB 스키마 구조를 확장하여 분산된 엣지 노드 중심의 다중 공정 네트워크를 완성할 예정이다. 또한, 현재 검증에 사용한 시뮬레이션용 더미 데이터셋을 넘어 실제 공장 환경의 실물 센서 데이터셋을 투입하더라도 파이프라인이 유연하게 연동되도록 호환성을 고도화할 예정이다.

---

## 💁‍♂️ 팀원

| 이름 | 역할 | 담당 파트 |
|----------|----------|----------|
| 정구빈 | PM/Edge AI/FW | 센서 및 AI 카메라 제어, DB 설계, 공장 환경 하드웨어 제작|
| 최지호 | BE/FE | 센서 데이터 모델 학습, streamlit 개발, RAG 파이프라인 구축 |
| 김은선 | Edge AI/FE | 비전 AI 모델 학습, streamlit 개발, 공정 진단용 프롬프트 |
| 박상호 | Edge AI/FE | 비전 AI 모델 학습, 상용화 단계 개발, 시스템 통합 |
