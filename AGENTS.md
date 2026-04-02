# AGENTS.md

이 문서는 `more` 서비스(모아)의 **AI 서버 개발 규정 + 에이전트 실행 규칙**을 함께 관리하는 살아있는 문서다.
문서 목적은 두 가지다.
- 에이전트가 서비스 목적/도메인 맥락을 잃지 않고 일관되게 작업하도록 한다.
- 프로젝트 진행에 따라 바뀌는 정책/알고리즘/우선순위를 누적 반영한다.

해당 프로젝트의 주요 역할은 다음과 같다.

1. 일반 모델 서빙
   - BERT 계열 분류 모델
   - 임베딩 모델
   - 기타 경량 추론 모델

2. 멀티에이전트 실행
   - LangGraph 기반 workflow
   - 도메인별 agent graph 구성
   - tools / prompts / states / nodes 분리

3. 도메인별 API 제공
   - 각 도메인별 엔드포인트 분리
   - request / response DTO 관리
   - 서비스 계층 중심 로직 구성

---

## 최상위 구조 원칙

- 운영 코드는 모두 `app/` 아래에 둔다.
- 실험, 노트북, 벤치마크는 `lab/` 아래에 둔다.
- 공통 설정, DB, 로깅, 모델 로더는 `app/global/`에 둔다.
- 실제 기능은 `app/domain/<도메인>/` 아래에 둔다.
- 각 도메인은 API / Service / Repository / Entity / Agents 중심으로 구성한다.
- `app/` 하위의 파일은 `lab` 폴더에 의존성을 가져서는 안된다. 반대의 경우는 일부 가능하다.

---
## 패키지 구조 예시
```
ai-server/
├─ app/
│  ├─ main.py                          # FastAPI 앱 실행 엔트리포인트
│  │
│  ├─ global/
│  │  ├─ config/
│  │  │  ├─ settings.py               # 환경변수, 설정값 로딩
│  │  │  └─ constants.py              # 전역 상수
│  │  │
│  │  ├─ logger/
│  │  │  └─ logging.py                # 공통 로깅 설정
│  │  │
│  │  ├─ error/
│  │  │  ├─ exceptions.py             # 공통 예외
│  │  │  └─ handlers.py               # FastAPI 예외 핸들러
│  │  │
│  │  ├─ db/
│  │  │  ├─ database.py               # DB 연결 생성
│  │  │  ├─ session.py                # 세션 관리
│  │  │  ├─ base.py                   # Base 선언
│  │  │  └─ dependency.py             # get_db 같은 DI
│  │  │
│  │  ├─ model/
│  │  │  ├─ registry.py               # 일반 모델 서빙용 레지스트리
│  │  │  ├─ loaders.py                # 모델 로딩
│  │  │  └─ runners.py                # 공통 추론 실행
│  │  │
│  │  ├─ llm/
│  │  │  ├─ client.py                 # LLM 호출 클라이언트
│  │  │  └─ prompt_loader.py          # 프롬프트 로딩 공통부
│  │  │
│  │  └─ utils/
│  │     ├─ ids.py
│  │     ├─ time.py
│  │     └─ helpers.py
│  │
│  ├─ domain/
│  │  ├─ inference/
│  │  │  ├─ api/
│  │  │  │  ├─ controller/
│  │  │  │  │  └─ inference_controller.py
│  │  │  │  └─ dto/
│  │  │  │     ├─ detailDto.py
│  │  │  │     └─ commonDto.py
│  │  │  │
│  │  │  ├─ service/
│  │  │  │  ├─ inference_service.py
│  │  │  │  └─ model_route_service.py
│  │  │  │
│  │  │  ├─ repository/
│  │  │  │  └─ inference_repository.py
│  │  │  │
│  │  │  ├─ entity/
│  │  │  │  └─ inference_entity.py
│  │  │  │
│  │  │  └─ tests/
│  │  │
│  │  ├─ merchant_classification/
│  │  │  ├─ api/
│  │  │  │  ├─ controller/
│  │  │  │  │  └─ merchant_classification_controller.py
│  │  │  │  └─ dto/
│  │  │  │     ├─ detailDto.py
│  │  │  │     └─ commonDto.py
│  │  │  │
│  │  │  ├─ service/
│  │  │  │  ├─ merchant_classification_service.py
│  │  │  │  ├─ category_policy_service.py
│  │  │  │  └─ validation_service.py
│  │  │  │
│  │  │  ├─ repository/
│  │  │  │  └─ merchant_repository.py
│  │  │  │
│  │  │  ├─ entity/
│  │  │  │  └─ merchant_entity.py
│  │  │  │
│  │  │  ├─ agents/
│  │  │  │  ├─ graph.py
│  │  │  │  ├─ nodes/
│  │  │  │  │  ├─ classify_node.py
│  │  │  │  │  ├─ validate_node.py
│  │  │  │  │  ├─ fallback_node.py
│  │  │  │  │  └─ finalize_node.py
│  │  │  │  ├─ tools/
│  │  │  │  │  ├─ search_tool.py
│  │  │  │  │  ├─ rerank_tool.py
│  │  │  │  │  └─ model_tool.py
│  │  │  │  ├─ prompts/
│  │  │  │  │  ├─ system_prompt.py
│  │  │  │  │  ├─ classify_prompt.py
│  │  │  │  │  └─ validate_prompt.py
│  │  │  │  └─ states/
│  │  │  │     ├─ state.py
│  │  │  │     ├─ input_state.py
│  │  │  │     └─ output_state.py
│  │  │  │
│  │  │  └─ tests/
│  │  │
│  │  └─ orchestration/
│  │     ├─ api/
│  │     │  ├─ controller/
│  │     │  │  └─ orchestration_controller.py
│  │     │  └─ dto/
│  │     │     ├─ detailDto.py
│  │     │     └─ commonDto.py
│  │     │
│  │     ├─ service/
│  │     │  ├─ orchestration_service.py
│  │     │  └─ workflow_service.py
│  │     │
│  │     ├─ repository/
│  │     │  └─ orchestration_repository.py
│  │     │
│  │     ├─ entity/
│  │     │  └─ orchestration_entity.py
│  │     │
│  │     ├─ agents/
│  │     │  ├─ graph.py
│  │     │  ├─ nodes/
│  │     │  ├─ tools/
│  │     │  ├─ prompts/
│  │     │  └─ states/
│  │     │
│  │     └─ tests/
│  │
│  └─ tests/
│
├─ lab/
│  ├─ notebooks/
│  ├─ experiments/
│  ├─ benchmarks/
│  ├─ scripts/
│  └─ README.md
│
├─ configs/
│  ├─ local.yaml
│  ├─ dev.yaml
│  ├─ prod.yaml
│  ├─ model_registry.yaml
│  └─ agent_registry.yaml
│
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
└─ .env.example
```
---
## 디렉토리 구조 원칙

### `app/global/`
전역 공통 기능만 둔다.

포함:
- 설정
- 로깅
- 예외 처리
- DB 연결 및 세션
- 공통 모델 로더/레지스트리
- 공통 LLM 클라이언트
- 공통 유틸

여기에 도메인별 비즈니스 로직을 두지 않는다.

---

### `app/domain/<도메인>/api/controller/`
각 도메인의 FastAPI 엔드포인트를 둔다.

역할:
- 라우터 선언
- 요청 수신
- DTO 파싱
- 서비스 호출
- 응답 반환

원칙:
- controller는 얇게 유지한다.
- 복잡한 분기와 비즈니스 로직은 service로 넘긴다.

---

### `app/domain/<도메인>/api/dto/detailDto.py`
해당 도메인 엔드포인트에서 사용하는 request / response `BaseModel`을 둔다.

원칙:
- 엔드포인트별 요청/응답 모델을 명확히 정의한다.
- 외부 API 계약은 DTO로 관리한다.
- DTO 이름은 역할이 드러나게 짓는다.

예:
- PredictRequest
- PredictResponse
- RunAgentRequest
- RunAgentResponse

---

### `app/domain/<도메인>/service/`
도메인별 비즈니스 로직을 둔다.

역할:
- 핵심 처리 흐름
- 모델 호출 조합
- 에이전트 실행 호출
- 검증/후처리
- repository 호출

원칙:
- service는 controller보다 두껍고, repository보다 위에 있다.
- 재사용 가능한 업무 로직은 service로 모은다.

---

### `app/domain/<도메인>/repository/`
DB 저장/조회 및 영속성 관련 처리를 둔다.

역할:
- 결과 저장
- 이력 조회
- 상태 업데이트
- DB 모델 접근

원칙:
- SQL/ORM 로직은 repository에서 처리한다.
- controller나 service에서 직접 DB 세부 구현을 다루지 않는다.

---

### `app/domain/<도메인>/entity/`
도메인 내부 핵심 데이터 구조를 둔다.

포함 가능:
- ORM 엔티티
- 내부 상태 모델
- 핵심 데이터 객체

원칙:
- 단순한 프로젝트라면 과도하게 세분화하지 않는다.
- 꼭 필요한 구조만 유지한다.

---

## 에이전트 구조 원칙

에이전트 구조는 모든 도메인에서 아래 형태를 유지한다.

```text
agents/
├─ graph.py
├─ nodes/
├─ tools/
├─ prompts/
└─ states/

---


## Jira 작업 규칙 (현재 저장소 공통)

- 작업 경로: `c:\Users\SSAFY\more\Jira`
- 조회는 항상 `search_issues.py` 사용
- 조회 절차:
    1. `python search_issues.py [options]`
    2. 출력의 `TEMP_FILE_PATH` 추출
    3. temp markdown 파일 읽기 후 결과 요약
- 프로젝트 키는 `.env`의 `JIRA_PROJECT_KEY`를 우선 사용

---

## 보안 규칙

- `.env`/토큰/자격증명은 출력 금지
- 로그 공유 전 민감정보 마스킹 필수
- 하드코딩 비밀값 커밋 금지

---

## 소스 오브 트루스

본 문서는 아래 노션을 기준으로 유지한다.
- 특화프로젝트 PMS: `https://www.notion.so/30cf987f10a180e4bf25fc2bbdedc932`
- 팀프로젝트현황: `https://www.notion.so/318f987f10a1803fbe2ef88d5baffc16`
- 서비스 개념 정리: `https://www.notion.so/318f987f10a180cd88ecf3f49c3a9e27`
- 명세서: `https://www.notion.so/314f987f10a18035959ef5d414a968ca`

---
## 후속 처리 규칙
모든 작업이 완료된 이후 작업한 내용은 반드시 노션 MCP를 사용하여 `개인이력서/특화프로젝트 이슈정리` 페이지에 작업처리일 날짜를 제목 토글로 하여 하위에 처리 작업에 대한 주제를 소제목으로 하는 토글을 추가로 작성하여 정리한다. 특히 정리 과정에서 AI 관련 작업은 전처리와 후처리를 어떤 방식으로 했는지, 왜 그렇게 했는지가 드러나야한다. 모든 작업은 반드시
- 문제 상황
- 원인
- 다양한 접근방법
- 선택한 해결방법
- 결과
  순으로 정리한다.

--- 