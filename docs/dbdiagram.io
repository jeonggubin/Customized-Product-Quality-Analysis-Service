Table products {
  id INT [pk, increment]  //제품 고유 ID(PK, 자동증가)
  name VARCHAR(100) [not null]  //제품명
  category VARCHAR(50)  //제품 분류(식품, 전자 부품 등)
  description TEXT  //제품 상세 설명
  
  //정상 환경 범위 : 이 범위를 벗어나면 결함 판정 기준이 됨
  temp_min FLOAT  //허용최저온도
  temp_max FLOAT  //허용최고온도
  humidity_min FLOAT  //허용 최저 습도
  humidity_max FLOAT  //허용 최고 습도
  lux_min FLOAT  //허용 최저 조도 
  lux_max FLOAT  //허용 최고 조도
  created_at TIMESTAMP [default: `now()`]  //레코드 생성 시간
}

//센서 측정값(IoT 센서가 수집한 환경 데이터를 시계열로 저장)
Table sensor_readings {
  id INT [pk, increment]  //측정값 고유 ID(PK)
  product_id INT [ref: > products.id]  //측정 대상 제품(FK-> PRODUCT)
  temperature FLOAT  //측정 온도
  humidity FLOAT  //측정 습도
  lux FLOAT  //측정 조도
  location VARCHAR(50)  //센서 설치 위치(EX : 라인 A1)
  recorded_at TIMESTAMP [default: `now()`]  //측정 시간
}

//검사 결과 테이블(비전 AI가 이미지 분석한 결과 저장, 어떤 제품을 어떤 센서환경에서 검사했는지 연결)
Table inspections {
  id INT [pk, increment]  //검사 고유 ID
  product_id INT [ref: > products.id]  //검사 대상 제품
  sensor_reading_id INT [ref: > sensor_readings.id]  //검사 시점 센서값
  result ENUM('normal', 'defect') [not null]  //검사 판정 : 정상/결함
  defect_type VARCHAR(100)  //결함 종류(EX, 스크래치, 변형),  result = normal이면 null
  confidence_score FLOAT  //AI 판정 신뢰도 
  image_path VARCHAR(255)  //검사에 사용된 이미지 파일 경로
  inspected_at TIMESTAMP [default: `now()`]
}

//결함 조치 테이블(검사에서 결함이 감지 되었을때 수행할 대응 이력-하나의 검사에 려거 조치가 연결될수 있음)
Table defect_actions {
  id INT [pk, increment]  //조치 고유 ID
  inspection_id INT [ref: > inspections.id]  //대상 검사(맥락을 기록)
  /*
  -robot_arm : 로봇암이 불량품을 물리적으로 분리
  -env_adjust : 온도, 습도, 조도 등 환경 자동 조정
  -alert : 담당자에게 알림 발송
  */
  action_type ENUM('robot_arm', 'env_adjust', 'alert') [not null] 
  /*
  -pending : 조치 대기 중
  -done : 조치 완료
  -failed : 조치 실패
  */
  status ENUM('pending', 'done', 'failed') [default: 'pending']
  actioned_at TIMESTAMP [default: `now()`]  //조치 수행(또는 등록) 시간
}

//챗봇 질의 로그 테이블(검사 결과에 대해 사용자가 질문한 내역 저장 및 RAG 파이프라인의 입출력을 모두 보관)
Table chat_logs {
  id INT [pk, increment]  //고유 로그 ID
  inspection_id INT [ref: > inspections.id]  //질문의 컨텍스트가 된 검사
  question TEXT [not null]  //사용자 질문 원문
  answer TEXT  //AI 생성 답변(생성 전이면 NULL)
  retrieved_docs TEXT  //RAG에서 검색된 참조 문서 목록
  created_at TIMESTAMP [default: `now()`]  //질문 생성 시
}
