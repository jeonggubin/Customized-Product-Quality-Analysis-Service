import socket
import time
from gpiozero import Servo

# ==========================================
# 1. 하드웨어 설정 (GPIO 27번 180도 서보모터)
# ==========================================
MIN_PW = 0.5 / 1000
MAX_PW = 2.5 / 1000
servo1 = Servo(27, min_pulse_width=MIN_PW, max_pulse_width=MAX_PW)

# 시작 위치 0도(최저점) 고정하여 대기
servo1.value = -1.0
print("🤖 하드웨어 정렬: 180도 서보모터 기본 위치(0도) 대기 완료")

# ==========================================
# 2. 와이파이 TCP 소켓 수신 서버 설정 (Server)
# ==========================================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", 8888))
server_socket.listen(1)

print("📡 [Wi-Fi] 소켓 서버가 열렸습니다. 수신 대기 중... (Port: 8888)")
print("비전 마스터 데스크톱 코드를 실행해 주세요...")

client_socket, client_info = server_socket.accept()
print(f"🔗 비전 시스템이 연결되었습니다! 클라이언트 IP: {client_info}")

try:
    while True:
        data = client_socket.recv(1024).decode('utf-8').strip().upper()
        if not data:
            break

        print(f"📥 수신된 데이터: [{data}]")

        target_value = None
        angle_text = ""

        # 💡 [수정 구문] 어떤 종류의 불량이든 들어오면 무조건 150도로 고정 처리
        if data in ["CRACK", "DEFECT", "BLOOM", "CRACK_BLOOM"]:
            target_value = 0.66  # 0도(-1.0) 기준 딱 150도 지점
            angle_text = "150도"

        # 미탐지(NON_DETECT) 또는 기타 예외 데이터 처리
        else:
            print(f"🟢 예외 처리 패킷 [{data}]: 서보모터를 구동하지 않고 0도 자리에 대기합니다.")
            continue

        # 지정된 결함 분류 동작 수행 (target_value가 세팅된 조건만 진입)
        if target_value is not None:
            print("\n-------------------------------------------")
            print(f"🚨 초콜릿 결함 [{data}] 접수! 지정 각도 배출 시퀀스를 시작합니다.")

            # Step 1. 지정된 압축 각도로 회전
            print(f"-> {angle_text} 위치로 격리 이동")
            servo1.value = target_value
            time.sleep(0.5)  # 물리적 회전 도달 시간 대기

            # Step 2. 지정된 각도를 유지하며 7초 동안 머무르기
            for i in range(7, 0, -1):
                print(f"-> [{data}] 분류 중... {angle_text} 위치 유지 ({i}초 대기)")
                time.sleep(1.0)

            # Step 3. 원래 위치(0도 원복)로 복귀
            print("-> 공정 원복: 기본 대기 위치(0도) 복귀")
            servo1.value = -1.0
            time.sleep(0.5)

            print("✨ [준비 완료] 다음 배출 공정 스탠바이")
            print("-------------------------------------------\n")

except KeyboardInterrupt:
    print("\n사용자에 의해 모터 프로그램이 종료됩니다.")

finally:
    # 안전 구동 종료 처리
    servo1.value = -1.0
    time.sleep(0.3)
    servo1.value = None
    client_socket.close()
    server_socket.close()
    print("소켓 서버가 안정적으로 폐쇄되었습니다.")
