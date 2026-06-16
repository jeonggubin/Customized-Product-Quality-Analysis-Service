import serial
import pymysql
import time

DB_CONFIG = {
    'host':     'localhost',
    'user':     'factory',
    'password': 'factory1234',
    'database': 'inspection_db',
    'charset':  'utf8mb4'
}

PRODUCT_ID  = 1
LOCATION    = '라인A-1번'
SERIAL_PORT = '/dev/serial0'
BAUD_RATE   = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
print(f"[시작] {SERIAL_PORT} 연결됨")

conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()
print("[시작] MariaDB 연결됨")

try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print(f"[오류] 시리얼 읽기 실패: {e}")
            time.sleep(0.1)
            continue

        if not line:
            continue

        print(f"[RAW] {repr(line)}")

        try:
            values = line.split(',')
            if len(values) != 4:
                print(f"[경고] 잘못된 데이터: {line}")
                continue
            lux  = float(values[0])
            air  = float(values[1])
            temp = float(values[2])
            humi = float(values[3])
        except ValueError:
            print(f"[경고] 변환 실패: {line}")
            continue

        print(f"[수신] 조도:{lux} 공기질:{air} 온도:{temp} 습도:{humi}")

        sql = """
            INSERT INTO sensor_readings
                (product_id, temperature, humidity, lux, air_quality, location)
            VALUES
                (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (PRODUCT_ID, temp, humi, lux, air, LOCATION))
        conn.commit()
        print(f"[저장] DB 입력 완료")

except KeyboardInterrupt:
    print("\n[종료] 프로그램 종료")

finally:
    cursor.close()
    conn.close()
    ser.close()
