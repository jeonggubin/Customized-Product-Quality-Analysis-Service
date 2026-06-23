#include "DHT11.h"
#include "string.h"
#include "stdio.h"

extern TIM_HandleTypeDef htim1;
extern UART_HandleTypeDef huart2;

static void delay_us(uint16_t us)
{
    __HAL_TIM_SET_COUNTER(&htim1, 0);
    while (__HAL_TIM_GET_COUNTER(&htim1) < us);
}

static void DHT11_Debug(char *msg)
{
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 100);
}

static void DHT11_SetInput(void)
{
    GPIO_InitTypeDef g = {0};

    g.Pin  = DHT11_PIN;
    g.Mode = GPIO_MODE_INPUT;
    g.Pull = GPIO_PULLUP;   // 중요: NOPULL 말고 PULLUP
    HAL_GPIO_Init(DHT11_PORT, &g);
}

static void DHT11_SetOutput(void)
{
	GPIO_InitTypeDef g = {0};

	g.Pin   = DHT11_PIN;
	g.Mode  = GPIO_MODE_OUTPUT_OD;
	g.Pull  = GPIO_PULLUP;
	g.Speed = GPIO_SPEED_FREQ_LOW;

	HAL_GPIO_Init(DHT11_PORT, &g);
}

static uint8_t DHT11_WaitForPin(GPIO_PinState state, uint16_t timeout_us)
{
    __HAL_TIM_SET_COUNTER(&htim1, 0);

    while (HAL_GPIO_ReadPin(DHT11_PORT, DHT11_PIN) == state)
    {
        if (__HAL_TIM_GET_COUNTER(&htim1) > timeout_us)
        {
            return 0;
        }
    }

    return 1;
}

static uint8_t DHT11_ReadByte(uint8_t *out)
{
    uint8_t data = 0;

    for (int i = 0; i < 8; i++)
    {
        // 1비트 시작: LOW 구간이 끝날 때까지 대기
        if (!DHT11_WaitForPin(GPIO_PIN_RESET, 200))
        {
            return 0;
        }

        // HIGH 길이 측정 전 40us 대기
        delay_us(40);

        data <<= 1;

        // 40us 후에도 HIGH면 1, LOW면 0
        if (HAL_GPIO_ReadPin(DHT11_PORT, DHT11_PIN) == GPIO_PIN_SET)
        {
            data |= 1;
        }

        // HIGH 구간 종료 대기
        if (!DHT11_WaitForPin(GPIO_PIN_SET, 200))
        {
            return 0;
        }
    }

    *out = data;
    return 1;
}

void DHT11_Init(void)
{
    DHT11_SetOutput();
    HAL_GPIO_WritePin(DHT11_PORT, DHT11_PIN, GPIO_PIN_SET);
    HAL_Delay(3000);
}

DHT11_Data DHT11_Read(void)
{
    DHT11_Data result = {0};

    uint8_t hum = 0;
    uint8_t hum_d = 0;
    uint8_t temp = 0;
    uint8_t temp_d = 0;
    uint8_t checksum = 0;

    // 시작 신호
    DHT11_SetOutput();
    HAL_GPIO_WritePin(DHT11_PORT, DHT11_PIN, GPIO_PIN_RESET);
    HAL_Delay(30);   // DHT11 시작 신호: 18ms 이상 LOW

    HAL_GPIO_WritePin(DHT11_PORT, DHT11_PIN, GPIO_PIN_SET);

    // 입력 전환
    DHT11_SetInput();
    delay_us(30);

    // 센서 응답 확인
    // DHT11 응답: 약 80us LOW -> 약 80us HIGH

    if (!DHT11_WaitForPin(GPIO_PIN_SET, 200))
    {
        DHT11_Debug("DHT F1: no response LOW\r\n");
        goto end;
    }

    if (!DHT11_WaitForPin(GPIO_PIN_RESET, 200))
    {
        DHT11_Debug("DHT F2: no response HIGH\r\n");
        goto end;
    }

    if (!DHT11_WaitForPin(GPIO_PIN_SET, 200))
    {
        DHT11_Debug("DHT F3: response HIGH stuck\r\n");
        goto end;
    }

    // 40비트 데이터 읽기
    if (!DHT11_ReadByte(&hum))
    {
        DHT11_Debug("DHT F4: hum read fail\r\n");
        goto end;
    }

    if (!DHT11_ReadByte(&hum_d))
    {
        DHT11_Debug("DHT F5: hum_d read fail\r\n");
        goto end;
    }

    if (!DHT11_ReadByte(&temp))
    {
        DHT11_Debug("DHT F6: temp read fail\r\n");
        goto end;
    }

    if (!DHT11_ReadByte(&temp_d))
    {
        DHT11_Debug("DHT F7: temp_d read fail\r\n");
        goto end;
    }

    if (!DHT11_ReadByte(&checksum))
    {
        DHT11_Debug("DHT F8: checksum read fail\r\n");
        goto end;
    }

    // 체크섬 확인
    if (checksum == (uint8_t)(hum + hum_d + temp + temp_d))
    {
        result.humidity = hum;
        result.temperature = temp;
        result.valid = 1;
    }
    else
    {
        char buf[80];
        snprintf(buf, sizeof(buf),
                 "DHT checksum fail H:%d HD:%d T:%d TD:%d C:%d SUM:%d\r\n",
                 hum, hum_d, temp, temp_d, checksum,
                 (uint8_t)(hum + hum_d + temp + temp_d));
        DHT11_Debug(buf);
    }

end:
    DHT11_SetOutput();
    HAL_GPIO_WritePin(DHT11_PORT, DHT11_PIN, GPIO_PIN_SET);

    return result;
}
