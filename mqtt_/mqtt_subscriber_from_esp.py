import paho.mqtt.client as mqtt
import json
from datetime import datetime

# ========== 配置 ==========
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "home/sensor/btn"

# ========== 回调函数 ==========
def on_connect(client, userdata, flags, rc):
    print(f"[{datetime.now()}] 连接成功，返回码: {rc}")
    client.subscribe(TOPIC)
    print(f"已订阅主题: {TOPIC}")

def on_message(client, userdata, msg):
        data = msg.payload.decode()
        print(f"[{datetime.now()}] 收到数据: {data}")

def on_disconnect(client, userdata, rc):
    print(f"[{datetime.now()}] 连接断开，返回码: {rc}")

# ========== 主程序 ==========
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

print(f"正在连接 {BROKER}:{PORT}...")
client.connect(BROKER, PORT, 60)

# 阻塞运行，持续接收消息
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n程序已停止")
    client.disconnect()


