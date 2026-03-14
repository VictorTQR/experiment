import pyttsx3
# 初始化引擎
engine = pyttsx3.init()
# 设置要朗读的文本
engine.say("你好，我是Python文本转语音引擎")
# 开始朗读并等待完成
engine.runAndWait()