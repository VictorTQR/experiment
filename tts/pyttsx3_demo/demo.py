import pyttsx3
import os
import time
import platform

# ==========================================
# 工具函数：打印分割线
# ==========================================
def print_header(title):
    print(f"\n{'='*20} {title} {'='*20}")

# ==========================================
# 1. 初始化引擎与列出声音
# ==========================================
def demo_list_voices(engine):
    """
    演示如何获取系统所有已安装的声音。
    不同的系统（Win/Mac/Linux）声音差异很大。
    """
    print_header("Demo 1: 获取系统声音列表")
    
    voices = engine.getProperty('voices')
    print(f"系统共检测到 {len(voices)} 个声音引擎。\n")
    
    # 仅展示前5个，避免刷屏
    display_count = min(5, len(voices))
    print(f"示例数据 (前{display_count}个):")
    
    for i in range(display_count):
        voice = voices[i]
        print(f"{i+1}. 名称: {voice.name}")
        print(f"   ID: {voice.id}")
        # 尝试解析语言信息
        langs = voice.languages
        if langs:
            print(f"   语言: {langs}")
        print("-" * 30)
        
    # 尝试自动查找中文声音
    print("正在尝试查找中文声音...")
    found_zh = False
    for voice in voices:
        # 关键字匹配，Windows通常是 'Chinese', Mac可能是 'zh_CN' 或 'Ting-Ting'
        if 'Chinese' in voice.name or 'zh' in voice.id.lower() or 'zh' in str(voice.languages).lower():
            print(f"✅ 找到中文声音候选: {voice.name} (ID: ...{voice.id[-20:]})")
            found_zh = True
    if not found_zh:
        print("❌ 未找到明显的中文声音，请检查系统语音设置或仅使用英文演示。")

# ==========================================
# 2. 基础文本朗读
# ==========================================
def demo_basic_speak(engine, text):
    """
    最基础的朗读功能。
    注意：say() 只是将指令加入队列，runAndWait() 才会真正执行并阻塞直到完成。
    """
    print_header("Demo 2: 基础文本朗读")
    print(f"朗读文本: '{text}'")
    
    engine.say(text)
    engine.runAndWait()
    print("朗读完毕。")

# ==========================================
# 3. 调节语速、音量与音调
# ==========================================
def demo_properties(engine, text):
    """
    演示如何动态调整语速、音量等属性。
    """
    print_header("Demo 3: 属性调节演示")
    
    # 获取当前属性
    current_rate = engine.getProperty('rate')
    current_volume = engine.getProperty('volume')
    
    print(f"当前语速 (Rate): {current_rate} (默认通常为200)")
    print(f"当前音量 (Volume): {current_volume} (范围 0.0 - 1.0)")
    
    # --- 演示语速调节 ---
    print("\n>>> 演示慢速朗读 (Rate=100)...")
    engine.setProperty('rate', 100)
    engine.say("这是慢速朗读的演示。")
    engine.runAndWait()
    
    print(">>> 演示快速朗读 (Rate=300)...")
    engine.setProperty('rate', 300)
    engine.say("这是快速朗读的演示。")
    engine.runAndWait()
    
    # --- 演示音量调节 ---
    print("\n>>> 演示小音量朗读 (Volume=0.3)...")
    engine.setProperty('volume', 0.3)
    engine.say("声音变小了。")
    engine.runAndWait()
    
    print(">>> 恢复大音量朗读 (Volume=1.0)...")
    engine.setProperty('volume', 1.0)
    engine.say("声音恢复了。")
    engine.runAndWait()
    
    # 恢复默认设置
    engine.setProperty('rate', current_rate)
    engine.setProperty('volume', current_volume)

# ==========================================
# 4. 切换声音（男声/女声/不同语言）
# ==========================================
def demo_change_voice(engine, text):
    """
    演示如何切换不同的声音ID。
    """
    print_header("Demo 4: 切换声音")
    
    voices = engine.getProperty('voices')
    
    if len(voices) < 2:
        print("系统仅有一个声音，无法演示切换。")
        return

    # 尝试切换到第二个声音
    print(f"正在切换到第二个声音: {voices[1].name}")
    engine.setProperty('voice', voices[1].id)
    engine.say(f"现在的声音是：{voices[1].name}")
    engine.runAndWait()
    
    # 切换回第一个声音
    print(f"正在切换回第一个声音: {voices[0].name}")
    engine.setProperty('voice', voices[0].id)
    engine.say(f"现在的声音是：{voices[0].name}")
    engine.runAndWait()

# ==========================================
# 5. 保存音频到文件
# ==========================================
def demo_save_to_file(engine, text, output_dir):
    """
    演示将朗读结果保存为音频文件。
    注意：Linux 环境下可能需要安装 espeak 并配置 ffmpeg 才能生成文件。
    """
    print_header("Demo 5: 保存音频文件")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 生成文件名
    filename = os.path.join(output_dir, "pyttsx3_output.mp3")
    print(f"正在生成音频文件 -> {filename}")
    
    # 捕获警告：在某些旧版本或特定系统上，此功能可能不可用
    try:
        # 注册保存事件
        engine.save_to_file(text, filename)
        # 必须调用 runAndWait 来触发保存逻辑
        engine.runAndWait()
        
        # 检查文件是否生成（Windows上通常会直接生成，Mac可能生成 .aiff 或 .wav）
        # 注意：pyttsx3 在 Windows 上默认可能生成空的 MP3，建议监听事件确认
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            if size > 0:
                print(f"✅ 文件保存成功！大小: {size} bytes")
            else:
                print("⚠️ 文件已创建，但大小为0。可能是格式不支持或驱动问题。")
        else:
            print("❌ 文件未生成。请检查系统权限或驱动支持。")
            
    except Exception as e:
        print(f"保存失败 (您的系统或版本可能不支持此功能): {e}")

# ==========================================
# 6. 事件监听（进阶）
# ==========================================
def demo_events(engine, text):
    """
    演示如何监听朗读开始、结束、单词朗读等事件。
    这对于做字幕同步或口型同步非常有用。
    """
    print_header("Demo 6: 事件监听")
    
    def on_start(name):
        print(f"  [事件] 开始朗读: {name}")

    def on_word(name, location, length):
        # location: 当前单词在文本中的起始位置
        # length: 当前单词的长度
        print(f"  [事件] 正在朗读单词 (位置: {location}, 长度: {length})")

    def on_end(name, completed):
        print(f"  [事件] 朗读结束: {name}, 是否完成: {completed}")

    # 注册回调函数
    engine.connect('started-utterance', on_start)
    engine.connect('started-word', on_word)
    engine.connect('finished-utterance', on_end)
    
    print("开始朗读并监听事件...")
    engine.say(text, name='event_demo')
    engine.runAndWait()
    
    # 注意：注册的事件会一直保留，如果不需要可以断开
    # 这里为了演示简单，不做断开处理

# ==========================================
# 主函数
# ==========================================
def main():
    print(f"当前操作系统: {platform.system()}")
    
    try:
        # 初始化引擎
        engine = pyttsx3.init()
    except Exception as e:
        print(f"初始化失败: {e}")
        print("如果是在 Linux 上，请确保安装了 espeak: sudo apt install espeak")
        return

    # 测试文本
    sample_text = "你好，这是一个 pyttsx3 离线语音合成演示。"
    output_dir = "output_audio_pyttsx3"

    # 1. 列出声音
    demo_list_voices(engine)

    # 2. 基础朗读
    demo_basic_speak(engine, sample_text)

    # 3. 属性调节
    demo_properties(engine, sample_text)

    # 4. 切换声音
    demo_change_voice(engine, sample_text)

    # 5. 保存文件 (视系统支持情况而定)
    demo_save_to_file(engine, sample_text, output_dir)

    # 6. 事件监听
    demo_events(engine, sample_text)
    
    print("\n所有演示完成！")

if __name__ == "__main__":
    main()