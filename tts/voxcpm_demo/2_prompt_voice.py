from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
  r"C:\Users\18890\.cache\modelscope\hub\models\OpenBMB\VoxCPM2",
  load_denoiser=False,
)

wav = model.generate(
    text="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
print("saved: demo.wav")