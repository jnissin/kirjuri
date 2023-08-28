import jax
import jax.numpy as jnp

from whisper_jax import FlaxWhisperPipline
from jax.experimental.compilation_cache import compilation_cache as cc
from datasets import load_dataset


cc.initialize_cache("./jax_cache")
print(f"JAX devices: {jax.devices()}")

pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)
test_dataset = load_dataset("sanchit-gandhi/whisper-jax-test-files", split="train")

# Load the second sample (5 mins) and get the audio
# This is expected to take long because we are populating the JAX JIT cache
audio_0 = test_dataset[0]["audio"]
audio_0_length_in_mins = len(audio_0["array"]) / audio_0["sampling_rate"] / 60
s_time = time.time()
outputs_0 = pipeline(audio_0, return_timestamps=True)
exec_time = time.time()-s_time
print(f"Transcription #1 took {exec_time} sec(s), {(audio_0_length_in_mins*60)/exec_time} audio (sec)/wall time (sec)")
text_0 = outputs_0["text"]

# Load the second sample (30 mins) and get the audio array
# This should be significantly faster as the JAX JIT cache is already populated
audio_1 = test_dataset[1]["audio"]
audio_1_length_in_mins = len(audio_1["array"]) / audio_1["sampling_rate"] / 60
print(f"Transcribing audio of length {audio_1_length_in_mins} min(s)")
s_time = time.time()
outputs_1 = pipeline(audio_1, return_timestamps=True)
exec_time = time.time()-s_time
print(f"Transcription #2 took {exec_time} sec(s), {(audio_1_length_in_mins*60)/exec_time} audio (sec)/wall time (sec)")
text_1 = outputs_1["text"] 
