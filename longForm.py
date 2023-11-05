import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

# Load your 'interview.mp3' file and convert it to the expected raw audio format
audio_array, sampling_rate = librosa.load("interview.mp3", sr=None)

# The pipeline expects a dictionary with a "raw" key for the audio data and a "sampling_rate" key
result = pipe({"raw": audio_array, "sampling_rate": sampling_rate})
transcript = result["text"]

# Write the transcript to a new file
with open("transcript.txt", "w") as file:
    file.write(transcript)

print("The transcript has been written to transcript.txt")
