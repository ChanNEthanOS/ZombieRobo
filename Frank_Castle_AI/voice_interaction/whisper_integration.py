import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

if __name__ == "__main__":
    audio_path = input("Enter the path to your audio file: ")
    print("Transcription:", transcribe_audio(audio_path))
