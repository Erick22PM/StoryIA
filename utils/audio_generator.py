# utils/audio_generator.py

from elevenlabs.client import ElevenLabs
import io

def generar_audio(texto: str, voice_id: str, api_key: str) -> bytes:
    """
    Genera un audio MP3 usando ElevenLabs.
    Recibe la API key directamente desde la interfaz.
    """

    client = ElevenLabs(api_key=api_key)

    audio_stream = client.text_to_speech.convert(
        text=texto,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    buffer = io.BytesIO()
    for chunk in audio_stream:
        buffer.write(chunk)

    return buffer.getvalue()
