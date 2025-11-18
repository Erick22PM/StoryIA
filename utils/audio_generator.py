# utils/audio_generator.py

from elevenlabs.client import ElevenLabs
import io

client_ElevenLabs = ElevenLabs(
    api_key="sk_d2c52cf31b281957594ccd599259f6e8583375a79df93b5c"
)

def generar_audio(texto: str, voice_id: str = "default_voice") -> bytes:
    """
    Genera un audio MP3 a partir de texto usando ElevenLabs y devuelve los bytes del audio.
    NO guarda nada en disco.
    """

    # Petición a Eleven Labs: esto devuelve un generador (streaming)
    audio_stream = client_ElevenLabs.text_to_speech.convert(
        text=texto,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Convertir el generador de chunks en un solo bytes object
    buffer = io.BytesIO()
    for chunk in audio_stream:
        buffer.write(chunk)

    return buffer.getvalue()
