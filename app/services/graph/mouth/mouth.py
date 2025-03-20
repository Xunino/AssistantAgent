import io
from loguru import logger
from typing import Dict
from pathlib import Path
from langchain_core.runnables import Runnable
from openai import OpenAI

import soundfile as sf
import sounddevice as sd

from app.settings import OpenAISettings
from app.services.graph.config import GraphState

speech_file_path = Path(__file__).parent / "tmp/speech.mp3"


class Mouth(Runnable):
    def __init__(self):
        self.client = OpenAI(api_key=OpenAISettings.api_key)
        self.model = "tts-1"
        self.voice = "alloy"

    def invoke(self, inputs: GraphState) -> Dict:
        """Invoke the mouth module."""
        # Create the speech
        if not inputs.assistant_response:
            return {"audio_path": None}

        response = self.client.audio.speech.create(
            model=self.model, voice=self.voice, input=inputs.assistant_response
        )

        # Process response with error handling
        try:
            buffer = io.BytesIO()
            for chunk in response.iter_bytes(chunk_size=4096):
                buffer.write(chunk)
            buffer.seek(0)

            # with sf.SoundFile(buffer, "r") as sound_file:
            #     data = sound_file.read(dtype="float32")
            #     sd.play(data, sound_file.samplerate)
            #     sd.wait()

            response.stream_to_file(speech_file_path)
        except Exception as err:
            logger.error(f"Error during speech processing: {err}")
            return {"audio_path": None}

        return {"audio_path": speech_file_path}

    async def ainvoke(self, inputs: GraphState):
        # TODO: Implement asynchronous version
        raise NotImplementedError("Method not implemented")
