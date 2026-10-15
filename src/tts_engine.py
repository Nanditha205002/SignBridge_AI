import os
from gtts import gTTS
import pygame

class AudioSpeaker:
    def __init__(self):
        pygame.mixer.init()

    def speak(self, text: str, lang_code: str = 'en'):
        """Converts text to speech and plays audio."""
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            filename = "temp_output.mp3"
            tts.save(filename)
            
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Audio playback error: {e}")