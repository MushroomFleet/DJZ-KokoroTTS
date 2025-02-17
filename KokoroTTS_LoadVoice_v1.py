#!/usr/bin/env python
import os
import numpy as np

class KokoroTTS_LoadVoice_v1:
    def __init__(self):
        self.type = "KokoroTTS_LoadVoice_v1"
        self.output_type = "VOICE_LOAD"
        self.required_extensions = []
        self.category = "Text-to-Speech"
        self.name = "Kokoro TTS Load Voice v1"
        self.description = "Loads a saved blended voice shape from a .kkv file."
    
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically list .kkv files from the customvoices directory
        customvoices_dir = os.path.join(os.path.dirname(__file__), "customvoices")
        kkvs = []
        if os.path.exists(customvoices_dir):
            files = os.listdir(customvoices_dir)
            kkvs = [f for f in files if f.endswith(".kkv")]
        return {
            "required": {
                "voice_file": (kkvs, )
            }
        }
        
    RETURN_TYPES = ("VOICE",)
    RETURN_NAMES = ("voice_shape",)
    FUNCTION = "load_voice"
    
    def load_voice(self, voice_file):
        customvoices_dir = os.path.join(os.path.dirname(__file__), "customvoices")
        file_path = os.path.join(customvoices_dir, voice_file)
        if not os.path.exists(file_path):
            raise ValueError(f"Voice file not found: {file_path}")
        try:
            voice_shape = np.load(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load voice file: {str(e)}")
        print(f"Loaded voice from {file_path}")
        return (voice_shape,)

# Node mapping
NODE_CLASS_MAPPINGS = {
    "KokoroTTS_LoadVoice_v1": KokoroTTS_LoadVoice_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KokoroTTS_LoadVoice_v1": "Kokoro TTS Load Voice v1"
}
