#!/usr/bin/env python
import os
import torch
import numpy as np
import math
import json
from kokoro_onnx import Kokoro


class KokoroTTS_v1:
   def __init__(self):
       self.type = "KokoroTTS_v1"
       self.output_type = "AUDIO"
       self.output_dims = 1
       self.compatible_decorators = []
       self.required_extensions = []
       self.category = "Text-to-Speech"
       self.name = "Kokoro TTS Processor"
       self.description = "Converts input text to speech audio using custom TTS synthesis."
       # Hardcoded voices list based on the Gradio example.
       self.voices = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ff_siwis", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"]
      
       # Load model paths from models.json
       config_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "kokoro", "models.json")
       try:
           with open(config_path, "r") as f:
               config = json.load(f)
           self.model_path = config.get("model_path", os.path.join("comfyui", "models", "kokoro", "kokoro-v1.0.onnx"))
           self.voices_path = config.get("voices_path", os.path.join("comfyui", "models", "kokoro", "voices-v1.0.bin"))
       except Exception as e:
           print(f"Could not load models.json: {e}")
           # Fallback default paths
           self.model_path = os.path.join("comfyui", "models", "kokoro", "kokoro-v1.0.onnx")
           self.voices_path = os.path.join("comfyui", "models", "kokoro", "voices-v1.0.bin")


   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "text": ("STRING", {"default": "Hello, world!"}),
               "voice": (["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ff_siwis", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"],),
               "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1})
           }
       }


   RETURN_TYPES = ("AUDIO",)
   RETURN_NAMES = ("audio",)
   FUNCTION = "synthesize"


   def synthesize(self, text, voice, speed):
       print(f"Synthesizing speech for text: {text}")
       print(f"Using voice: {voice} at speed: {speed}")
      
       # Default language is set to "en-us"
       lang = "en-us"
       try:
           kokoro = Kokoro(self.model_path, self.voices_path)
           samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
       except Exception as e:
           raise ValueError(f"TTS synthesis failed: {str(e)}")
      
       # Ensure samples is a numpy array with shape [batch, channels, samples]
       if isinstance(samples, list):
           samples = np.array(samples)
       # If samples is 1-dimensional, reshape to [1, 1, samples]
       if samples.ndim == 1:
           samples = np.expand_dims(np.expand_dims(samples, 0), 0)
       elif samples.ndim == 2:
           # Assume shape [channels, samples], add batch dimension.
           samples = np.expand_dims(samples, 0)
       elif samples.ndim == 3:
           # Assume already [batch, channels, samples]
           pass
       else:
           raise ValueError("Unexpected shape for generated audio samples.")
      
       try:
           audio_tensor = torch.from_numpy(samples).float()
       except Exception as e:
           raise ValueError(f"Failed to convert samples to tensor: {str(e)}")
      
       # Ensure tensor dimensions are [batch, channels, samples]
       if audio_tensor.dim() == 2:
           audio_tensor = audio_tensor.unsqueeze(1)
       elif audio_tensor.dim() == 3 and audio_tensor.shape[1] > audio_tensor.shape[2]:
           audio_tensor = audio_tensor.transpose(1, 2)
      
       result = {
           "waveform": audio_tensor.contiguous().detach(),
           "sample_rate": sample_rate,
           "path": None
       }
       print("TTS synthesis complete.")
       return (result,)


NODE_CLASS_MAPPINGS = {
   "KokoroTTS_v1": KokoroTTS_v1
}


NODE_DISPLAY_NAME_MAPPINGS = {
   "KokoroTTS_v1": "Kokoro TTS v1"
}
