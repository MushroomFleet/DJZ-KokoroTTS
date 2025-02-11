#!/usr/bin/env python
import os
import torch
import numpy as np
import json
from kokoro_onnx import Kokoro
import math

class KokoroTTS_v3:
    def __init__(self):
        self.type = "KokoroTTS_v3"
        self.output_type = "AUDIO"
        self.output_dims = 1
        self.compatible_decorators = []
        self.required_extensions = []
        self.category = "Text-to-Speech"
        self.name = "Kokoro TTS Processor v3"
        self.description = "Converts input text to speech audio using blended voice styles with selectable blend modes, including spherical interpolation (slerp)."
        
        # Voices list copied from v2.
        self.voices = [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", 
            "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
            "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", 
            "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", 
            "hf_beta","hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", 
            "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
        ]
        
        # Load model paths from models.json (same relative structure as v1 and v2)
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "kokoro", "models.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.model_path = config.get("model_path", os.path.join("comfyui", "models", "kokoro", "kokoro-v1.0.onnx"))
            self.voices_path = config.get("voices_path", os.path.join("comfyui", "models", "kokoro", "voices-v1.0.bin"))
        except Exception as e:
            print(f"Could not load models.json: {e}")
            self.model_path = os.path.join("comfyui", "models", "kokoro", "kokoro-v1.0.onnx")
            self.voices_path = os.path.join("comfyui", "models", "kokoro", "voices-v1.0.bin")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello, world!"}),
                "voice1": ([
                    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", 
                    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                    "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", 
                    "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", 
                    "hf_beta","hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", 
                    "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
                ],),
                "voice2": ([
                    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", 
                    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                    "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", 
                    "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", 
                    "hf_beta","hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", 
                    "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
                ],),
                "speed1": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "speed2": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "blend_modes": (["default", "slerp"], {"default": "default"}),
                "slerp_t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1})
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    
    def synthesize(self, text, voice1, voice2, speed1, speed2, blend_modes, slerp_t):
        print(f"Synthesizing blended speech for text: {text}")
        print(f"Using voices: {voice1} and {voice2} with speeds: {speed1} and {speed2}")
        print(f"Blend mode: {blend_modes} with slerp_t: {slerp_t}")
        
        lang = "en-us"
        try:
            kokoro = Kokoro(self.model_path, self.voices_path)
            style1 = kokoro.get_voice_style(voice1)
            style2 = kokoro.get_voice_style(voice2)
            
            if blend_modes == "default":
                blended_voice = (0.5 * style1) + (0.5 * style2)
            elif blend_modes == "slerp":
                # Define spherical linear interpolation (slerp)
                def slerp(v0, v1, t):
                    # Remove the singleton dimension to handle batch processing
                    v0_s = np.squeeze(v0, axis=1)  # shape: (N, D) where N is batch size, D=256
                    v1_s = np.squeeze(v1, axis=1)
                    # Compute norms row-wise
                    norm0 = np.linalg.norm(v0_s, axis=1, keepdims=True)
                    norm1 = np.linalg.norm(v1_s, axis=1, keepdims=True)
                    # Avoid division by zero
                    norm0[norm0 == 0] = 1
                    norm1[norm1 == 0] = 1
                    # Normalize each row
                    v0_norm = v0_s / norm0
                    v1_norm = v1_s / norm1
                    # Compute row-wise dot product
                    dot = np.sum(v0_norm * v1_norm, axis=1, keepdims=True)
                    dot = np.clip(dot, -1.0, 1.0)
                    # Compute the angle omega for each row
                    omega = np.arccos(dot)
                    sin_omega = np.sin(omega)
                    # Compute interpolation factors with safe division
                    factor0 = np.sin((1.0 - t) * omega) / sin_omega
                    factor1 = np.sin(t * omega) / sin_omega
                    # Handle cases where sin_omega is near zero
                    factor0[sin_omega < 1e-10] = 1.0 - t
                    factor1[sin_omega < 1e-10] = t
                    # Perform the slerp interpolation for each row
                    blended = factor0 * v0_s + factor1 * v1_s
                    # Reshape to reintroduce the singleton dimension
                    return blended[:, np.newaxis, :]
                
                blended_voice = slerp(style1, style2, slerp_t)
            else:
                raise ValueError(f"Unknown blend mode: {blend_modes}")
            
            # Compute effective speed as the average of the two speeds.
            effective_speed = (speed1 + speed2) / 2.0
            
            samples, sample_rate = kokoro.create(text, voice=blended_voice, speed=effective_speed, lang=lang)
        except Exception as e:
            raise ValueError(f"TTS synthesis failed: {str(e)}")
        
        # Ensure samples is a numpy array with shape [batch, channels, samples]
        if isinstance(samples, list):
            samples = np.array(samples)
        if samples.ndim == 1:
            samples = np.expand_dims(np.expand_dims(samples, 0), 0)
        elif samples.ndim == 2:
            samples = np.expand_dims(samples, 0)
        elif samples.ndim == 3:
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
        print("Blended TTS synthesis complete using mode:", blend_modes)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "KokoroTTS_v3": KokoroTTS_v3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KokoroTTS_v3": "Kokoro TTS v3 (Custom Blend Modes)"
}
