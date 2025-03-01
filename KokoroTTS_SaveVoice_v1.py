#!/usr/bin/env python
import os
import numpy as np
import json
import datetime
from kokoro_onnx import Kokoro

class KokoroTTS_SaveVoice_v1:
    def __init__(self):
        self.type = "KokoroTTS_SaveVoice_v1"
        self.output_type = "VOICE_SAVE"
        self.required_extensions = []
        self.category = "Text-to-Speech"
        self.name = "Kokoro TTS Save Voice v1"
        self.description = "Saves blended voice shape to a .kkv file for re-use."
        # Load model paths from models.json
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
                "voice1": ([
                    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", 
                    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                    "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", 
                    "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", 
                    "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", 
                    "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
                ],),
                "voice2": ([
                    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", 
                    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                    "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", 
                    "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", 
                    "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", 
                    "jf_nezumi", "jf_tebukuro", "jm_kumo", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
                ],),
                "blend_modes": (["default", "slerp"], {"default": "default"}),
                "slerp_t": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "save_name": ("STRING", {"default": "custom_voice"})
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_voice"
    
    def save_voice(self, voice1, voice2, blend_modes, slerp_t, save_name):
        print(f"Saving blended voice for voices: {voice1} and {voice2} with blend mode: {blend_modes}")
        
        # Initialize Kokoro and get voice styles
        kokoro = Kokoro(self.model_path, self.voices_path)
        style1 = kokoro.get_voice_style(voice1)
        style2 = kokoro.get_voice_style(voice2)
        
        # Blending logic
        if blend_modes == "default":
            blended_voice = 0.5 * style1 + 0.5 * style2
        elif blend_modes == "slerp":
            # Define spherical linear interpolation (slerp)
            def slerp(v0, v1, t):
                v0_s = np.squeeze(v0, axis=1)
                v1_s = np.squeeze(v1, axis=1)
                norm0 = np.linalg.norm(v0_s, axis=1, keepdims=True)
                norm1 = np.linalg.norm(v1_s, axis=1, keepdims=True)
                norm0[norm0 == 0] = 1
                norm1[norm1 == 0] = 1
                v0_norm = v0_s / norm0
                v1_norm = v1_s / norm1
                dot = np.sum(v0_norm * v1_norm, axis=1, keepdims=True)
                dot = np.clip(dot, -1.0, 1.0)
                omega = np.arccos(dot)
                sin_omega = np.sin(omega)
                factor0 = np.sin((1.0 - t) * omega) / sin_omega
                factor1 = np.sin(t * omega) / sin_omega
                factor0[sin_omega < 1e-10] = 1.0 - t
                factor1[sin_omega < 1e-10] = t
                blended = factor0 * v0_s + factor1 * v1_s
                return blended[:, np.newaxis, :]
            blended_voice = slerp(style1, style2, slerp_t)
        else:
            raise ValueError(f"Unknown blend mode: {blend_modes}")
        
        # Determine the customvoices directory path
        customvoices_dir = os.path.join(os.path.dirname(__file__), "customvoices")
        if not os.path.exists(customvoices_dir):
            os.makedirs(customvoices_dir)
        
        # Ensure file name ends with .kkv
        if not save_name.endswith(".kkv"):
            save_name += ".kkv"
        file_path = os.path.join(customvoices_dir, save_name)
        
        # Append timestamp if file already exists
        if os.path.exists(file_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(save_name)[0]
            file_path = os.path.join(customvoices_dir, f"{base_name}_{timestamp}.kkv")
        
        # Save the blended voice using numpy.save
        np.save(file_path, blended_voice)
        
        # Remove the .npy extension that numpy automatically adds
        if os.path.exists(file_path + '.npy'):
            os.rename(file_path + '.npy', file_path)
            
        print(f"Voice saved to {file_path}")
        return (file_path,)

# Node mapping
NODE_CLASS_MAPPINGS = {
    "KokoroTTS_SaveVoice_v1": KokoroTTS_SaveVoice_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KokoroTTS_SaveVoice_v1": "Kokoro TTS Save Voice v1"
}
