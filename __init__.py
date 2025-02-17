NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .KokoroTTS_v1 import KokoroTTS_v1
    NODE_CLASS_MAPPINGS["KokoroTTS_v1"] = KokoroTTS_v1
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_v1"] = "Kokoro TTS v1"
except ImportError:
    print("Unable to import KokoroTTS_v1. This node will not be available.")

try:
    from .KokoroTTS_v2 import KokoroTTS_v2
    NODE_CLASS_MAPPINGS["KokoroTTS_v2"] = KokoroTTS_v2
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_v2"] = "Kokoro TTS v2"
except ImportError:
    print("Unable to import KokoroTTS_v2. This node will not be available.")

try:
    from .KokoroTTS_v3 import KokoroTTS_v3
    NODE_CLASS_MAPPINGS["KokoroTTS_v3"] = KokoroTTS_v3
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_v3"] = "Kokoro TTS v3"
except ImportError:
    print("Unable to import KokoroTTS_v3. This node will not be available.")

try:
    from .KokoroTTS_SaveVoice_v1 import KokoroTTS_SaveVoice_v1
    NODE_CLASS_MAPPINGS["KokoroTTS_SaveVoice_v1"] = KokoroTTS_SaveVoice_v1
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_SaveVoice_v1"] = "KokoroTTS Save Voice v1"
except ImportError:
    print("Unable to import KokoroTTS_SaveVoice_v1. This node will not be available.")

try:
    from .KokoroTTS_v4 import KokoroTTS_v4
    NODE_CLASS_MAPPINGS["KokoroTTS_v4"] = KokoroTTS_v4
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_v4"] = "Kokoro TTS v4"
except ImportError:
    print("Unable to import KokoroTTS_v4. This node will not be available.")

try:
    from .KokoroTTS_LoadVoice_v1 import KokoroTTS_LoadVoice_v1
    NODE_CLASS_MAPPINGS["KokoroTTS_LoadVoice_v1"] = KokoroTTS_LoadVoice_v1
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_LoadVoice_v1"] = "KokoroTTS_LoadVoice_v1"
except ImportError:
    print("Unable to import KokoroTTS_LoadVoice_v1. This node will not be available.")


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
