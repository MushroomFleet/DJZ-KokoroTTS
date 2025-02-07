NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .KokoroTTS_v1 import KokoroTTS_v1
    NODE_CLASS_MAPPINGS["KokoroTTS_v1"] = KokoroTTS_v1
    NODE_DISPLAY_NAME_MAPPINGS["KokoroTTS_v1"] = "KokoroTTS_v1"
except ImportError:
    print("Unable to import ThinkSeeker. This node will not be available.")


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
