from lingua import Language
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OUTPUT_DIR = 'output/'
LANGUAGES = [Language.ENGLISH, Language.SPANISH, Language.CHINESE]
