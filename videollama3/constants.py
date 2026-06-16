CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Arguments for video + image projector
DEFAULT_IMAGE_PROJ_TOKEN = "<IMAGE_PROJ_FEATURES>"

# LLAVIDAL arguments
DEFAULT_OBJECT_TOKEN = "<object>"
DEFAULT_OBJECT_PATCH_TOKEN = "<obj_patch>"
DEFAULT_OBJECT_START_TOKEN = "<obj_start>"
DEFAULT_OBJECT_END_TOKEN = "<obj_end>"

DEFAULT_SKELETON_TOKEN = "<skeleton>"
DEFAULT_SKELETON_PATCH_TOKEN = "<ske_patch>"
DEFAULT_SKELETON_START_TOKEN = "<ske_start>"
DEFAULT_SKELETON_END_TOKEN = "<ske_end>"

# Cross-view query arguments
DEFAULT_QUERY_TOKEN = "<query>"

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 128
MAX_FRAMES = 768
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"

# Stream arguments
STREAM_START_TOKEN = "<|stream_start|>"
STREAM_END_TOKEN = "<|stream_end|>"
STREAM_MAX_FRAMES = 400

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
}

subimage_token_num=196