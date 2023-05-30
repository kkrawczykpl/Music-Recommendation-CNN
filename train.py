from utils.constants import MODEL_DIR_NAME, SPECTOGRAM_IMAGAES_DIR_NAME
from utils.file_handler import create_if_not_exists

create_if_not_exists(SPECTOGRAM_IMAGAES_DIR_NAME)
create_if_not_exists(MODEL_DIR_NAME)

# Load dataset
# Train CNN