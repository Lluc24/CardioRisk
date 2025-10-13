from helpers import process_metadata
import logging
import pathlib
from data_cleaning import get_headers
from run import DATASET_DIR, METADATA_FILE

logger = logging.getLogger(__name__)

def test_metadata():
    headers = get_headers(DATASET_DIR)

    # Should not raise any exceptions
    metadata = process_metadata(METADATA_FILE, headers)
    logger.info("Metadata processed successfully.")
    logger.info(metadata)