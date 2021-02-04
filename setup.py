import os
import logging
logging.basicConfig()
logging.root.setLevel(logging.ERROR)

import numpy as np

from google.cloud import storage

from dotenv import load_dotenv
load_dotenv()

# download up to date questions & ids

storage_client = storage.Client()
bucket = storage_client.bucket(os.environ.get('GS_BUCKET'))

# store question and ids as text files
blob = bucket.blob(os.environ.get('GS_FILE_IDS'))
blob.download_to_filename('ids.txt')

blob = bucket.blob(os.environ.get('GS_FILE_QS'))
blob.download_to_filename('questions.txt')

# store embeddings
blob = bucket.blob(os.environ.get('GS_FILE_NPY'))
blob.download_to_filename('temp.npy')
