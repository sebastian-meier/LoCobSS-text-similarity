import mysql.connector
import os
import logging
logging.basicConfig()
logging.root.setLevel(logging.ERROR)

import tensorflow_hub as hub
import numpy as np
import hashlib

import gzip
import shutil
import urllib.request
import tarfile

from dotenv import load_dotenv
load_dotenv()

# ----------------------
# Cache the tensorflow model

os.environ["TFHUB_CACHE_DIR"] = './tmp/tfhub'
handle = "https://tfhub.dev/google/universal-sentence-encoder/4"
cache_folder = os.environ["TFHUB_CACHE_DIR"] + "/" + hashlib.sha1(handle.encode("utf8")).hexdigest()
if not os.path.exists(cache_folder):
  os.makedirs(cache_folder)

  cache_file = "4.tar.gz"
  urllib.request.urlretrieve("http://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz", cache_folder + "/" + cache_file)

  with gzip.open(cache_folder + "/" + cache_file, 'rb') as f_in:
    with open(cache_folder + "/" + cache_file.split(".gz")[0], 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)

      with tarfile.open(cache_folder + "/" + cache_file.split(".gz")[0], "r") as tar:
          tarlist = []
          for member in tar.getmembers():
              tarlist.append(member)
          tar.extractall(cache_folder, tarlist)
          tar.close()

      os.remove(cache_folder + "/" + cache_file.split(".gz")[0])
  os.remove(cache_folder + "/" + cache_file)
  logging.info("Cached tensorflow_hub {}".format(handle))

# ----------------------
# Connect to questions-database and save all questions to a text file (questions.txt)

db = mysql.connector.connect(
  host=os.environ.get('MYSQL_SERVER'),
  user=os.environ.get('MYSQL_USER'),
  password=os.environ.get('MYSQL_PASS'),
  database=os.environ.get('MYSQL_DB')
)

cursor = db.cursor()

cursor.execute("SELECT id, question FROM {}".format(os.environ.get('MYSQL_TABLE')))

result_q = []
result_id = []

for r in cursor.fetchall():
  result_id.append(str(r[0]))
  result_q.append(r[1].decode("utf-8"))

with open('./questions.txt', mode='wt', encoding='utf-8') as file:
  file.write('\n'.join(result_q))

with open('./ids.txt', mode='wt', encoding='utf-8') as file:
  file.write('\n'.join(result_id))

logging.info("questions cached")

# ----------------------
# To compare existing questions against one another we pre-embed them
# New question require a new embedding-process and are, therefore, slower

# load sentence encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
def embed(input):
  return model(input)

# load questions
text_file = open("questions.txt", "r")
messages = np.array(text_file.read().splitlines())

# embed questions
message_embeddings = embed(messages)

# save embedments
data = np.array(message_embeddings)
np.save('./pre-embed.npy', data)

logging.info("embeddings cached")