import logging
logging.basicConfig()
logging.root.setLevel(logging.ERROR)

from flask import Flask
from flask_restful import Api, Resource, reqparse

import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import vptree

from google.cloud import storage

from random import shuffle

app = Flask(__name__)
app.config['DEBUG'] = False

# load cached ids and vectors
ids_file = open("ids.txt", "r")
ids = np.array(ids_file.read().splitlines())
embeds = np.load('temp.npy')

def compare(p1, p2):
  return np.sqrt(np.sum(np.power(p2['data'] - p1['data'], 2)))

def build_tree (ids, embeds):
  tree_data = []
  for i, id in enumerate(ids.tolist()):
    tree_data.append(dict(
      id = id,
      data = embeds[i]
    ))

  return vptree.VPTree(tree_data, compare)

tree = build_tree(ids, embeds)

def get_similar(tree, ids, embeds, id, limit):
  try:
    id_pos = ids.tolist().index(id)
  except ValueError:
    return 'ID not found', 404

  query = dict(
    id = id,
    data = embeds[id_pos]
  )

  matches = tree.get_n_nearest_neighbors(query, limit)

  return results_list(matches, id)

def results_list (matches, id):
  result_ids = []
  for match in matches:
    r_id = match[1]["id"]
    if r_id != id:
      result_ids.append(r_id)
  
  return result_ids

@app.route('/')
def root():
  return 'Hello', 200

# get similar items, limit 10
@app.route('/similar/<id>', methods=['GET'])
def similar(id):
  result_ids = get_similar(tree, ids, embeds, id, 10)

  out = {
    'ids': result_ids
  }

  return out, 200

# get similar items randomized from top 50, limit 10
@app.route('/similar_random/<id>', methods=['GET'])
def similar_random(id):
  result_ids = get_similar(tree, ids, embeds, id, 50)

  shuffle(result_ids)

  result_ids = result_ids[0:10]

  out = {
    'ids': result_ids
  }

  return out, 200

# get similar items to a new question
@app.route('/update/similar/<id>', methods=['GET'])
def similar_new(id):

  storage_client = storage.Client()
  bucket = storage_client.bucket(os.environ.get('GS_BUCKET'))

  # store ids as text files
  blob = bucket.blob(os.environ.get('GS_FILE_IDS'))
  blob.download_to_filename('new_ids.txt')

  # store embeddings
  blob = bucket.blob(os.environ.get('GS_FILE_NPY'))
  blob.download_to_filename('new_temp.npy')

  # load cached ids and vectors
  new_ids_file = open("new_ids.txt", "r")
  new_ids = np.array(new_ids_file.read().splitlines())
  new_embeds = np.load('new_temp.npy')

  new_tree = build_tree(new_ids, new_embeds)

  result_ids = get_similar(new_tree, new_ids, new_embeds, id, 10)

  out = {
    'ids': result_ids
  }

  # Just to be sure...
  del new_tree
  del new_ids
  del new_ids_file
  del new_embeds
  del storage_client
  del blob
  del bucket

  return out, 200

if __name__ == "__main__":
  # use 0.0.0.0 to use it in container
  app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))