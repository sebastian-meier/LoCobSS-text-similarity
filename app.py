import logging
logging.basicConfig()
logging.root.setLevel(logging.ERROR)

from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from flasgger import Swagger

import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import vptree

from google.cloud import storage

from random import shuffle

from scipy.spatial import distance

import tensorflow_hub as hub
import tensorflow as tf

app = Flask(__name__)
app.config['DEBUG'] = False

swagger = Swagger(app)

# load cached ids and vectors
ids_file = open("ids.txt", "r")
ids = np.array(ids_file.read().splitlines())
embeds = np.load('temp.npy')

def compare(p1, p2):
  return distance.euclidean(p1['data'], p2['data'])

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
    return False

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
  """Default endpoint for testing
    ---
    produces:
      - text/plain
    responses:
      200:
        description: Service is alive
        examples:
          text/plain: Hello
  """
  return 'Hello', 200

# get similar items, limit 10
@app.route('/similar/list', methods=['GET'])
def similar_list():
  """Endpoint for generating list of similar question's ids.
    ---
    parameters:
      - name: ids
        description: comma-separated list of ids(integer)
        type: string
        required: true
      - name: limit
        description: number of ids to return
        type: integer
    responses:
      400:
        description: ids missing
      200:
        description: List of ids
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: integer
        examples: 
          application/json: {'ids': [1, 2]}
  """
  ids_str = request.args.get('ids')

  limit = 10
  if request.args.get('limit'):
    limit = int(request.args.get('limit'))

  if (ids_str == False or ids_str == None or len(ids_str) == 0):
    return 'No ids in request', 400

  id_list = ids_str.split(',')

  results = []
  for id in id_list:
    result_ids = get_similar(tree, ids, embeds, id, limit)
    if result_ids != False:
      for rid in result_ids:
        if rid not in results:
          results.append(rid)

  out = {
    'ids': results
  }

  return out, 200

# get similar items, limit 10
@app.route('/similar/<id>', methods=['GET'])
def similar(id):
  """Endpoint for generating list of similar question's ids.
    ---
    parameters:
      - name: id
        type: integer
        required: true
    responses:
      404:
        description: ID not found
      200:
        description: List of ids
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: integer
        examples: 
          application/json: {'ids': [1, 2]}
  """
  result_ids = get_similar(tree, ids, embeds, id, 10)

  if result_ids == False:
    return 'ID not found', 404

  out = {
    'ids': result_ids
  }

  return out, 200

# get similar items randomized from top 50, limit 10
@app.route('/similar_random/<id>', methods=['GET'])
def similar_random(id):
  """Endpoint for generating list of similar question's ids.
    ---
    parameters:
      - name: id
        type: integer
        required: true
    responses:
      404:
        description: ID not found
      200:
        description: List of ids
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: integer
        examples: 
          application/json: {'ids': [1, 2]}
  """
  result_ids = get_similar(tree, ids, embeds, id, 50)

  if result_ids == False:
    return 'ID not found', 404

  shuffle(result_ids)

  result_ids = result_ids[0:10]

  out = {
    'ids': result_ids
  }

  return out, 200

# get similar items to a new question
@app.route('/update/similar/<id>', methods=['GET'])
def similar_new(id):
  """Endpoint for generating list of similar question's ids for new question.
    ---
    parameters:
      - name: id
        type: integer
        required: true
    responses:
      404:
        description: ID not found
      200:
        description: List of ids
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: integer
        examples: 
          application/json: {'ids': [1, 2]}
  """
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

  if result_ids == False:
    return 'ID not found', 404

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

# get embedding vectors for a string
@app.route('/embed', methods=['POST'])
def embed():
    """Embed new text and retrieve similar ids.
    ---
    parameters:
      - name: text
        type: string
        required: true
    responses:
      400:
        description: no text received
      200:
        description: List of vectors (embed) and similar ids
        schema:
          type: object
          properties:
            vectors:
              type: array
              items:
                type: float
            similar:
              type: array
              items:
                type: integer
        examples: 
          application/json: {'ids': [1, 2]}
  """
  if not request.json['text']:
    return {
      "message": "no text received"
    }, 400

  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
  model = hub.load(module_url)
  model_output = model([request.json['text']])

  return_json = {
    "vectors": np.array(model_output).tolist()
  }

  if 'includeSimilar' in request.json and (request.json['includeSimilar'] == True or request.json['includeSimilar'] == 'true'):
    temp_ids = np.concatenate((ids, np.array(['-1'])))
    temp_embeds = tf.concat([embeds, model_output], 0)
    temp_tree = build_tree(temp_ids, temp_embeds)
    result_ids = get_similar(temp_tree, temp_ids, temp_embeds, '-1', 10)
    if result_ids != False:
      return_json['similar'] = result_ids

  return return_json, 200

if __name__ == "__main__":
  # use 0.0.0.0 to use it in container
  app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))
