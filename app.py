import logging
logging.basicConfig()
logging.root.setLevel(logging.ERROR)

from flask import Flask
from flask_restful import Api, Resource, reqparse

# import six
# from google.cloud import translate_v2 as translate
# import html

import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import vptree

import tensorflow_hub as hub

from random import shuffle

app = Flask(__name__)
app.config['DEBUG'] = False

# load cached ids and vectors
ids_file = open("./ids.txt", "r")
ids = np.array(ids_file.read().splitlines())
embeds = np.load('./pre-embed.npy')

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

def get_similar(id, limit):
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

# load model if neccessary
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
def embed(input):
  return model(input)

@app.route('/')
def root():
  return 'Hello', 200

# get similar items, limit 10
@app.route('/similar/<id>', methods=['GET'])
def similar(id):
  result_ids = get_similar(id, 10)

  out = {
    'ids': result_ids
  }

  return out, 200

# get similar items randomized from top 50, limit 10
@app.route('/similar_random/<id>', methods=['GET'])
def similar_random(id):
  result_ids = get_similar(id, 50)

  shuffle(result_ids)

  result_ids = result_ids[0:10]

  out = {
    'ids': result_ids
  }

  return out, 200

# get similar items to a new question
@app.route('/similar_new', methods=['POST'])
def similar_new():

  parser = reqparse.RequestParser()
  parser.add_argument('question')
  args = parser.parse_args()

  new_question = args["question"]

  # load questions
  old_questions = open("questions.txt", "r")
  merged_questions = np.array(old_questions.read().splitlines())
  merged_questions = np.insert(merged_questions, 0, new_question)

  old_ids = open("ids.txt", "r")
  old_ids = np.array(old_ids.read().splitlines())
  
  # embed questions
  new_embeds = embed(merged_questions)

  new_tree = build_tree(old_ids, new_embeds[1:])

  query = dict(
    id = -1,
    data = new_embeds[0]
  )

  matches = new_tree.get_n_nearest_neighbors(query, 10)

  result_ids = results_list(matches, -1)

  out = {
    'ids': result_ids
  }

  # Just to be sure...
  del new_tree
  del old_questions
  del merged_questions
  del old_ids
  del new_embeds

  return out, 200

if __name__ == "__main__":
  # use 0.0.0.0 to use it in container
  app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))