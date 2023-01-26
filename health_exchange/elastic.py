from elasticsearch import Elasticsearch, helpers
import elasticsearch as elastic
import json
import pandas as pd
import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re
import os
from config.config import logger
from typing import List, Dict
from health_exchange.utils import text_extract, utils
from argparse import Namespace

# globals
tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words("english")
stopword_pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
porter = PorterStemmer()

max_words = 128
lower = True
remove_punct = True
remove_stops = True
stem = False

VERBOSE = False

def connect_to_es() -> elastic.client.Elasticsearch:
    """Connects to the elasticsearch server.

    Returns:
        es (elastic.client.Elasticsearch): the Elasticsearch client
    """
    es = None
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if es.ping():
        logger.info('Connection to Elasticsearch successful')
    else:
        logger.error('Could not connect to Elasticsearch')
    return es


def create_index(es_object: elastic.client.Elasticsearch, index_name: str):
    """Creates an Elasticsearch index

    Args:
        es_object (elastic.client.Elasticsearch): the Elasticsearch client
        index_name (str): the name of the index

    Returns:
        _type_: _description_
    """
    created = False
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "members": {
                "dynamic": "strict",
                "properties": {
                    "snippet": {
                        "type": "text"
                    }
                }
            }
        }       
    }
    try:
        if not es_object.indices.exists(index_name):
            es_object.indices.create(index=index_name, ignore=400, body=settings)
        logger.info(f"Created Elasticsearch Index with name: {index_name}")
        created = True
    except Exception as ex:
        logger.error(f"create_index: Could not create Elasticsearch index {index_name}: {str(ex)}")
    finally:
        return created


def store_record(elastic_object: elastic.client.Elasticsearch, index_name: str, record: object):
    """Stores a single document as a record in the index

    Args:
        elastic_object (elastic.client.Elasticsearch): the Elasticsearch object 
        index_name (str): the name of the index
        record (_type_): _description_
    """
    try:
        outcome = elastic_object.index(index=index_name, body=record)
    except Exception as ex:
        logger.error("store_record: Error in indexing data: ", str(ex))


def bulk_store_records(es_object: elastic.client.Elasticsearch, records: List, index_name: str):
    """Stores records in bulk

    Args:
        es_object (elastic.client.Elasticsearch): the Elasticsearch client
        records (List): list of all the snippets in the search corpus, preprocessed, as json objects    
        index_name (str): the Elasticsearch index
    """
    try:
        helpers.bulk(es_object, records, index=index_name)
    except Exception as ex:
        logger.error(f"bulk_store_records: Elasticsearch error in bulk indexing {str(ex)}")



def get_query_results(es_object: elastic.client.Elasticsearch, raw_query: str, prepped_query: str, index_name: str, snippet_info_file: str) -> List[Dict]:
    """Submits a raw and preprocessed query to Elasticsearch client and index, embeds the latter into a json search string, and returns top 5 search
    results, formatted as document id, snippet id, snippet, and tags. 

    Args:
        es_object (elastic.client.Elasticsearch): the Elasticsearch client
        raw_query (str): the query, as submitted by user
        prepped_query (str): preprocessed query
        index_name (str): Elasticsearch index
        snippet_info_file (str): filepath of dictionary to retrieve display info for the snippet

    Raises:
        ValueError: if the snippet_info_file cannot be found

    Returns:
        List[Dict]: list of top 5 search results, with keys for display items
    """
    if not os.path.exists(snippet_info_file):
        raise ValueError("Error: could not find Snippet Info file")
    snippet_info = utils.load_dict(snippet_info_file)
    res = es_object.search(index=index_name,
                        # body={"query": {
                        #             "more_like_this": {
                        #                 "fields": ["snippet"],
                        #                 "like": prepped_query,
                        #                 "min_term_freq": 1,
                        #                 "max_query_terms": 50,
                        #                 "min_doc_freq": 1
                        #             }
                        #         }
                        # })
                        body={"query": {
                            "match": {
                                "snippet": prepped_query
                            }
                        }})
    out = []
    if VERBOSE: print("Query:", prepped_query)
    rank = 0
    for r in res["hits"]["hits"]:
        if rank > 4: break
        prepped_snippet = r["_source"]["snippet"]
        if prepped_snippet in snippet_info.keys():
            rank += 1
            snip_id = snippet_info[prepped_snippet][0] + ':' + snippet_info[prepped_snippet][1]
            if VERBOSE: print(f"query: {raw_query.strip()}, grade: , rank: {rank}, id: {snip_id}, snippet: {snippet_info[prepped_snippet][2]}, tags: {snippet_info[prepped_snippet][3]}, notes: ")
            out.append({'query': raw_query.strip(), 'grade': '', 'rank': rank, 'id': snip_id, 'snippet': snippet_info[prepped_snippet][2], 'tags': snippet_info[prepped_snippet][3], 'notes': ''})
        else:
            logger.error("query_search_es: Could not find info for", prepped_snippet)
    if VERBOSE: print("----------------------------------------------------------------------")
    return out
