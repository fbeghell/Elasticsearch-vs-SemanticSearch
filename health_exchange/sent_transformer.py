from sentence_transformers import SentenceTransformer, util
from health_exchange.utils import text_extract
import torch
import time
import pickle
import os.path
import warnings
import os
from argparse import Namespace
from config.config import logger
from typing import List, Dict, Union

VERBOSE = False

def load_model(model_filepath: str, device: str):
    """Loads a model from Huggingface 

    Args:
        model_filepath (str): the Higgingface filepath of the model
        device (object): "cuda" (GPU) or "cpu"

    Returns:
        object: the model
    """
    model_name = os.path.basename(model_filepath)
    model = SentenceTransformer(model_filepath, device)
    logger.info(f"load_model: Loaded model {model_name}")
    return model


def encode_corpus(embedder: object, corpus: Union[str, List[str]]) -> List:
    """Computes sentence embeddings for a corpus of texts.

    Args:
        embedder (object): the Transformers model
        corpus (List): list of texts (snippets, paragraphs, sentences) to create sentence embeddigns from

    Returns:
        List: list of sentence embeddings (tensors)
    """
    return embedder.encode(corpus, convert_to_tensor=True)


def get_search_results_for_query(query: str, embedder: object, embeddings: List, corpus: List, snippet_info: Dict, num_top_results: int):
    """Encodes user query into a sentence embedding, and compares for similarity with the list of text embeddings from
    the search corpus. Returns the texts of the top 5 most similar corpus embeddings as search results, along with display info, such 
    as their document id, snippet id, and tags. 

    Args:
        query (str): the user query
        embedder (object): the Transformer model
        embeddings (List): list of embeddings (tensors) from the search corpus
        corpus (List): list of preprocessed texts from the search corpus
        snippet_info (Dict): dictionary to retrieve display info for the preprocessed snippet
        num_top_results (int): number of top results to return

    Returns:
        List: list of top num_top_results search results, with keys for display items (doc id, snippet id, tags)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        out = []
        results = torch.topk(cos_scores, k=num_top_results)
        rank = 0
        for _score, idx in zip(results[0], results[1]):
            prepped_snippet = corpus[idx]
            if prepped_snippet in snippet_info.keys():
                rank += 1
                snip_id = snippet_info[prepped_snippet][0] + ':' + snippet_info[prepped_snippet][1]
                out.append({'query': query.strip(), 'grade': '', 'rank': rank, 'id': snip_id, 
                            'snippet': snippet_info[prepped_snippet][2], 'tags': snippet_info[prepped_snippet][3], 'notes': 'Score '+ str(round(_score.item(), 2))})
                if VERBOSE: print(f"query: {query.strip()}, grade: , rank: {rank}, id: {snip_id}, \
                                    snippet: {snippet_info[prepped_snippet][2]}, notes: Score {str(round(_score.item(), 2))}")
    return out