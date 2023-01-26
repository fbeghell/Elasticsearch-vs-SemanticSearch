import pandas as pd
import os
from health_exchange.utils.text_split  import split_into_paragraphs, split_into_sentences, cost_based_combine, chunk_long_sentences
from health_exchange.utils.text_extract import get_word_len, clean_html
import xml.etree.ElementTree as ET
import typing
import numpy as np
import csv
from typing import Dict, List
from config.config import logger


def extract_raw_data(xml_filename: str, fields: list) -> pd.DataFrame:
    """Extract raw data from XML StackExchange file.
    Each record in StackExchange XML files is tagged as 'row' with attributes (=the fields).
    Not all rows have consistently the same list of attributes. 

    Args:
        xml_filename (str): xml file from Stackexchange
        fields (list): list of row attributes to extract

    Returns:
        pd.DataFrame: dataframe with raw data
    """
    with open(xml_filename, 'r', encoding='utf-8') as fin:
        xml_data = fin.read()
    try:
        root = ET.XML(xml_data)
        data = []
        for i, row, in enumerate(root):
            attribs_dict = row.attrib
            attribs = []
            for field in fields:
                if field in attribs_dict:
                    if field == "":
                        attribs.append(None)
                    else: 
                        attribs.append(attribs_dict[field])
                else:
                    attribs.append(None)
            data.append(attribs)
    except Exception as e:
        logger.error(f"stackexchange_extract_raw_data: {str(e)}, row: {i}")
    df = pd.DataFrame(data, columns=fields)
    logger.info(f"Extracted {len(df)} data rows from {xml_filename}")
    return df


def preprocess_to_snippets(df: pd.DataFrame, outfile: str, max_words: int, min_words: int):
    """Preprocess Posts.xml by extracting text from HTML and splitting "Body" into sequences
    of sentences for processing as mini-documents for search. Sequences of consecutive sentences
    are referred to as 'snippets', and they approximate the target length of max_words.
    An output csv file is written with fields: Id, Snip(pet)Idx, Title, Snippet, Tags.

    Args:
        df (pd.DataFrame): Dataframe that contains most of the fields in Posts.xml
        outfile (typing.TextIO): csv file with Id, Snip(pet)Idx, Title, Snippet, Tags
        max_words (int): target max number of words for a snippet
        min_words (int): target min number of words for a snippet

    Returns:
        list(int): list of word lengths for all the extracted snippets
    """
    assert set(["Id", "Body", "Title", "Tags"]).issubset(set(df.columns))  # ensure the needed columns are there
    snippets_wcs = []
    with open(outfile, 'w', encoding='utf-8') as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["Id","SnipIdx","Title","Snippet","Tags"])
        for idx, row in df.iterrows():
            title = row["Title"]
            body = clean_html(row["Body"])
            orig_paragraphs = split_into_paragraphs(body)
            paragraphs = cost_based_combine(orig_paragraphs, max_words, min_words)
            s_idx = 0
            for _, parag in enumerate(paragraphs):
                sentences = split_into_sentences(parag)
                sentences = chunk_long_sentences(sentences, max_words)
                snippets = cost_based_combine(sentences, max_words, min_words)
                for i, s in enumerate(snippets):
                    s_idx += 1
                    writer.writerow([row["Id"], s_idx, row["Title"], s, row["Tags"]])
                    snippets_wcs.append(get_word_len(s))
    logger.info(f"Generated {len(snippets_wcs)} snippets from DataFrame")
    return snippets_wcs


def update_csv(infile: str, outfile: str, incolumns: List, outcolumns: List, update_fn: callable):
    """ Applies 'update_fn' function to a csv file, and saves the changes to a new
    csv file.

    Args:
        infile (typing.TextIO): the csv file to update
        outfile (typing.TextIO): the csv file with the changes
        incolumns (list): the columns in infile to apply the update_fn function to
        outcolumns (list): the infile columns in the outfile
        update_fn (function): the function to apply
    """
    with open(infile, 'r', encoding='utf-8') as fin:
        updated_cnt = 0
        reader = csv.DictReader(fin)
        updated = []
        for row in reader:
            update = update_fn(row, columns=incolumns, updated_cnt=updated_cnt)
            if update is not None:
                updated.append(update)
    
    with open(outfile, 'w', encoding='utf-8', newline='') as fin:
        writer = csv.DictWriter(fin, delimiter=',', fieldnames=outcolumns, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(dict((header, header) for header in outcolumns))
        writer.writerows(updated)
        
    logger.info(f"Updated {len(updated)} rows in {outfile}")


def remove_empty_body_rows(row: object, columns: List, updated_cnt: int) -> object:
    """Plug in function for 'update_csv': skips rows that have an empty or null 'Body' column

    Args:
        row (dict): row in DictReader
        columns (list): the column(s) to apply the function to
        updated_cnt (int): the number of rows that the function applied to

    Raises:
        ValueError: Exception thrown if the columns are not found in row

    Returns:
        dict: the updated row
    """
    if 'Body' not in columns:
        raise ValueError("Column to update not found")
    if not row['Body'] or row['Body'] == "":
        updated_cnt += 1
        return None
    else:
        return row


def remove_empty_title_tags_rows(row: object, columns: List, updated_cnt: int) -> object:
    """Plug in function for 'update_csv': skips rows that have empty or null 'Title' or 'Tags' column(s)

    Args:
        row (dict): row in DictReader
        columns (list): the column(s) to apply the function to
        updated_cnt (int): the number of rows that the function applied to

    Raises:
        ValueError: Exception thrown if the columns are not found in row

    Returns:
        dict: the updated row
    """
    if not (set(['Title', 'Tags']) <= set(columns)):
        raise ValueError("Column to update not found")
    if not row['Title'] or row['Title'] == "" or row['Title'] == 'None' or row['Title'] is None \
        or not row['Tags'] or row['Tags'] == "" or row['Tags'] == 'None' or row['Tags'] is None:
        updated_cnt += 1
        return None
    else:
        return row


def remove_empty_cols_rows(row: object, columns: List, updated_cnt: int) -> object:
    """Checks if a row has empty or null Body, Title, and/or Tags. Discards such rows.

    Args:
        row (Dict): Row in posts file
        columns (List): the columns to check
        updated_cnt (int): the number of rows filtered through

    Raises:
        ValueError: throws if the columns to check are not present

    Returns:
        Dict: input row with non-empty Body, Title, or Tags
    """
    if not (set(['Body', 'Title', 'Tags']) <= set(columns)):
        raise ValueError("Column to update not found")
    if not row['Title'] or row['Title'] == "" or row['Title'] is None \
        or not row['Tags'] or row['Tags'] == "" or row['Tags'] is None \
        or not row['Body'] or row['Body'] == "" or row['Body'] is None:
        updated_cnt += 1
        return None
    else:
        return row


def select_top_rows_by_viewcount(posts_df: pd.DataFrame, snippets_csv_file: str, row_limit=20000):
    """Selects top 'row_limit' (=20000) rows from snippets csv file, sorted by descending ViewCount

    Args:
        posts_df (pd.Dataframe): dataframe with posts id and their ViewCount
        snippets_csv_file (str): csv file with Id, SnippetId, Title, Body, Tags
        row_limit (int, optional): Defaults to 20000.
    """
    posts_df['ViewCount'] = posts_df['ViewCount'].astype(int)
    posts_df = posts_df[['Id', 'ViewCount']]
    posts_by_view_df = posts_df.sort_values(by=['ViewCount'], ascending=False)
    data_dir = './data'
    # Create dictionary post_id => snippets
    post_id_to_snippets = dict()
    with open(snippets_csv_file, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            id = row['Id']
            if id not in post_id_to_snippets.keys():
                post_id_to_snippets[id] = [(0, row['Title'], row['Tags'])]
            post_id_to_snippets[id].append((row['SnipIdx'], row['Snippet'], row['Tags']))
    # Write corpus file
    corpus_file = os.path.join(data_dir, "search_corpus.csv")
    count = 0
    with open(corpus_file, 'w', encoding='utf-8', newline='') as fout:
        columns = ['Id', 'SnipIdx', 'Snippet', 'Tags']
        writer = csv.DictWriter(fout, fieldnames=columns)
        writer.writeheader()
        for _, row in posts_by_view_df.iterrows():
            if count > row_limit:
                break
            id = str(row['Id'])
            if id in post_id_to_snippets.keys():
                values = post_id_to_snippets[id]
                for (snipidx, snippet, tags) in values:
                    writer.writerow({'Id': id, 'SnipIdx': snipidx, 'Snippet': snippet, 'Tags': tags})
                    count +=1  
        logger.info(f"Written search corpus file {corpus_file}")


if __name__ == "__main__":
    data_dir = './data'
    health_data_path = os.path.join(os.path.abspath(data_dir), "health.stackexchange.com")
    posts_xml_file = os.path.join(health_data_path, "Posts.xml")
    posts_fields = ['Id', 'PostTypeId', 'CreationDate', 'Score', 'ViewCount', 'Body', 
                'OwnerUserId', 'LastEditDate', 'LastActivityDate', 'Title',
                'Tags', 'AnswerCount', 'CommentCount', 'ContentLicense']
    # read Posts.xml into dataframe and csv
    posts_df = extract_raw_data(posts_xml_file, posts_fields)
    print(f"Extracted {len(posts_df)} raw posts from {os.path.basename(posts_xml_file)}")
    posts_csv_file = os.path.join(data_dir, "posts.csv")
    posts_df.to_csv(posts_csv_file, encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
    # remove rows with empty Body, Title, and/or Tags
    validated_posts_csv_file = os.path.join(data_dir, "posts_validated.csv")
    update_csv(posts_csv_file, validated_posts_csv_file, ['Body', 'Title', 'Tags'], posts_fields, remove_empty_cols_rows)
    validated_posts_df = pd.read_csv(validated_posts_csv_file, sep=',', usecols=['Id','Title','Body','Tags','ViewCount'])
    # preprocess dataframe
    snippets_csv_file = os.path.join(data_dir, "posts_snippets_validated.csv")
    wcs = preprocess_to_snippets(validated_posts_df, max_words=30, min_words=5, outfile=snippets_csv_file)
    print(f"Snippet Word Counts: Mean={round(np.mean(wcs),3)}, Median= {np.median(wcs)} STD={round(np.std(wcs), 3)}")
    # select top 20K rows by ViewCount and write to file
    select_top_rows_by_viewcount(validated_posts_df, snippets_csv_file)
