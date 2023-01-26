import json
from typing import Dict, List
import os
import errno
from config.config import logger
import pickle


def read_text_file(filepath: str) -> List[str]:
    """Read a text file line by line

    Args:
        filepath (str): the file path to read from

    Returns:
        List[str]: List of strings, each a line in the filename
    """
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        return lines
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    if os.path.isfile(filepath):
        with open(filepath) as fp:
            d = json.load(fp)
        return d
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)    


def write_dict(d: Dict, filepath: str, msg: str, error_msg: str):
    """Write a dictionary to a JSON's filepath.

    Args:
        d (Dict): dictionary to write
        filepath (str): location of file to write to
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(d))
            logger.info(msg)
    except OSError as e:
        logger.error(error_msg, e.strerror)


def textify_tags(tags: str) -> str:
    """Converts Tags into a text string

    Args:
        tags (str): a concatenation of tags, each in format "<..>"

    Returns:
        str: a text string with the content of tags separated by space
    """
    tags = tags[1:-1]
    return " ".join(tag for tag in tags.split("><"))


def delete_file(filepath: str, msg: str, error_msg: str):
    """Deletes a file, with logging

    Args:
        filepath (str): the file to delete
        msg (str): log message for successful deletion
        error_msg (str): log message for unsuccessful deletion
    """
    if os.path.isfile(filepath):
        try:
            os.remove(filepath)
            logger.info(msg)
        except OSError as e:
            logger.error(error_msg, e.strerror)
    else:
        logger.warning(f"delete_file: file {filepath} was not found")
    

def pickle_list_to_file(lst: List, filepath: str, msg: str, error_msg: str):
    """Compresses a list into a pickle file

    Args:
        lst (List): the list to compress
        filepath (str): filepath of the pickle file
        msg (str): success message
        error_msg (str): error message
    """
    try:
        with open(filepath, 'wb') as fp:
            pickle.dump(lst, fp)
    except OSError as e:
        logger.error(error_msg, e.strerror)


def read_pickled_file(filepath: str, msg: str, error_msg: str) -> object:
    """Loads a pickle file

    Args:
        filepath (str): the path of the pickle file
        msg (str): success message
        error_msg (str): error message

    Raises:
        FileNotFoundError: if file path could not be found

    Returns:
        object: _description_
    """
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as fp:
                logger.info(msg)
                return pickle.load(fp)
        except OSError as e:
            logger.error(error_msg, e.strerror)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)



            