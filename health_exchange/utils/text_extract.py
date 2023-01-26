
import pandas as pd

from io import StringIO
from html.parser import HTMLParser
import re


class MLStripper(HTMLParser):
    """Wrapper around HTMLParser to extract text only from HTML.

    Args:
        HTMLParser (html.parser.HTMLParser): Parses text formatted in HTML.
        Is able to handle malformed HTML
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()


def strip_tags(html: str) -> str:
    """Wrapper around MLStripper

    Args:
        html (str): A text string formatted in HTML

    Returns:
        str: Text content of html string
    """
    s = MLStripper()
    s.feed(html)
    return s.get_data().strip()


def clean_html(html: str) -> str:
    """Adds functionality to function 'strip_tags': preserves paragraph breaks

    Args:
        html (str): A text string formatted in HTML

    Returns:
        str: Text only string, no tags, with single line breaks
    """
    html = html.replace("</p>", "\n")   # keep track of paragraph breaks
    text = strip_tags(html)
    text = re.sub("\n\n+", "\n", text)
    return re.sub(" +", " ", text)


def get_word_len(text: str) -> int:
    """Counts number of words in text string

    Args:
        text (str): plain text string

    Returns:
        int: Number of words (space separated)
    """
    if text is not None:
        return len(text.split())
    else:
        return 0
