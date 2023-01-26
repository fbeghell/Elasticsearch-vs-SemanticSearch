import pytest
import pandas as pd
import csv
import numpy as np
from health_exchange.utils import text_extract
from health_exchange.utils import text_split



@pytest.fixture(scope="module")
def df():
    post1_body = """One Two. Three Four Five Six Seven. Eight. Nine Ten"""
    post2_body = """One Two Three Four Five. Six Seven Eight. Nine Ten"""
    post3_body = """One Two Three Four; Five Six: Seven. Eight. Nine Ten"""
    post4_body = """One Two Three Four Five Six Seven. Eight. Nine Ten"""
    data = [
        {"Id": "1", "Title": "Title 1", "Body": post1_body, "ViewCount": "1", "Tags": "<tag1><tag11>"},
        {"Id": "2", "Title": "Title 2", "Body": post2_body, "ViewCount": "2", "Tags": "<tag2><tag22>"},
        {"Id": "3", "Title": "Title 3", "Body": post3_body, "ViewCount": "3", "Tags": "<tag3><tag33>"},
        {"Id": "4", "Title": "Title 4", "Body": post4_body, "ViewCount": "4", "Tags": "<tag4><tag44>"},
    ]
    df = pd.DataFrame(data)
    return df


@pytest.mark.parametrize(
    "html, text",
    [
        ("<p>paragraph <ol><li>list 1 </li><li>list 2 </li></ol>Some text</p>", "paragraph list 1 list 2 Some text"),
        ("no tags here", "no tags here"),
        ("""<p>The following tooth cracks are noticeable: </p><p><img src="https://i.stack.imgur.com/2sgis.jpg" alt="Teeth 1">\
            <img src="https://i.stack.imgur.com/k3R8j.jpg" alt="Teeth 2"></p>""", "The following tooth cracks are noticeable:"),
        ("", ""),
    ],
)
def test_strip_tags(html, text):
    assert text_extract.strip_tags(html) == text


@pytest.mark.parametrize(
    "text, wc",
    [
        ("One, Two, Three, Four", 4),
        ("One Two Three", 3),
        ("", 0),
        (" ", 0),
    ],
)
def test_get_word_len(text, wc):
    assert text_extract.get_word_len(text) == wc


@pytest.mark.parametrize(
    "html, text",
    [
        ("""<p>\nTwo   images    </p><p><img src="https://i.stack.imgur.com/2sgis.jpg" alt="Teeth 1">\
            <img src="https://i.stack.imgur.com/k3R8j.jpg"  alt="Teeth 2"></p>\n\n""", "Two images"),
    ]
)
def test_clean_html(html, text):
    print("XX" + text_extract.clean_html(html) + "XX")
    assert text_extract.clean_html(html) == text


@pytest.mark.parametrize(
    "long_sentences, max_words, excess, chunked_sentences",
    [
        (['One two', 'three four five; six seven: eight nine ten', 'eleven twelve thirteen; fourteen', 'sixteen, seventeen'],
         3, 1, ['One two', 'three four five', 'six seven', 'eight nine ten', 'eleven twelve thirteen; fourteen', 'sixteen, seventeen']),
        (['One two', 'three four five; six seven: eight nine ten', 'eleven twelve thirteen; fourteen', 'sixteen, seventeen'],
         2, 1, ['One two', 'three four five', 'six seven', 'eight nine ten', 'eleven twelve thirteen', 'fourteen', 'sixteen, seventeen']),
    ]
)
def test_chunk_long_sentences(long_sentences, max_words, excess, chunked_sentences):
    assert len(text_split.chunk_long_sentences(long_sentences, max_words=max_words, allowed_excess=excess)) == len(chunked_sentences)


@pytest.mark.parametrize(
    "sentences, max_words, min_words, out_sentences",
    [
        (["First sentence one two three four", "Second", "Third sentence"], 3, 1, 
         ['First sentence one two three four', 'Second Third sentence']),
        (["First sentence one two three four", "Second", "Third sentence five"], 3, 1,
         ['First sentence one two three four', 'Second', 'Third sentence five']),
        (["First sentence", "Second", "Third", "Fourth sentence one two three four"], 3, 1,
         ['First sentence Second Third', 'Fourth sentence one two three four'])
    ]
)
def test_cost_based_combine(sentences, max_words, min_words, out_sentences):
    assert text_split.cost_based_combine(sentences, max_words=max_words, min_words=min_words) == out_sentences


@pytest.mark.parametrize(
    "sentence_word_count, max_words, cost",
    [
        (23, 30, 7),
        (34, 30, 12),
        (30, 30, 0)
    ]
)
def test_cost(sentence_word_count, max_words, cost):
    assert text_split.cost(sentence_word_count, max_words=max_words, penalty=-3) == cost


@pytest.mark.parametrize(
    "text, sentences",
    [
        ("Mr. Smith and Ms. Smith et al. 23 Elm St. A. 22.3.1", 
         ['Mr. Smith and Ms. Smith et al. 23 Elm St. A. 22.3.']),
        ("Acme, Inc. and AAA, Corp. at acme.co.uk and aaa.org A.K.A. aaa.com.", 
         ['Acme, Inc. and AAA, Corp. at acme.co.uk and aaa.org A.K.A. aaa.com.']),
        ("https://www.kidney-international.org/article/S0085-2538(20)30369-0/fulltext, 26 autopsies were conducted on people.",
         ["https://www.kidney-international.org/article/S0085-2538(20)30369-0/fulltext, 26 autopsies were conducted on people."])
    ]
)
def test_split_into_sentences(text, sentences):
    assert text_split.split_into_sentences(text) == sentences


    