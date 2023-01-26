import re
from health_exchange.utils.text_extract import get_word_len


# Common English abbreviations: should not split when these are followed by '. '
alphabets= r"([A-Za-z])"
prefixes = r"(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|www)[.]"
suffixes = r"(Inc|Ltd|Jr|Sr|Co|Corp|al|co)"  # added 'al', 'Corp', 'co'
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = r"[.](com|net|org|io|gov|me|edu|co|uk)"  # added 'co', 'uk'
digits = r"([0-9])"


def split_into_sentences(text: str) -> list:
    """RegEx English sentence splitter with configurable abbreviations. 
    Courtesy of D. Greenberg, who posted this as an answer on 
    https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences

    Args:
        text (str): The continuous text to be split into sentences

    Returns:
        list: List of individual sentences in text
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def split_into_paragraphs(text: str) -> list:
    """Paragraph splitter. Paragraph is defined as non-empty text preceded or followed by
    a line break ('\n"). 

    Args:
        text (str): the text to be split into paragraphs

    Returns:
        list: list of paragraphs in the input text
    """
    return list(filter(None, text.split("\n")))


def cost(word_count: int, max_words: int, penalty: int=-3):
    """Determine whether combining two texts into a new text incurs an acceptable cost, based on a target max_word length. 
    * If the first text has fewer words that the target size, the cost is the difference between target word size and text word size.
    * If the combined text has more words than the target size, the cost is higher: the difference between the number of words
      in the combined texts and the target size, multiplied by a constant, the 'penalty' (e.g. -2 or -3, etc.)

    Args:
        word_count (int): the number of words of a single text or two consecutive texts (e.g. sentences or paragraphs)
        max_words (int): the target size in number of words to aproximate
        penalty (int): the cost of exceeding the target size. Defaults to -3.

    Returns:
        int: cost of a text with word_count words
    """
    cost = max_words - word_count
    if cost < 0:
        cost = cost * penalty
    return cost


DEBUG = False
def cost_based_combine(texts: list, max_words: int=30, min_words: int=5, cost_fn=cost):
    """Goal is to combine consecutive short sentences within each paragraph into sequences of sentences (aka 'Snippets')
    whose overall word count is close to 30 words. This is the "max_words" setting. Snippets are the mini-documents
    that are used in semantic search. Similarly, consecutive paragraphs in a Post can be re-arranged to approximate
    a target word count.
    Combination relies on a 'cost' function: snippets that are shorter than max_words incur a lower cost than 
    snippets that exceed max_words. For each consecutive sentence in a paragraph, combination is done if the cost of 
    combining with the next sentence is less than the cost of not combining. All combination is paragraph internal:
    for semantic coherence, we avoid combination across paragraph boundaries. 

    Args:
        texts (list): list of consecutive texts to possibly recombine to aproximate ideal target word length
        max_words (int): the target word count of texts (e.g. 30)
        min_words (int): the minimal number of words for a text (e.g. 5)
        cost_fn (_type_): function to evaluate the distance in word length of a text from the ideal target

    Returns:
        list: list of consecutive texts, possibly with some texts combined
    """
    output_texts = []
    i = 0
    done = False  # last text handled?
    while i < (len(texts) - 1):
        if len(texts[i].strip()) == 0:
            i += 1
            continue
        curr_text_wc = get_word_len(texts[i])
        if curr_text_wc < max_words:        # short text: get cost of combining with next text
            output_text_wc = curr_text_wc
            if DEBUG : print(f"\t\tin: {curr_text_wc}")
            output_text = texts[i]
            while (cost_fn(output_text_wc + get_word_len(texts[i+1]), max_words) < cost_fn(output_text_wc, max_words)) or \
                ((get_word_len(texts[i+1]) < min_words*2) and (output_text_wc + get_word_len(texts[i+1])) < max_words + min_words*2):
                output_text_wc += get_word_len(texts[i+1])
                output_text = output_text + ' ' + texts[i+1]
                if DEBUG : print(f"\t\tin +: {get_word_len(texts[i+1])} -- out: {output_text_wc}")
                i += 1
                if i == len(texts) - 1:
                    done = True
                    break
            i += 1
            output_texts.append(output_text)
        else:                               # text is long. Keep as is.
            output_texts.append(texts[i])
            i += 1
    if not done and i < len(texts) and len(texts[i].strip()) > 0:  # if we have not processed the last text
        output_texts.append(texts[i])
    return output_texts


def chunk_long_sentences(orig_sentences: list, max_words: int, allowed_excess: int=10, punct: list=[';', ':']) -> list:
    """Splits a long sentence into smaller chunks using punctuation as separators (e.g. ';', ':')

    Args:
        orig_sentences (list): the sentences to possibly chunk
        max_words (int): the max_word target length for sentences
        allowed_excess (int, optional): allow sentences to exceed target by these many words. Defaults to 10.
        punct (list, optional): punctuation to split by. Defaults to [';', ':'].

    Returns:
        list: new list of sentences, some of which may be chunks of orig_sentences
    """
    out_sentences = []
    punct_re =  '|'.join(map(re.escape, punct))
    for orig_sent in orig_sentences:
        if get_word_len(orig_sent) > max_words + allowed_excess:
            if re.search(punct_re, orig_sent):
                chunks = re.split(punct_re, orig_sent)
                for chunk in chunks:
                    out_sentences.append(chunk.strip())
            else:
                out_sentences.append(orig_sent)
        else:
            out_sentences.append(orig_sent)
    return out_sentences