import unicodedata
import re
from vocabulary import Voc

MAX_LENGTH = 25 # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')        

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"i'm", "i am", s)
    s = re.sub(r"he's", "he is", s)
    s = re.sub(r"she's", "she is", s)
    s = re.sub(r"that's", "that is", s)
    s = re.sub(r"what's", "what is", s)
    s = re.sub(r"where's", "where is", s)
    s = re.sub(r"how's", "how is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"n't", " not", s)
    s = re.sub(r"won't", "will not", s)
    s = re.sub(r"can't", "cannot", s)
    s = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", s)
    return s   
    

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff bot senteces in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#def sortPairs(pairs):
#    """Sorting questions and answers by the length of questions 
#       Speed up the training and reduce the loss, reduce amount of padding"""
#    for length in range(1, 25 + 1):
#        for pair in pairs:
#            if (pair[1]) == length:
#   for i in enumerate(questionsToInt):
#        if len(i[1]) == length:
#            sorted_clean_questions.append(questionsToInt[i[0]])
#            sorted_clean_answers.append(answersToInt[i[0]])

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

