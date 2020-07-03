import unicodedata
import re
from vocabulary import Voc

MAX_LENGTH = 20 # Maximum sentence length to consider

# Trimming rarely used words out of our vocabulary for faster convergence
MIN_COUNT = 3 # Minimum word count threshold for trimming 

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')        

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    
    # Unicode string to plain ASCII
    s = unicodeToAscii(s.lower().strip())
    
    # Replacing any .!? by a whitespace plus the caracter ' \1' means
    # the first bracketed group r is not to consider ' \1' as an individual 
    # character r in r" \1" is to escape the backslash
    s = re.sub(r"([.!?])", r" \1", s)
    
    # Removing any character that is not a sequence of lower or upper case letters
    # + means one or more
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    
    # Removing a sequence of whitespace characters
    s = re.sub(r"\s+", r" ", s).strip()
    
    #s = re.sub(r"i'm", "i am", s)
    #s = re.sub(r"he's", "he is", s)
    #s = re.sub(r"she's", "she is", s)
    #s = re.sub(r"that's", "that is", s)
    #s = re.sub(r"what's", "what is", s)
    #s = re.sub(r"where's", "where is", s)
    #s = re.sub(r"how's", "how is", s)
    #s = re.sub(r"\'ll", " will", s)
    #s = re.sub(r"\'ve", " have", s)
    #s = re.sub(r"\'re", " are", s)
    #s = re.sub(r"\'d", " would", s)
    #s = re.sub(r"n't", " not", s)
    #s = re.sub(r"won't", "will not", s)
    #s = re.sub(r"can't", "cannot", s)
    #s = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", s)
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
    if p:
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH 
    else:
        return False

# Filter pairs using filterPair condition
def filterPairs(pairs):
    pairs = [pair for pair in pairs if pair != ['']]
    return [pair for pair in pairs if filterPair(pair)]

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

def trimRareWords(voc, pairs):
    #Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    #Filter out the pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair [0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        
        #Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        
        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
            
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),
                                                                len(keep_pairs),
                                                                len(keep_pairs)/len(pairs)))
    return keep_pairs