# Default word tokens
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token

class Voc: 
    """Class to keep track of number of words, and their index in the vocabulary known"""
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        """For each word in the sentence add it"""
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        """If the word is not known in the vocabulary add it to the index, 
           , count for this word is one (first time we see it), increase numbers of words by one"""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
            
    def trim(self, min_count):
        """Remove words below a certain count threshold"""
        
        #Singleton trim just one time
        if self.trimmed:
            return
        self.trimmed = True
        
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))
        
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens
        
        for word in keep_words:
            self.addWord(word)
