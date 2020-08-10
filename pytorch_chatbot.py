from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torch.nn as nn 
from torch import optim
import os

from formatData  import corpus, corpus_name, datafile
from cleanData   import loadPrepareData, trimRareWords
from evaluation  import GreedySearchDecoder, BeamSearchDecoder
from encoder     import EncoderRNN
from decoder     import LuongAttnDecoderRNN
from trainingProcedure import trainIters
from vocabulary import Voc
from evaluationProcedure import evaluateInput
from load_model import load_model
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")



class ChatBot: 
    """Class to keep track of number of words, and their index in the vocabulary known"""
    def __init__(self, name):
        self.name = name
        self.voc, self.encoder, self.decoder = load_model()
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder)
        
    def get_response(self, input_text):
        return evaluateInput(self.encoder,self.decoder,
                             self.searcher,self.voc, input_text)
