# -*- coding: utf-8 -*-
import os
import torch
from vocabulary import Voc
import torch.nn as nn 

from encoder     import EncoderRNN
from decoder     import LuongAttnDecoderRNN

from formatData  import corpus_name
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def load_model():
    hidden_size = 512
    encoder_n_layers = 2 
    decoder_n_layers = 2
    dropout = 0.1
    attn_model = 'dot'
    
    loadFilename = os.path.join(r"C:\Users\uros\Desktop\chatbot-udes\data\save\cb_model101\openSubtitles+cornell\2-2_512\10000_checkpoint.tar")
    
    checkpoint = torch.load(loadFilename)
    
    #If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']
    
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        
        embedding.load_state_dict(embedding_sd)
        
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, 
                                  decoder_n_layers, dropout)
    
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder.eval()
    decoder.eval()
    
    return voc, encoder, decoder