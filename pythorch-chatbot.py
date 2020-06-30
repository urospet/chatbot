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
from evaluationProcedure import evaluateInput
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#############################################################################
# PART 1 : LOAD & PREPROCESS DATA 
# Using the Cornell Movie-Dialogs Corpus :  Conversational exchanges from 617 movies
# Reformat data into structures of form question-answers
# Output : formatted_movie_lines.txt
#############################################################################

#Uncomment to format the data
#formatData()


##############################################################################
#PART 2 : LOAD AND TRIM DATA 
# Mapping each unique word to a discrete numerical space
##############################################################################


# Preprocessing : convert to ascii, lowercase, trim non-letter, max-length
# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

#Print some pairs to validate
#print("\npairs:")
#for pair in pairs[:10]:
#    print(pair)

# Trim voc and pairs
pairs = trimRareWords(voc, pairs)


#############################################################################
# PART 3: PREPARE DATA FOR MODELS
# Models expect numerical torch tensors as inputs
# USing mini-batch we need to have same length for sentences so we pad
#############################################################################

# Example for validation
#small_batch_size = 5
#batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
#input_variable, lengths, target_variable, mask, max_target_len = batches

#print("input_variable:", input_variable)
#print("lengths:", lengths)
#print("target_variable:", target_variable)
#print("mask:", mask)
#print("max_target_len:", max_target_len)

##############################################################################
# PART 4 : DEFINE SEQ2SEQ MODEL
# Using two RNN one for encoder (historical data) and one for decoder (Predictions)
##############################################################################

###################
# PART 4.1: ENCODER
# See encooder.py

####################
# PART 4.2 : DECODER
# See decoder.py 
    
##############################################################################
# PART 5 : DEFINE TRAINING PROCEDURE
##############################################################################

# See trainingProcedure.py
        
##############################################################################
# PART 6 : DEFINING EVALUATION
# Talkin to the bot
# Defining how the model decode the encoded input
##############################################################################

# See evaluationProcedure.py

##############################################################################
# PART 7 : RUN THE MODEL
# Choose to start from scratch or set a checkpoint to load from
#############################################################################

# Configure models
model_name = 'cb_model7'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 512
encoder_n_layers = 2 
decoder_n_layers = 2
dropout = 0.1
batch_size = 32

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 8000
checkpoint = ''
#loadFilename = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(
 #                  encoder_n_layers, decoder_n_layers, hidden_size),
 #                  '{}_chceckpoint.tar'.format(checkpoint_iter))
loadFilename = os.path.join(r"C:\Users\uros\Desktop\chatbot-udes\data\save\cb_model7\cornell movie-dialogs corpus\2-2_512\8000_checkpoint.tar")
# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    
    #If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
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
print('Models built and ready to go!')

##############################################################################
# STEP 8 : RUN THE TRAINING
##############################################################################
"""
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 8000
print_every = 10
save_every = 500
# Ensure dropout
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, 
           teacher_forcing_ratio, checkpoint, hidden_size)

##############################################################################
# FINAL STEP : TALKING WITH THE BOT
##############################################################################
"""
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)
#searcher = BeamSearchDecoder(encoder, decoder, beamWidth = 5)
# Begin chatting 
evaluateInput(encoder,decoder,searcher,voc)
