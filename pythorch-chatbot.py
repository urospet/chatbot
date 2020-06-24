from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
from torch.jit import script, trace
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F 
import csv
import random
import os
import codecs
from io import open
import itertools
import math

from torch.utils.tensorboard import SummaryWriter
import torchvision

from formatData import loadLines, loadConversations, extractSentencePairs
from vocabulary import Voc, PAD_token, EOS_token, SOS_token
from cleanData  import loadPrepareData, normalizeString, MAX_LENGTH
from evaluation import GreedySearchDecoder, BeamSearchDecoder
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def printLines(file, n=10):
    """function to print n lines from a file """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

#############################################################################
# PART 1 : LOAD & PREPROCESS DATA 
# Using the Cornell Movie-Dialogs Corpus :  Conversational exchanges from 617 movies
# Reformat data into structures of form question-answers
# Output : formatted_movie_lines.txt
#############################################################################

#Looking at the original format
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

# printLines(os.path.join(corpus, "movie_lines.txt"))

#Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")


# Initialize lines dict, conversations list and field ids
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS= ["charcter1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), 
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file
print("\nWritting newly formatted file...")

delimiter = '\t'
#Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

#Print a sample of line
print("\nSample lines from file:")
printLines(datafile)

##############################################################################
#PART 2 : LOAD AND TRIM DATA 
# Mapping each unique word to a discrete numerical space
##############################################################################


# Preprocessing : convert to ascii, lowercase, trim non-letter, max-length
# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
#Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# Trimming rarely used words out of our vocabulary for faster convergence
MIN_COUNT = 10 # Minimum word count threshold for trimming 

def trimRareWords(voc, pairs, MIN_COUNT):
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

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


#############################################################################
# PART 3: PREPARE DATA FOR MODELS
# Models expect numerical torch tensors as inputs
# USing mini-batch we need to have same length for sentences so we pad
#############################################################################

#Converting words to their indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
 
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

##############################################################################
# PART 4 : DEFINE SEQ2SEQ MODEL
# Using two RNN one for encoder (historical data) and one for decoder (Predictions)
##############################################################################

#############################################################################
# PART 4.1: ENCODER
# Steps : 1. Convert word indexes to embeddings
#         2. Pack padded batch of sequences for RNN module.
#         3. Forward pass through GRU
#         4. Unpack padding
#         5. Sum bidirectional GRU outputs
#         6. Return output and final hidden state
# Inputs : input_seq - batch of input sentences - shape(max_length, batch_size)
#       input_lengths - list of sentence lengths corresponding to each sentence in the batch - shape (batch_size)
#       hidden - hidden state - shape(n_layers * num_directions, batch_size, hidden_size)
# Output : outputs - output features from the last hidden layer of the GRU (sum of bidirectional outputs) - shape = (max_length, batch_size, hidden_size)
#       hidden : updated hidden state from GRU - shape(n_layer*num_directions, batch_size, hidden_size)

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        
        # Intialize GRU : the input_size and hidden_size params are both set to 'hidden_size'
        # because out input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, 
                          hidden_size, 
                          n_layers, 
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        #Convert word indexes to embeddings 
        embedded = self.embedding(input_seq)
        
        #Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        #Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        
        #Unpack padding 
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        #Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        #Return output and final hidden state
        return outputs, hidden


##############################################################################
# PART 4.2 : DECODER
# Generate the answer sentence, use encoder's context vectors
# Steps : 1. Get embedding of current input word.
#         2. Forward through unidirectional GRU
#         3. Calculate attention weights from the current GRU output from(2)
#         4. Multiply attention weights to encoder outputs to get new "weighted sum" context vector
#         5. Concatenate weighted context vector and GRU output using Luong eq. 5.
#         6. Predict next word using Luong eq. 6. (without softmax)
#         7. Return output and final hidden state
# Inputs : input_step - one time step (one word) of input sequence batch, shape(1, batch_size)
#         last_hidden - final hidden layer of GRU, shape(n_layer * num_directions, batch_size, hidden_size)
#         encoder_outputs - encoder model's output - shape(max_length, batch_size, hidden_size)
# Outputs : output - softmax normalized tensor giving the probabilites of each word being the correct next word in the decoded sequence - shape (batch_size, voc.num_words)
#           hidden : final hidden state of GRU - shape(n_layer * num_directions, batch_size, hidden_size)

#Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    
    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)
    
    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                                      encoder_output), 2)).tanh()
        
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        #Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        
        #Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = Attn(attn_model, hidden_size)
        
    def forward(self, input_step, last_hidden, encoder_outputs):
        #Note : we run this one step (word) at a time
        
        #Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        #Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        #Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        #Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        
        #Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
    
        #Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        #Return output and final hidden state
        return output, hidden
    
##############################################################################
# PART 5 : DEFINE TRAINING PROCEDURE
#Steps: 1. Forward pass entire input batch through encoder
#       2. Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state
#       3. Forward input batch sequence through decoder one time step at a time
#       4. If teacher forcing : set next decoder input as the current target; eslse :set next decoder input as current decoder output
#       5. Calculate and accumulate loss. 
#       6. Perform backpropagation
#       7. Clip gradients
#       8. Update encoder and decoder model parameters
##############################################################################

#Calculate loss based on decoder's output tensor, target tensor, and a binary mask tensor
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

#Algorithm for a signle training iteration (a single batch of inputs)
# Using teacher forcing, gradient clipping 
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    # Forward batch of sequence through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len): 
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            #Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1,-1)
            
            #Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # No teacher forcing : next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            
    # Perform backpropagation
    loss.backward()
    
    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses) / n_totals

# Training iterations 
# Tie the full training procedure together with the data
# Running n iterations
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every,
               save_every, clip, corpus_name, loadFilename):
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]  
    
    writer = SummaryWriter('runs/mnist_experiment_1')
    
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        
    # Training loop
    print('Training ...')
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            writer.add_scalar('training loss', print_loss_avg, iteration)
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
        
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.
                                     format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
        
##############################################################################
# PART 6 : DEFINING EVALUATION
# Talkin to the bot
# Defining how the model decode the encoded input
##############################################################################


# Evaluating, input: our sentence, output: bot answer
# Format input sentence as a batch
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    #words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    #print(scores)
    #print(tokens)
    # Indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            
            # Check if it is quit case 
            if input_sentence == 'q' or input_sentence == 'quit': break
            
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            
            # Evaluate sentence 
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        
        except KeyError:
            print("Error: Encountered unknown word.")
            
##############################################################################
# PART 7 : RUN THE MODEL
# Choose to start from scratch or set a checkpoint to load from
#############################################################################

# Configure models
model_name = 'cb_model6'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2 
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000

#loadFilename = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(
 #                  encoder_n_layers, decoder_n_layers, hidden_size),
 #                  '{}_chceckpoint.tar'.format(checkpoint_iter))
#loadFilename = os.path.join(r"C:\Users\uros\Desktop\chatbot-udes\data\save\cb_model5\cornell movie-dialogs corpus\2-2_500\4000_checkpoint.tar")
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
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

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

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
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
           print_every, save_every, clip, corpus_name, loadFilename)
"""
##############################################################################
# FINAL STEP : TALKING WITH THE BOT
##############################################################################

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
#searcher = GreedySearchDecoder(encoder, decoder)
searcher = BeamSearchDecoder(encoder, decoder, beamWidth = 10)
# Begin chatting 
evaluateInput(encoder,decoder,searcher,voc)
"""