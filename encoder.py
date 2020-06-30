import torch.nn as nn 


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
