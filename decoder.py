import torch.nn as nn 
import torch
import torch.nn.functional as F 

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