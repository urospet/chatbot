import torch
import torch.nn as nn 
import math
import statistics
from vocabulary import Voc, PAD_token, EOS_token, SOS_token
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
##############################################################################
#Greedy decoding
##############################################################################
# For each time step, choose the word with the highest softmax value
# Steps: 1. Forward input through encoder model.
#        2. Prepare encoder's final hidden layer to be first hidden input to the decoder
#        3. Initialize decoder's first input as SOS_token
#        4. Initialize tensors to append decoded words to
#        5. Iteratively decode one word token at a time:
#               1. Forward pass through decoder
#               2. Obtain most likely word token and its softmax score
#               3. Record token and score
#               4. Prepare current token to be nex decoder input
#        6. Return collections of word tokens and scores
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, input_seq, input_length, max_length):
        # 1.Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        # 2.Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        
        # 3.Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        
        # 4.Initialize tensors to append decoded words to 
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        
        # 5.Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Obtain most likely word token and tis softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            
        #Return collections of word tokens and scores
        return all_tokens, all_scores

##############################################################################
# Beam search decoder
# keeps track of k states rather than just one
# For each time step, choose k_width words with the highest softmax value
# Steps: 1. Forward input through encoder model.
#        2. Prepare encoder's final hidden layer to be first hidden input to the decoder
#        3. Initialize decoder's first input as SOS_token
#        4. Initialize tensors to append decoded words to
#        5. Iteratively decode one word token at a time:
#               1. Forward pass through decoder
#               2. Obtain most likely word token and its softmax score
#               3. Record token and score
#               4. Prepare current token to be nex decoder input
#        6. Return collections of word tokens and scores
##############################################################################

# x = beamWidth, 10 serait cool, faut tester la vitesse...
#Choisi les x les plus probables
#Pour chacun des x choix les plus probables. Quel est le  
#On veut garder les x paires de deux mots les plus probables
#Pour calculer la probabilité : probabilité du premier * probabilité du deuxième sachant le premier
#Multiplier juste les probabilité obtenu. 
#Faire une moyenne pour ne pas que ca prefere les phrasescourtes divisé par le nombre de mots exposant 0,7 (parametre) pour ne pas faire une normalisation complète



class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, beamWidth):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beamWidth = beamWidth
        
    def forward(self, input_seq, input_length, max_length):
        # 1.Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        # 2.Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        
        # 3.Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        
        # 4.Initialize tensors to append decoded words to 
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device) 
        beamDecoders = [[decoder_input, all_tokens, all_scores, all_scores]]
        
        # 5.Iteratively decode one word token at a time
        for _ in range(max_length):
            possibleInputs = []
            
            # For each decoder_input decode beamWidth 
            for decoder_input, tokens, scores, meanScores in beamDecoders:
                #print("decoder start", decoder_input)
                if decoder_input.item() == EOS_token:
                    node = [decoder_input, tokens, scores, meanScores]
                    possibleInputs.append(node)
                else:   
                    # Forward pass through decoder
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    
                    # Keep the beamWidth  most likely word token and it's softmax score
                    prob, indexes = torch.topk(decoder_output, self.beamWidth)
                    for i in range(self.beamWidth):
                        decoded_input= indexes[0][i].view(1, -1)                    
                        log_p = [math.log(prob[0][i].item())*-1]
                      #  print("INPUT",decoded_input)
                     #   print("log_prob", log_p)
                        newTokens = torch.cat((tokens, decoded_input), dim=0)
                        newScores = torch.cat((scores, torch.cuda.FloatTensor(log_p)), dim=0)                    
                        
                        # Prepare current token to be next decoder input (add a dimension)                    
                        #decoder_input = torch.unsqueeze(decoder_input, 0)
                        
                        node = [decoded_input, newTokens, newScores, statistics.mean(newScores.cpu().numpy())]
                        possibleInputs.append(node)
                 #   print("----")
            #Clear beamDecoders to prepare for the next loop
            beamDecoders = []
            sortedPossibleInputs = sorted(possibleInputs, key= lambda x: x[3], reverse=True)  
          
            #Keep the beamWidth best results
            for i in range(self.beamWidth):
                beamDecoders.append(sortedPossibleInputs.pop())
        
        final_tokens = beamDecoders[0][1]
        final_scores = beamDecoders[0][2]
        #Return collections of word tokens and scores
        return final_tokens, final_scores
        