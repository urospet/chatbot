import torch

from cleanData   import  MAX_LENGTH, normalizeString
from prepareData import  indexesFromSentence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

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
            