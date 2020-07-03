import sys
import csv
import codecs
from formatData import printLines

FILENAME = 'data/twitter/chatTwitter.txt'
OUTPUTNAME = 'data/twitter/formatedTwitter.txt'


def writeCsvFormatedData(datafile, pairs):
    
    delimiter = '\t'
    #Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in pairs:
            writer.writerow(pair)
    
    #Print a sample of line
    print("\nSample lines from file:")
    printLines(datafile)


""""Extracts pairs of sentences from conversations"""
def extractSentencePairs(conversations):
    qa_pairs= []

    for i in range(0,len(conversations)-1, 2): #Ignore the last line (no answer for it)
        inputLine = conversations[i].strip()
        targetLine = conversations[i+1].strip()
        # Filter wrong samples (if one of the lists is empty)
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])
    return qa_pairs


""" Read lines from file
return [list of lines] """
def read_lines(filename):
    return open(filename, encoding="iso-8859-1").read().split('\n')[:-1]

def process_data():
    
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)
    pairs = extractSentencePairs(lines)
    print(len(pairs))
    writeCsvFormatedData(OUTPUTNAME, pairs)  

process_data()