# -*- coding: utf-8 -*-

###########################
#Create formatted data file 
###########################
import re
import os
import csv
import codecs

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS= ["charcter1ID", "character2ID", "movieID", "utteranceIDs"]

#Looking at the original format
#corpus_name = "cornell movie-dialogs corpus"
corpus_name = "openSubtitles+cornell"
corpus = os.path.join("data", corpus_name)

#Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")


def formatData():

    corpus_lines = os.path.join(corpus, "movie_lines.txt")
    corpus_conversations = os.path.join(corpus, "movie_conversations.txt")
    
    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(corpus_lines)
    print("\nLoading conversations...")
    conversations = loadConversations(corpus_conversations, lines)
    
    # Write new csv file
    print("\nWritting newly formatted file...")
    
    writeCsvFormatedData(datafile, conversations)



def printLines(file, n=10):
    """function to print n lines from a file """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

def loadLines(fileName):
    """" Split each line of the file into a dictionary of field (lineID, characterID, movieID, text"""
    #printLines(fileName)     
                                                            
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            #Extract fields
            lineObj = {}
            for i, field in enumerate(MOVIE_LINES_FIELDS):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines
            
def loadConversations(fileName, lines):
    """"Groups fields of lines from 'loadLines' into coversations base on 'movie_conversations.txt' """
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            #Extract fields
            convObj = {}
            for i, field in enumerate(MOVIE_CONVERSATIONS_FIELDS):
                convObj[field] = values[i]
            #Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            #Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds: 
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations
            
def extractSentencePairs(conversations):
    """"Extracts pairs of sentences from conversations"""
    qa_pairs= []
    for conversation in conversations: 
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"])-1): #Ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def writeCsvFormatedData(datafile, conversations):
    
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