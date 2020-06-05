# -*- coding: utf-8 -*-

###########################
#Create formatted data file 
###########################
import re

def loadLines(fileName, fields):
    """" Split each line of the file into a dictionary of field (lineID, characterID, movieID, text"""
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            #Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines
            
def loadConversations(fileName, lines, fields):
    """"Groups fields of lines from 'loadLines' into coversations base on 'movie_conversations.txt' """
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            #Extract fields
            convObj = {}
            for i, field in enumerate(fields):
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

