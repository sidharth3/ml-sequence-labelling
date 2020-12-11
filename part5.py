from math import log
from evalResult import eval
import numpy as np
import pandas as pd
import copy
import pickle
dir = './data'
file = open(dir + '/train', "r", encoding='utf8')
lines_in_pairs = [line.rstrip('\n').split(" ") for line in file]

word_tag_pairs = [['__Start__', "Start"]]  # First Start tag

for pair in lines_in_pairs:
	if len(pair) == 1:
		word_tag_pairs.append(['__Stop__', "Stop"])
		word_tag_pairs.append(['__Start__', "Start"])
	else:
		word_tag_pairs.append(pair)

word_tag_pairs = word_tag_pairs[:-1]
word_tag_df = pd.DataFrame(word_tag_pairs)
tags = word_tag_df[1].unique()
vocab = word_tag_df[0].unique()
tagc_dict = word_tag_df[1].value_counts().to_dict()
tags = np.sort(tags)
tags = tags[::-1]


def addCount(parent, child, d):
	# Increment the count of [parent][child] in dictionary d
	if parent in d:
		if child in d[parent]:
			d[parent][child] += 1
		else:
			d[parent][child] = 1
	else:
		d[parent] = {child: 1}

def getDiscriminativeEmissions(file, k=1):
	"""
	input = training file
	output = emission parameters (dict)
	@param k: Words appearing less than k times will be
	replaced with #UNK#
	dict format = {i: {o:emission prob}}
	"""
	emissions = {}
	forward_emissions = {}
	backward_emissions = {}
	forward2_emissions = {}
	backward2_emissions = {}
	words = []
	tags = []
	with open(file, encoding="utf-8") as f:
		for line in f:
			temp = line.strip()

			# ignore empty lines
			if len(temp) == 0:
				continue
			else:
				last_space_index = temp.rfind(" ")
				word = temp[:last_space_index].lower()
				tag = temp[last_space_index + 1:]
				
				words.append(word)
				tags.append(tag)

	for i in range(0, len(words)):
		# update count(y->x)
		word = words[i]
		prev_word = words[i-1] if i > 0 else 'START_WORD'
		prev2_word = words[i-2] if i > 1 else 'START_WORD2'
		next_word = words[i+1] if i < len(words)-1 else 'END_WORD'
		next2_word = words[i+2] if i < len(words)-2 else 'END_WORD2'
		tag = tags[i]

		addCount(word, tag, emissions)
		addCount(prev_word, tag, forward_emissions)
		addCount(next_word, tag, backward_emissions)
		addCount(prev2_word, tag, forward2_emissions)
		addCount(next2_word, tag, backward2_emissions)

	for word, tagCountDict in emissions.items():
		count = sum(tagCountDict.values())
		for tag, tagCount in tagCountDict.items():
			tagCountDict[tag] = tagCount / count
	
	for word, tagCountDict in forward_emissions.items():
		count = sum(tagCountDict.values())
		for tag, tagCount in tagCountDict.items():
			tagCountDict[tag] = tagCount / count
	
	for word, tagCountDict in backward_emissions.items():
		count = sum(tagCountDict.values())
		for tag, tagCount in tagCountDict.items():
			tagCountDict[tag] = tagCount / count
	
	for word, tagCountDict in forward2_emissions.items():
		count = sum(tagCountDict.values())
		for tag, tagCount in tagCountDict.items():
			tagCountDict[tag] = tagCount / count
	
	for word, tagCountDict in backward2_emissions.items():
		count = sum(tagCountDict.values())
		for tag, tagCount in tagCountDict.items():
			tagCountDict[tag] = tagCount / count
	
	unique_tags = set(tags)

	tag_counts = {}

	for tag in tags:
		if tag in tag_counts:
			tag_counts[tag] += 1
		else:
			tag_counts[tag] = 1

	total_count = sum(tag_counts.values())
	for key, count in tag_counts.items():
		tag_counts[key] = count/total_count

	for tag in unique_tags:
		emissions["#UNK#"] = tag_counts
		forward_emissions["#UNK#"] = tag_counts
		backward_emissions["#UNK#"] = tag_counts
		forward2_emissions["#UNK#"] = tag_counts
		backward2_emissions["#UNK#"] = tag_counts

	# replace with unk		

	return emissions, forward_emissions, backward_emissions, forward2_emissions, backward2_emissions, unique_tags, tag_counts


def getTransitions(file):
	start = "_START"
	stop = "_STOP"
	transitions = {}
	count = {start: 0}
	prev = start
	with open(file, encoding="utf-8") as f:
		for line in f:
			temp = line.strip()

			# sentence has ended
			if len(temp) == 0:
				addCount(prev, stop, transitions)
				prev = start

			# part of a sentence
			else:
				last_space_index = temp.rfind(" ")
				curr = temp[last_space_index + 1:]
				# update count(start) if new sentence
				if prev == start:
					count[start] += 1                    

				# update count(y)
				if curr in count:
					count[curr] += 1
				else:
					count[curr] = 1
				
				# update count(prev, curr)
				addCount(prev, curr, transitions)

				prev = curr

		# add count(prev, stop) if no blank lines at EOF
		if prev != start:
			addCount(prev, stop, transitions)
			prev = start

	# convert counts to transitions
	for prev, currDict in transitions.items():
		for curr, currCount in currDict.items():
			currDict[curr] = currCount / float(count[prev])
	return transitions

def convert(file):
	# convert train file to set of unique words
	out = set()
	with open(file, encoding="utf-8") as f:
		for line in f:
			temp = line.strip()

			# ignore empty lines
			if len(temp) == 0:
				continue
			else:
				last_space_index = temp.rfind(" ")
				word = temp[:last_space_index].lower()
				out.add(word)

	return out

def isMissing(child, parent, d):
	# check whether child's parent is parent in given dictionary
	# return (parent not in d) or (child not in d[parent]) or (d[parent][child] == 0)
	return (child not in d[parent]) or (d[parent][child] == 0)

def setHighscores(i, highscore, currTag, parentTag, score):
	if (i) in score:
		score[i][currTag] = [highscore, parentTag]
	else:
		score[i] = {currTag: [highscore, parentTag]}


_parentless_stop = 0
_deep_parentless_count = 0
def discriminativeViterbiAlgo(emissions, forward_emissions, backward_emissions, forward2_emissions, backward2_emissions, transitions, weights, vocab, tags, sentence):
	highscores = {}
	highscores[0] = {"_START": [0.0, None]}
	#highscores[1] is score for sentence[0]
	
	# forward algorithm
	for i in range(len(sentence)):
		word = sentence[i].lower()
		prev_word = sentence[i-1].lower() if i > 1 else 'START_WORD'
		prev2_word = sentence[i-2].lower() if i > 2 else 'START_WORD2'
		next_word = sentence[i+1].lower() if i < len(sentence)-1 else 'END_WORD'
		next2_word = sentence[i+2].lower() if i < len(sentence)-2 else 'END_WORD2'

		# Replace word with #UNK# if not in train
		if word not in vocab:
			word = "#UNK#"
		if prev_word not in vocab:
			prev_word = "#UNK#"
		if next_word not in vocab:
			next_word = "#UNK#"
		if prev2_word not in vocab:
			prev2_word = "#UNK#"
		if next2_word not in vocab:
			next2_word = "#UNK#"

		for currTag in tags:
			highScore = None
			parentTag = None

			if i == 0: #then prevTag only has 1 option: "_START":
				prevScoreParentPair = highscores[0]["_START"]
				if isMissing(currTag, "_START", transitions) or isMissing(currTag, word, emissions):
					setHighscores(i+1, None, currTag, None, highscores)
				else:
					a = transitions["_START"][currTag]
					b = emissions[word][currTag] # if not isMissing(currTag, word, emissions) else 1
					b_forward = forward_emissions[prev_word][currTag] if not isMissing(currTag, prev_word, forward_emissions) else 1
					b_backward = backward_emissions[next_word][currTag] if not isMissing(currTag, next_word, backward_emissions) else 1
					b_forward2 = forward2_emissions[prev2_word][currTag] if not isMissing(currTag, prev2_word, forward2_emissions) else 1
					b_backward2 = backward2_emissions[next2_word][currTag] if not isMissing(currTag, next2_word, backward2_emissions) else 1
					tempScore = prevScoreParentPair[0] * weights[0] + log(a) * weights[1] + log(b) * weights[2] + log(b_forward) * weights[3] + log(b_backward) * weights[4] + log(b_forward2) * weights[5] + log(b_backward2) * weights[6]
					# tempScore = prevScoreParentPair[0] * 0.8 + log(a) * 3 + log(b) * 5.8 + log(b_forward) * 1.5 + log(b_backward) * 0.1 
					highScore = tempScore
					parentTag = "_START"
					setHighscores(i+1, highScore, currTag, parentTag, highscores)

			else: #prevTags can be any of the available tags
				for prevTag in tags:	
					prevScoreParentPair = highscores[i][prevTag]
					# if prev node is disjointed, aka no score
					if prevScoreParentPair[0] == None or isMissing(currTag, prevTag, transitions) or isMissing(currTag, word, emissions):
						#then this prevTag has no path to curTag
						continue
					else:
						a = transitions[prevTag][currTag]
						b = emissions[word][currTag] # if not isMissing(currTag, word, emissions) else 1
						b_forward = forward_emissions[prev_word][currTag] if not isMissing(currTag, prev_word, forward_emissions) else 1
						b_backward = backward_emissions[next_word][currTag] if not isMissing(currTag, next_word, backward_emissions) else 1
						b_forward2 = forward2_emissions[prev2_word][currTag] if not isMissing(currTag, prev2_word, forward2_emissions) else 1
						b_backward2 = backward2_emissions[next2_word][currTag] if not isMissing(currTag, next2_word, backward2_emissions) else 1
						tempScore = prevScoreParentPair[0] * weights[0] + log(a) * weights[1] + log(b) * weights[2] + log(b_forward) * weights[3] + log(b_backward) * weights[4] + log(b_forward2) * weights[5] + log(b_backward2) * weights[6]
						# tempScore = prevScoreParentPair[0] * 0.8 + log(a) * 3 + log(b) * 5.8 + log(b_forward) * 1.5 + log(b_backward) * 0.1 
						if highScore is None or tempScore > highScore:
							highScore = tempScore
							parentTag = prevTag

				if highScore is None:
					#if even after iterating through all possibilities the highscore is none, this means there were no possible paths from the previous node, and so we set this node as disjointed
					#disjointed means, no score and no parent
					setHighscores(i+1, None, currTag, None, highscores)
				else:
					setHighscores(i+1, highScore, currTag, parentTag, highscores)
			
	# _STOP case
	highScore = None
	parentTag = None
	i = len(sentence)
	for prevTag in tags:	
		prevScoreParentPair = highscores[i][prevTag]
		# if prev node is disjointed, aka no score
		if prevScoreParentPair[0] == None or isMissing("_STOP", prevTag, transitions):
			continue
		else:
			prevScoreParentPair = highscores[i][prevTag]
			a = transitions[prevTag]["_STOP"]
			b_forward = forward_emissions[prev_word][currTag] if not isMissing(currTag, prev_word, forward_emissions) else 1
			tempScore = prevScoreParentPair[0] * weights[0] + log(a) * weights[1] + log(b_forward) * weights[3]
			# tempScore = prevScoreParentPair[0] * 0.8 + log(a) * 3 + log(b_forward) * 1.5
			if highScore is None or tempScore > highScore:
				highScore = tempScore
				parentTag = prevTag
	
	if highScore is None:
		#this means there are no possible paths to _STOP		
		setHighscores(i+1, None, "_STOP", None, highscores)
	else:
		setHighscores(i+1, highScore, "_STOP", parentTag, highscores)
	

	prediction = []
	currTag = "_STOP"	
	for i in range(len(sentence)+1, 0, -1): #back to front
		parentTag = highscores[i][currTag][1]
		if parentTag == None:
			global _parentless_stop
			_parentless_stop += 1			
			#this is a disjointed sentence
			#lets choose a parent that has a parent
			candidateHighscore = None
			bestParentCandidateTag = None
			for candidateParentTag in list(highscores[i-1].keys()):
				candidateScoreParentPair = highscores[i-1][candidateParentTag]
				if candidateScoreParentPair[1] == None or candidateScoreParentPair[0] == None:
					continue
				else:
					if candidateHighscore == None or candidateScoreParentPair[0] > candidateHighscore:
						candidateHighscore = candidateScoreParentPair[0]
						bestParentCandidateTag = candidateParentTag

			if bestParentCandidateTag == None:
				global _deep_parentless_count
				_deep_parentless_count += 1
				if list(highscores[i-1].keys())[0] == "_START":
					parentTag = "_START"
				else:
					parentTag = 'O' #defaults to O if no parent because O is the most common tag
			else:
				parentTag = bestParentCandidateTag
		
		if parentTag == "_START":
			break
			
		# print(currTag, parentTag)
		prediction.append(parentTag)
		currTag = parentTag

	prediction.reverse()
	return prediction


def predictWithViterbi(emissions, forward_emissions, backward_emissions, forward2_emissions, backward2_emissions, transitions, weights, vocab, tags,  inputFile, outputFile):
	with open(inputFile) as f, open(outputFile, "w") as out:
		sentence = []

		for line in f:
			# form sentence
			if line != "\n":
				word = line.strip()
				sentence.append(word)

			# predict tag sequence
			else:
				sequence = discriminativeViterbiAlgo(emissions, forward_emissions, backward_emissions, forward2_emissions, backward2_emissions, transitions, weights, vocab, tags,  sentence)
				for i in range(len(sequence)):
					out.write("{} {}\n".format(sentence[i], sequence[i]))
				out.write("\n")
				sentence = []
	print("Prediction Done!")


emissions, forward_emissions, backward_emissions, forward2_emissions, backward2_emissions, tags, tag_counts = getDiscriminativeEmissions(dir + '/train')

transitions = getTransitions(dir + '/train')
vocab = convert(dir + '/train')

weights = [1, 3.3, 6, 1.5, 0.1, 0, 0]
predictWithViterbi(emissions, forward_emissions, backward_emissions, forward2_emissions, backward2_emissions, transitions, weights, vocab, tags, dir + '/dev.in', dir + '/dev.p5.out')
# f = eval(dir + '/dev.out', dir + '/dev.p5.out')