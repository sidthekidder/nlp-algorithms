import time
import math
import numpy as np

def get_ngrams(n, text):
	# n: size of ngram
	# text: list of words/strings
	# output: n-gram tuples of (word, context)
	# word is a string
	# context is tuple of the n-1 preceding words

	# if unigram required then just split each word and return
	if n == 1:
		return [(x, ()) for x in ' '.join(text).split(' ')]
	
	# add <s> </s> tokens to text
	# also number of start tokens = n-1 so that first n-1 words can be part of n-gram
	# eg this is a fat cat, n = 5 -> <s> <s> <s> <s> this is a fat cat </s>
	text = ['<s> '*(n-1) + x + ' </s>'*(n-1) for x in text]

	precedingNgrams = get_ngrams(1, text)

	# run loop n-1 times, each time updating a tuple's 2nd element with previous index tuple
	while n > 1:
		n -= 1
		newList = []
		for x in range(1, len(precedingNgrams)):
			if precedingNgrams[x-1][1] == (): # if 2nd element is empty, then skip it
				newList.append((precedingNgrams[x][0], precedingNgrams[x-1][0]))
			else:
				newList.append((precedingNgrams[x][0], precedingNgrams[x-1]))
		precedingNgrams = newList
	return precedingNgrams

class NgramModel:
	def __init__(self, n, delta=0):
		self.n = n
		self.ngram_counts = {}
		self.context_counts = {}
		self.vocabulary = set(['<s>', '</s>'])
		self.delta = delta

	def update(self, text):
		ngram_list = get_ngrams(self.n, text)

		# get dictionary of counts from list
		for idx, ngr in enumerate(ngram_list):
			if ngr not in self.ngram_counts:
				self.ngram_counts[ngr] = 1
			else:
				self.ngram_counts[ngr] += 1

		# if it's not a unigram model
		if self.n > 1:
			# store 2nd elements (contexts) of all ngram tuples in a set
			context_list = [x[1] for x in ngram_list]
		else:
			# store context same as ngrams
			context_list = ngram_list

		# get dictionary of counts from list
		for idx, ctx in enumerate(context_list):
			if ctx not in self.context_counts:
				self.context_counts[ctx] = 1
			else:
				self.context_counts[ctx] += 1

		# store vocabulary as set of all words
		self.vocabulary = self.vocabulary.union(set(' '.join(text).split(' ')))

	def word_prob(self, word, context):
		numerator = self.delta
		if (word, context) in self.ngram_counts:
			# print((word, context))
			numerator += self.ngram_counts[(word, context)]

		denominator = self.delta * len(self.vocabulary)
		if context in self.context_counts:
			denominator += self.context_counts[context]

		if denominator == 0:
			return 1.0 / len(self.vocabulary)

		return (float(numerator) / float(denominator))

	def random_word(self, context):
		vocab = list(self.vocabulary)[1:]
		vocab.sort()
		r = random.random()

		# represents the number line of probabilities
		probabilities = []

		accumulator = 0
		for idx, word in enumerate(vocab):
			cur_probability = self.word_prob(word, context)
			probabilities.append({'word': word, 'value': accumulator + cur_probability})
			accumulator += cur_probability

		for i in range(len(probabilities) - 1):
			if probabilities[i]['value'] <= r and probabilities[i+1]['value'] >= r:
				return probabilities[i]['word']
		return probabilities[-1]['word'] # last word

	def likeliest_word(self, context):
		vocab = list(self.vocabulary)[1:]
		vocab.sort()

		# represents all probabilities
		probabilities = []
		for idx, word in enumerate(vocab):
			cur_prob = self.word_prob(word, context)
			if cur_prob > 0:
				probabilities.append({'word': word, 'value': cur_prob})

		# sample a element with probability proportional to its value
		max_prob = max(probabilities, key=lambda x: x['value'])
		maxes = [x['word'] for idx, x in enumerate(probabilities) if x['value'] == max_prob['value']]
		
		return random.choice(maxes)

# create NgramModel model reading from corpus_path
def init_ngram_model(n, corpus_path, delta=0):
	with open(corpus_path, 'r', encoding='utf-8-sig') as file:
		contents = file.read().splitlines()

	# create the ngram object and update contents
	ngram = NgramModel(n, delta)
	ngram.update(contents)

	return ngram

# predict probability for a sentence
def text_prob(model, text):
	if not isinstance(text, str):
		print("Text must be of type string!")
		exit()

	# get ngrams for text input
	ngrams = get_ngrams(model.n, [text])

	# get probabilities of each ngram in the sentence
	probabilities = [model.word_prob(x[0], x[1]) for x in ngrams]

	# convert to log probability and multiply together
	probability = np.sum([math.log(x) for x in probabilities])
	return probability


import random

def random_text(model, max_length, delta=0):
	finalstr = ''

	def create_ngram_padding(n, curContext):
		if n == 1:
			return curContext
		return create_ngram_padding(n-1, ('<s>', curContext))
	curToken = create_ngram_padding(model.n-1, ('<s>'))

	for i in range(max_length): # generate max_length words
		nextToken = model.random_word(curToken)

		if nextToken == '</s>':
			return finalstr
		finalstr += ' ' + nextToken

		# build up the next context using current context's 2nd part
		# (t1, t2) -> move outwards => (newtok, t1)
		# (t1, (t2, t3)) -> (newtok, (t1, t2))
		# (t1, (t2, (t3, t4))) -> (newtok, (t1, (t2, t3)))
		def create_next_context(curContext, new_word):
			# reached base case - simple tuple
			if isinstance(curContext[1], str):
				return (new_word, curContext[0])
			ret = create_next_context(curContext[1], curContext[0])
			return (new_word, ret)

		curToken = create_next_context(curToken, nextToken)

	return finalstr

def likeliest_text(model, max_length, delta=0):
	finalstr = ''

	# create start context with same nesting level as model's n-gram
	def create_ngram_padding(n, curContext):
		if n == 1:
			return curContext
		return create_ngram_padding(n-1, ('<s>', curContext))
	curToken = create_ngram_padding(model.n-1, ('<s>'))

	# print(curToken)
	for i in range(max_length): # generate max_length words
		nextToken = model.likeliest_word(curToken)

		# end if stop token generated
		if nextToken == '</s>':
			# if generated string is too short, can call this function again
			return finalstr
		finalstr += ' ' + nextToken

		# build up the next context using current context's 2nd part
		# (t1, t2) -> move outwards => (t2, newtok)
		# (t1, (t2, t3)) -> (t2, (t3, newtok))
		# (t1, (t2, (t3, t4))) -> (t2, (t3, (t4, newtok)))
		def create_next_context(curContext, new_word):
			# reached base case - simple tuple
			if isinstance(curContext[1], str):
				return (new_word, curContext[0])
			ret = create_next_context(curContext[1], curContext[0])
			return (new_word, ret)

		curToken = create_next_context(curToken, nextToken)

	return finalstr

# beam search
def beam_search_text(model, max_length, k):
	finalstr = ''

	# helper function to create new context when we get a new word
	def create_next_context(curContext, new_word):
		# reached base case - simple tuple
		if isinstance(curContext[1], str):
			return (new_word, curContext[0])
		ret = create_next_context(curContext[1], curContext[0])
		return (new_word, ret)

	# create start context with same nesting level as model's n-gram
	def create_ngram_padding(n, curContext):
		if n == 1:
			return curContext
		return create_ngram_padding(n-1, ('<s>', curContext))
	curToken = create_ngram_padding(model.n-1, ('<s>'))
	
	# initialize beam state with single start token that has joint probability 1
	beam_state = [{'curToken': curToken, 'pathTillNow': '', 'probability': 1}]

	# run a bfs-style loop that processes one sequence at a time
	# recalculated k tokens from the current token, adds them to beam state
	# with joint probability as multiplied probabilities so far
	# then prunes the beam state to top k sequences in terms of probability
	while len(beam_state) > 0:
		beam_state.sort(key=lambda x: x['probability'], reverse=True)

		# select the next sequence to consider processing
		cur_seq = beam_state[0]
		beam_state = beam_state[1:]

		# check if </s> generated or max length reached
		cur_seq_word = cur_seq['pathTillNow'].split(' ')[-1]

		if len(cur_seq['pathTillNow'].split(' ')) >= max_length:
			return cur_seq['pathTillNow']

		# add new states from cur_seq to the current beam state
		vocab = list(model.vocabulary)
		vocab.sort()
		probabilities = []
		for idx, word in enumerate(vocab):
			probb = model.word_prob(word, cur_seq['curToken'])
			if probb > 0 and word != '</s>': # filter </s> token
				probabilities.append({'word': word, 'probability': probb})

		# sort probabilities by highest probability in descending order
		probabilities.sort(key=lambda x: x['probability'], reverse=True)

		# pick top k probability sequences that were generated from current token
		new_sequences = []
		for idx, ele in enumerate(probabilities[:k]):
			new_sequences.append({
				'curToken': create_next_context(cur_seq['curToken'], ele['word']), 
				'pathTillNow': cur_seq['pathTillNow'] + ' ' + ele['word'],
				'probability': cur_seq['probability'] * ele['probability']
			})

		# add elements from new_sequences to current beam state
		beam_state += new_sequences

		# sort and prune current beam state to top k entries
		beam_state.sort(key=lambda x: x['probability'], reverse=True)
		beam_state = beam_state[:k]

	return finalstr

