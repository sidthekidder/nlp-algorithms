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
		# check if word in vocabulary, if not replace with unk
		if word not in self.vocabulary:
			word = '<unk>'

		# check if all strings in context are in vocabulary, if not replace with unk
		def context_checker(model, context_var):
			if len(context_var) == 0:
				return context_var

			# base case when context_var is (str, str)
			if isinstance(context_var, str):
				if context_var not in model.vocabulary:
					return '<unk>'
				else:
					return context_var

			# case when context_var is (str, (nested tuple))
			if context_var[0] not in model.vocabulary:
				new_context = ('<unk>', context_checker(model, context_var[1]))
			else:
				new_context = (context_var[0], context_checker(model, context_var[1]))
			return new_context
		context = context_checker(self, context)

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

# return corpus where <unk> replaces all unique words
def mask_rare(corpus):
	# check input must be list of strings
	if not isinstance(corpus, list):
		print("Corpus must be of type list!")
		exit()

	# dict to hold the word frequencies
	word_counts = {}

	# create a single joint string to sum frequencies
	corpus_split = " ".join(corpus).split()

	# sum frequencies of each word
	for x in corpus_split:
		try:
			if x in word_counts:
				word_counts[str(x)] += 1
			else:
				word_counts[str(x)] = 1
		except:
			continue

	# build up corpus again, checking for 1 frequency words and replacing them with <unk>
	new_corpus = []
	for sentence in corpus:
		new_sentence = ''
		for word in sentence.split():
			try:
				if word_counts[str(word)] == 1:
					new_sentence += " <unk>"
				else:
					new_sentence += " " + word
			except:
				continue
		new_corpus.append(new_sentence)

	return new_corpus

# create NgramModel model reading from corpus_path
def create_ngramlm(n, corpus_path, delta=0):
	with open(corpus_path, 'r', encoding='utf-8-sig') as file:
		contents = file.read().splitlines()

	contents = mask_rare(contents)

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


class NGramInterpolator:
	def __init__(self, n, lambdas):
		self.n = n
		self.lambdas = lambdas
		self.ngrams = []
		for i in range(n):
			self.ngrams.append(NgramModel(i+1))
		# print("***\n")
		# print(self.lambdas)
		# print(self.ngrams)

	def update(self, text):
		for i in range(len(self.ngrams)):
			self.ngrams[i].update(text)

	def word_prob(self, word, context):
		final_probability = 0
		# print("***\n")
		# print(self.lambdas)
		# print(self.ngrams)
		for i in range(self.n):
			final_probability += self.lambdas[i] * self.ngrams[i].word_prob(word, context)
		return final_probability

def ngi_prob(ngi, text):
	# create a single list containing all ngrams - uni, bi, tri etc
	ngrams = []
	for i in range(1, ngi.n + 1):
		ngrams = ngrams + get_ngrams(i, [text])

	# get probabilities of each ngram in the sentence
	probabilities = [ngi.word_prob(x[0], x[1]) for x in ngrams]

	# convert to log probability and multiply together
	probability = np.sum([math.log(x) for x in probabilities])

	return probability

# returns perplexity of a trained model on the data from corpus_path
def perplexity(model, corpus_path):
	with open(corpus_path, 'r', encoding='utf-8-sig') as file:
		contents = file.read().splitlines()

	# count total number of tokens in text data
	N = len(" ".join(contents).split())
	print("N = " + str(N))

	# calculate perplexity using formula
	l = (1.0 / N) * np.sum([text_prob(model, i) for i in contents])
	# print('l = ' + str(l))
	return math.pow(math.e, -l)
