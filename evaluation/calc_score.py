from nltk.translate.bleu_score import corpus_bleu
import sys
import string
from nltk.tokenize import word_tokenize


p = '../GYAFC_Corpus/Entertainment_Music/test/'
reference0 = open(p+'formal.ref0', 'r').readlines()
reference1 = open(p+'formal.ref1', 'r').readlines()
reference2 = open(p+'formal.ref2', 'r').readlines()
reference3 = open(p+'formal.ref3', 'r').readlines()
candidate = open(sys.argv[1], 'r').readlines()

k = len(candidate)
if (len(sys.argv) < 3):

	references =[]
	candidates = []

	for i in range(k):
		candidates.append(word_tokenize(candidate[i].strip()))
		references.append([word_tokenize(reference0[i].strip()), word_tokenize(reference1[i].strip()), word_tokenize(reference2[i].strip()), word_tokenize(reference3[i].strip())])
	score = corpus_bleu(references, candidates)
	print("The bleu score is: "+str(score))
