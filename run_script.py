import numpy as np
import os
from tqdm.auto import tqdm
import argparse

from nltk.translate.meteor_score import meteor_score
# Create the parser
parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
parser.add_argument('train_path',
                       metavar='path',
                       type=str,
                       help='the path to the training text')

parser.add_argument('test_path',
                       metavar='path',
                       type=str,
                       help='the path to the testing text')

parser.add_argument('nlp_path',
                       metavar='path',
                       type=str,
                       help='the path to the testing text')

args = parser.parse_args()

train_name = args.train_path
test_name = args.test_path
nlp_path = args.nlp_path

# stanford nlp
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(nlp_path)

trainval_x = []
trainval_y = []

test_x = []
test_y = []


with open(train_name, 'r') as file1:
    paras = file1.read().split('\n\n')
    
    for para in tqdm(paras):
        items = para.split('\n')

        chn, ref, cand, bleu, label = items[0], items[1], items[2], float(items[3]), items[4]

        # Labels
        if label == 'M':
        	trainval_y.append(0)
        else:
        	trainval_y.append(1)

        # Features

        meteor = meteor_score(ref, cand)

        # sentence length
        n = len(cand.split(' '))
        n_ref = len(ref.split(' '))

        cand_tree = nlp.parse(cand)
        n_s = cand_tree.count('S')/n
        n_np = cand_tree.count('NP')/n
        n_nn = cand_tree.count('NN')/n
        n_vp = cand_tree.count('VP')/n
        n_sbar = cand_tree.count('SBAR')/n

        n_dt = cand_tree.count('DT')/n
        n_jj = cand_tree.count('JJ')/n

        trainval_x.append([n/n_ref, bleu, meteor, n_s, n_np, n_nn, n_vp, n_sbar, n_dt, n_jj])
 

with open(test_name, 'r') as file2:
    paras = file2.read().split('\n\n')
    
    for para in tqdm(paras):
        items = para.split('\n')

        chn, ref, cand, bleu, label = items[0], items[1], items[2], float(items[3]), items[4]

        ## Labels
        if label == 'M':
        	test_y.append(0)
        else:
        	test_y.append(1)

        ## Features
        meteor = meteor_score(ref, cand)

        # sentence length
        n = len(cand.split(' '))
        n_ref = len(ref.split(' '))

        # parsing tree features
        cand_tree = nlp.parse(cand)
        n_s = cand_tree.count('S')/n
        n_np = cand_tree.count('NP')/n
        n_nn = cand_tree.count('NN')/n
        n_vp = cand_tree.count('VP')/n
        n_sbar = cand_tree.count('SBAR')/n

        n_dt = cand_tree.count('DT')/n
        n_jj = cand_tree.count('JJ')/n


        test_x.append([n/n_ref, bleu, meteor, n_s, n_np, n_nn, n_vp, n_sbar, n_dt, n_jj])

nlp.close()


trainval_x = np.array(trainval_x)
trainval_y = np.array(trainval_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

# Normalize
from sklearn.preprocessing import normalize
trainval_x = normalize(trainval_x)
test_x = normalize(test_x)

## train val split
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(trainval_x, trainval_y, test_size=0.2, random_state = 100, shuffle = True)

## Build SVM
from sklearn.svm import SVC
from sklearn.metrics import f1_score

best_acc = float('-inf')
best_f1 = float('-inf')
best_clf = None

# hyperparameter tuning: regularization (C) and gamma
C = [0.1, 0.2, 0.5, 1, 10, 20, 50, 100, 200, 500, 1000]

G = ['auto', 'scale']

for g in G:

	for c in C:

		clf = SVC(C = c, kernel = 'rbf', gamma = g)
		clf.fit(train_x, train_y)

		pred  = clf.predict(val_x)
		f1 = f1_score(val_y, pred)

		# print('train acc: {:.3f}, val_acc: {:.3f}, val_f1: {:.3f}'.format(clf.score(train_x, train_y), clf.score(val_x, val_y), f1))

		if f1 >= best_acc:
			best_acc = f1
			best_C = c
			best_g = g
			best_clf = clf

print('---')		
print('Training Accuracy', best_clf.score(train_x, train_y))
print('Training F1', f1_score(train_y, best_clf.predict(train_x)))

print('---')		
print('Validation Accuracy', best_clf.score(val_x, val_y))
print('Validation F1', f1_score(val_y, best_clf.predict(val_x)))


## Test on the test dataset

final_clf = SVC(C = best_C, kernel = 'rbf', gamma = best_g)
final_clf.fit(trainval_x, trainval_y)
print('---')
print('Test Accuracy:', final_clf.score(test_x, test_y))
print('Test F1: ', f1_score(test_y, final_clf.predict(test_x)))


