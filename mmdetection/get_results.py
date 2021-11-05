import json
import numpy as np
import torch
import sklearn.mixture as sm
import argparse
import scipy.stats as st
from performance_metrics import *
from utils import fit_gmms, gmm_uncertainty
import tqdm
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *


def parse_args():
	parser = argparse.ArgumentParser(description='Test with Distance')
	parser.add_argument('dType', default = 'FRCNN', help='FRCNN or retinanet')
	parser.add_argument('--dataset', default = 'voc', help='voc or coco')
	parser.add_argument('--unc', default = 'all', help='how to measure uncertainty? score, entropy, gmm, simple or all?')
	parser.add_argument('--saveNm', default = None, help='what is the save name of the results?')
	parser.add_argument('--iouThresh', default = 0.6, type = float, help='what is the cutoff iou for logits used to estimate class centres?')
	parser.add_argument('--scoreThresh', default = 0.7, type = float, help='what is the cutoff score for logits used to estimate class centres?')
	parser.add_argument('--numComp', default = None, type = int, help='do you want to test with a specific number of GMM components?')
	parser.add_argument('--saveResults', default = False, type = bool, help='do you want to save results? Only works with args.unc == all')
	args = parser.parse_args()
	return args

args = parse_args()

results_dir = f'{BASE_RESULTS_FOLDER}/{args.dType}/associated/{args.dataset}'

if args.unc == 'all':
	uncTypes = ['score', 'entropy', 'simple', 'gmm']
else:
	uncTypes = [args.unc]

if args.dataset == 'voc':
	num_classes = 15
elif args.dataset == 'coco':
	num_classes = 50
else:
	num_classes = 80
	print('implement')
	exit()

#load in the test data
with open(f'{results_dir}/test/{args.saveNm}.json', 'r') as f:
	testData = json.load(f)

testType = np.asarray(testData['type'])
testLogits = np.asarray(testData['logits'])
testScores = np.asarray(testData['scores'])
testIoUs = np.asarray(testData['ious'])

#we want results in terms of AUROC, and TPR at 5%, 10% and 20% FPR
fprRates = [0.05, 0.1, 0.2]


allResults = {}
for unc in uncTypes:
	if unc == 'score':
		#correctly classified detections of known objects
		tpKnown = testScores[testType == 0]
		
		#open-set errors
		fpUnknown = testScores[testType == 2]

	
	elif unc == 'entropy':
		#faster r-cnn uses softmax
		if args.dType == 'FRCNN':
			softmaxLayer = torch.nn.Softmax(dim = 1)
			tensorLogits = torch.Tensor(testLogits)
			softmaxScores = softmaxLayer(tensorLogits).cpu().detach().tolist()

		#retinanet uses sigmoid
		else:
			print('implement')
			exit()
		
		entropy = st.entropy(softmaxScores, axis = 1)
		entropy = np.asarray(entropy)

		#for entropy, a higher score means greater uncertainty. therefore we use the negative entropy, so that a lower score means greater uncertainty
		tpKnown = -entropy[testType == 0]
		fpUnknown = -entropy[testType == 2]

	#load in training and val logits for distance-based measures
	elif unc == 'simple' or unc == 'gmm':
		with open(f'{results_dir}/train/{args.saveNm}.json', 'r') as f:
			trainData = json.load(f)

		with open(f'{results_dir}/val/{args.saveNm}.json', 'r') as f:
			valData = json.load(f)

		trainLogits = np.array(trainData['logits'])
		trainLabels = np.array(trainData['labels'])
		trainScores = np.array(trainData['scores'])
		trainIoUs = np.array(trainData['ious'])

		valLogits = np.array(valData['logits'])
		valTypes = np.array(valData['type'])

		#fit distance-based models
		if unc == 'gmm':
			#find the number of components that gives best performance on validation data, unless numComp argument specified
			if args.numComp != None:
				gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = args.numComp)
			else:
				allAurocs = []
				nComps = [nI for nI in range(3, 16)]
				print('Finding optimal component number for the GMM')
				for nComp in tqdm.tqdm(nComps, total = len(nComps)):
					gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = nComp)

					gmmScores = gmm_uncertainty(valLogits, gmms)
					valTP = gmmScores[valTypes == 0]
					valFP = gmmScores[valTypes == 1]
					_, _, auroc = aurocScore(valTP, valFP)
					allAurocs += [auroc]

				allAurocs = np.array(allAurocs)
				bestIdx = np.argmax(allAurocs)
				preferredComp = nComps[bestIdx]

				print(f'Testing GMM with {preferredComp} optimal components')
				gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = preferredComp)

		else:
			gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = 1, covariance_type = 'spherical')
		
		gmmScores = gmm_uncertainty(testLogits, gmms)
		tpKnown = gmmScores[testType == 0]
		fpUnknown = gmmScores[testType == 2]

	else:
		print('That uncertainty measure has not been implemented. Check the args.unc input argument.')
		exit()
	
	scoreResults = summarise_performance(tpKnown, fpUnknown, fprRates, True, args.saveNm + f' with uncertainty {unc}')
	allResults[unc] = scoreResults

if args.saveResults and args.unc == 'all':
	final_dir = f'../results/{args.dType}/final/{args.dataset}/'	
	if not os.path.exists(final_dir):
		os.makedirs(final_dir)
	with open(f'{final_dir}{args.saveNm}.json', 'w') as outFile:
		json.dump(allResults, outFile)
	