import sklearn.metrics
import numpy as np

def aurocScore(inData, outData):
	allData = np.concatenate((inData, outData))
	labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = 0)  
	return fpr, tpr, sklearn.metrics.auc(fpr, tpr)

def tprAtFpr(tpr, fpr, fprRate = 0.05):
	fprAdjust = np.abs(np.array(fpr)-fprRate)
	fprIdx = np.argmin(fprAdjust)
	tpratfpr = tpr[fprIdx]

	return tpratfpr, fpr[fprIdx]

def minUE(inData, outData):
	allData = np.concatenate((inData, outData))

	thresholds = np.sort(allData)
	uncertainty_error = []
	#assuming that lower means more uncertain, less confident
	for t in thresholds:
		#number of in detections that are rejected incorrectly
		in_error = np.sum(inData < t)/len(inData)
		#number of out detections that are accepted incorrectly
		out_error = np.sum(outData >= t)/len(outData)

		ue = 0.5*in_error + 0.5*out_error
		uncertainty_error += [ue]
	
	return np.min(uncertainty_error), uncertainty_error
		
def summarise_performance(inData, outData, fprRates = [], printRes = True, methodName = ''):
	results = {}

	fpr, tpr, auroc = aurocScore(inData, outData)
	results['auroc'] = auroc
	results['fpr'] = list(fpr)
	results['tpr'] = list(tpr)

	specPoints = []
	for fprRate in fprRates:
		tprRate = tprAtFpr(tpr, fpr, fprRate)
		specPoints += [tprRate]

		results[f'tpr at fprRate {fprRate}'] = tprRate
	
	if printRes:
		print(f'Results for Method: {methodName}')
		print(f'------ AUROC: {round(auroc, 3)}')
		for point in specPoints:
			fp = point[1]
			tp = point[0]
			print(f'------ TPR at {round((100.*fp), 1)}FPR: {round((100.*tp), 1)}')

	return results