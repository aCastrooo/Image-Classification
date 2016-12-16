import math
import util
import random
import math
import operator

PRINT = True

class NNClassifier:
	"""
	Perceptron classifier.

	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	"""
	def __init__(self, legalLabels):
		self.legalLabels = legalLabels
		self.type = "nearestneighbors"
		self.weights = {}
		for label in legalLabels:
			self.weights[label] = util.Counter() # this is the data-structure you should use



	def train(self, trainingData, trainingLabels, testData, testLabels, k_number_of_neighbors):
		"""
		predictions = []
		for x in range(len(trainingData)):
			neighbors = self.getNeighbors(trainingData, testData[x], k_number_of_neighbors)
			result = self.getLabels(neighbors)
			predictions.append(result)


			#acc = self.accuracy(testData, predictions)
			#print "Adfadfasf   " + acc + "%"
			#accuracy = self.accuracy(trainingData, predictions)
			#print "Accuracy: " + repr(accuracy) + "%"
		"""

		test = []
		predictions = []
		for x in range(len(testData)):
			part = list(testData[x].values())
			test.append(part)


		for x in range(len(test)):
			neighbors = self.getNeighbors(trainingData, trainingLabels, test[x], k_number_of_neighbors)
			predict = self.getPredictions(neighbors)
			predictions.append(predict)
			accurate = self.howAccurate(testLabels, predictions)
			print "Predicted: " + repr(predictions) + " Actual: " + repr(testLabels[0])

		print "Accuracy: " + repr(accurate)




		
		#for x in range(len(neighbors)):
		#	print neighbors[x][2]


		#predictions.append(predict)
		#print repr(testLabels[0])
		#print repr(predictions[0])

		



	def EuclideanDistance(self, dataOne, dataTwo, length):
		distance = 0
		for x in range(length):
			distance += pow((dataOne[x] - dataTwo[x]), 2)
		return math.sqrt(distance)


	def getNeighbors(self, trainingData, trainingLabels, fromTest, k_number_of_neighbors):
		distances = []
		neighbors = []
		neighborLabels = []
		train = []
		for x in range(len(trainingData)):
			part = list(trainingData[x].values())
			train.append(part)
	
		length = len(fromTest)-1
		for x in range(len(trainingData)):
			dist = self.EuclideanDistance(fromTest, train[x], length)
			distances.append((trainingData[x], dist, trainingLabels[x]))
		distances.sort(key=lambda tup: tup[1])

		for x in range(k_number_of_neighbors):
			neighbors.append(distances[x])


		return neighbors

	def getPredictions(self, neighbors):
		labels = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in labels:
				labels[response] += 1
			else:
				labels[response] = 1

		sortedLabels = sorted(labels.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedLabels[0][0]


































	def getResult(self, neighbors):
		labels = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in labels:
				labels[response] += 1
			else:
				labels[response] = 1
		sortedLabels = sorted(labels.iteritems(), key=operator.itemgetter(1), reverse=True)

		return sortedLabels[0][0]


	def howAccurate(self, testLabels, predictions):
		correct = 0

		for x in range(len(predictions)):
			if testLabels[x] == predictions[x]:
				correct += 1

		return (correct / float(len(testLabels))) * 100.0
		


"""























	def euclideanDistance(self, pointOne, pointTwo, length):
				Returns the Euclidean Distance between the two points
				distance = 0
		for x in range(length):
			distance += pow(pointTwo[x] - pointOne[x], 2)
		return math.sqrt(distance)


	def getNeighbors(self, trainingData, point, k):
		Returns k number of neighbors that are the closest to instance
				distances = []
		neighbors = []
		length = len(point)-1
		for x in range(len(trainingData)):
			dist = self.euclideanDistance(point, trainingData[x], length)
			distances.append((trainingData[x], dist))
		distances.sort()
		for x in range(k):
			neighbors.append(distances[x][0])

		return neighbors


	def getLabels(self, neighbors):
		Gets the labels of the surrounding k-neighbors
		labels = {}
		for x in range(len(neighbors)):
			neighborLabel = neighbors[x][-1]
			if neighborLabel in labels:
				labels[neighborLabel] = labels[neighborLabel] + 1
			else:
				labels[neighborLabel] = 1
		sortedLabels = sorted(labels.iteritems(), key=operator.itemgetter(1), reverse=True)

		return sortedLabels[0][0]


	def classify(self, data):
		Classifies each datum as the label that most closely matches the prototype vector
		for that label.  See the project description for details.

		Recall that a datum is a util.counter...
		guesses = []
		for datum in data:
			vectors = util.Counter()
			for l in self.legalLabels:
				vectors[l] = self.weights[l] * datum
			guesses.append(vectors.argMax())
		return guesses


	def accuracy(self, dataSet, predictions):
		correct = 0
		for x in range(len(dataSet)):
			if dataSet[x][-1] == predictions[x]:
				correct += 1

		percent = (correct / float(len(dataSet))) * 100.0
		return percent




		"""