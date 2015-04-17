#!/public/spark-1.3.0/bin/pyspark
import os
import sys
import numpy as np

# Path for spark source folder
os.environ['SPARK_HOME'] = "/Users/jinsub/Desktop/ML_Big_Datasets/spark-1.3.0-bin-cdh4"

# Append pyspark  to Python Path
sys.path.append("/Users/jinsub/Desktop/ML_Big_Datasets/spark-1.3.0-bin-cdh4/python")

from pyspark import SparkContext

numFactors = int(sys.argv[1])
numWorkers = int(sys.argv[2])
numIterations = int(sys.argv[3])
BETA = float(sys.argv[4])
LAMBDA = float(sys.argv[5])
inputVFilePath = sys.argv[6]
outputWFilePath = sys.argv[7]
outputHFilePath = sys.argv[8]
TAU = 100.0
EPSILON = (TAU + 1) ** (-BETA)
Wfile = open(outputWFilePath, 'w')
Hfile = open(outputHFilePath, 'w')



def initialiseWH(Vrdd):
    numRows = int(Vrdd.values().max()[0])
    colValues = Vrdd.values().map(lambda x: x[1])
    numCols = int(colValues.max())

    # create factor matrix W with row number as key
    noIndexW = np.random.uniform(0.1, 10, (500, numFactors))

    indexW = [(x, ("W", noIndexW[x])) for x in range(len(noIndexW))]
    del noIndexW
    initialWrdd = sc.parallelize(indexW)
    del indexW

    # create factor matrix H with col number as key
    noIndexH = np.random.uniform(0.1, 10, (500, numFactors))
    indexH = [(x, ("H", noIndexH[x])) for x in range(len(noIndexH))]
    del noIndexH
    initialHrdd = sc.parallelize(indexH)
    del indexH

    return initialWrdd, initialHrdd

#parser used to read directory of netflix files - put to triple format
def wholeTextParser(contentTuple):
    contentTuple = contentTuple[1].split("\n")
    movieID = contentTuple[0].split(":")[0]
    del contentTuple[0]
    content = [x.split("|") for x in contentTuple]
    result = []
    for data in content:
        if len(data) == 3:
            userID = int(data[0])
            result.append((userID % numWorkers, (data[0], movieID, data[1])))
    return result

#performs sgd - passed to mapPartitions
def sgd(iterator):
    Vij = []
    Wi = []
    Hj = []
    #stored needed elements in memory
    for tuple in iterator:
        for triple in tuple[1]:
            for element in triple:
                if element[0] == "W":
                    Wi.append(element[1])
                elif element[0] == "H":
                    Hj.append(element[1])
                else:
                    Vij.append(element)
    WUpdates = []
    HUpdates = []
    # do math
    for dataIndex in range(len(Vij)):
        WUpdates.append((int(Vij[dataIndex][0]), EPSILON * -2.0 * (int(Vij[dataIndex][2]) - np.dot(Wi[dataIndex % len(Wi)],
                                                                                              Hj[dataIndex % len(Hj)]
                                                                                              )) * Hj[dataIndex % len(Hj)] + 2.0 *
                         LAMBDA/len(Vij) * Wi[dataIndex % len(Wi)]))
        HUpdates.append((int(Vij[dataIndex][1]), EPSILON * -2.0 * (int(Vij[dataIndex][2]) - np.dot(Wi[dataIndex % len(Wi)],
                                                                                        Hj[dataIndex % len(Hj)]
                                                                                                   )) * Wi[dataIndex % len(Wi)] + 2.0 *
                         LAMBDA/len(Vij) * Hj[dataIndex % len(Hj)]))

    return WUpdates, HUpdates

#Loss function
def calcLoss(iterator):
    Vij = []
    Wi = []
    Hj = []
    for tuple in iterator:
        for triple in tuple[1]:
            for element in triple:
                if element[0] == "W":
                    Wi.append(element[1])
                elif element[0] == "H":
                    Hj.append(element[1])
                else:
                    Vij.append(element)
    # return Wi
    loss = []
    for dataIndex in range(len(Vij)):
        loss.append((int(Vij[dataIndex][2]) - np.dot(Wi[dataIndex % len(Wi)], Hj[dataIndex % len(Hj)])) ** 2)

    return [sum(loss)]


def performUpdates(numIterations, V, W, H):
    Woriginal = W.map(lambda x: x)
    Horiginal = H.map(lambda x: x)
    for t in range(numIterations):
        global EPSILON
        EPSILON = (TAU + t) ** (-BETA)
        for stratumNumber in range(numWorkers):
            W = Woriginal.map(lambda x: (x[0] % numWorkers, x[1]))
            H = Horiginal.map(lambda x: (x[0] % numWorkers, x[1]))
            currStratum = V.filter(lambda x: (abs(int(x[0]) - int(x[1][1])) % numWorkers) == stratumNumber)
            grouped = currStratum.groupWith(W, H)
            partitioned = grouped.partitionBy(numWorkers, lambda x: x)
            updates = partitioned.mapPartitions(sgd)

            def getW(input):
                if len(input) > 0:
                    return input[0]
                else: return (0,np.array([0]))
            def getH(input):
                if len(input) > 1:
                    return input[1]
                else: return (0,np.array([0]))
            WUpdates = updates.map(getW)
            HUpdates = updates.map(getH)

            # sum up np.array updates - ignore the "W" or "H" tag in values tuple
            def sumUpdates(inputValues):
                total = 0
                for element in inputValues:
                    if element == "W" or element == "H":
                        label = element
                    if type(element) == np.ndarray:
                        total += element
                return (label, total)

            Woriginal = Woriginal.mapValues(sumUpdates)
            Horiginal = Horiginal.mapValues(sumUpdates)
    return Woriginal, Horiginal



sc = SparkContext('local')
if os.path.isdir(inputVFilePath):
    rdd = sc.wholeTextFiles(inputVFilePath)
    Vrdd = rdd.flatMap(wholeTextParser)

    Wrdd = initialiseWH(Vrdd)
    Hrdd = initialiseWH(Vrdd)

    print Vrdd.collect()

elif os.path.isfile(inputVFilePath):
    rddLines = sc.textFile(inputVFilePath)
    rddLines = rddLines.map(lambda x: x.split(","))

    Vrdd = rddLines.map(lambda x: ((int(x[0]) % numWorkers), (x[0], x[1], x[2])))

    # initialise W and H
    initialisedFactors = initialiseWH(Vrdd)
    Wrdd = initialisedFactors[0]
    Hrdd = initialisedFactors[1]

    # performed dsgd - collect W and H
    updated = performUpdates(numIterations, Vrdd, Wrdd, Hrdd)
    factoredW = updated[0]
    factoredH = updated[1]

    #unwrap W and H to output as CSV
    Wout = factoredW.map(lambda x: x[1][1:len(x[1])])
    Wout = np.asarray(Wout.collect())
    Wout = np.asarray([item for sublist in Wout for item in sublist])
    np.savetxt(Wfile, Wout, delimiter=",")

    Hout = factoredH.map(lambda x: x[1][1:len(x[1])])
    Hout = np.asarray(Hout.collect())
    Hout = np.asarray([item for sublist in Hout for item in sublist])
    Hout = np.transpose(Hout)
    np.savetxt(Hfile, Hout, delimiter=",")



# W, H, V, all RDDs
# assign blocks as long as row and cols are independent, the set on a diagonal is a stratum
# RDD for a strata.
# partitionBy() function to define STRATA , mapPartitions() to operate on strata to give UPDATES to W and H via JOIN...


#sgd function returns list of tuples feed this to mapPartitions - return w,h = ([(row,np.array),()],[(col,np.array),()]) - then map
#join
# groupwith()
# partitionby()
# mappartitions(sgd)
# sgd works on a block
# map to get back to w and h

