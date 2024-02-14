import math as m
import time as t
import Quartz
import os
from Quartz.CoreGraphics import CGEventCreateKeyboardEvent
from Quartz.CoreGraphics import CGEventCreateMouseEvent
from Quartz.CoreGraphics import CGEventPost
from Quartz.CoreGraphics import kCGEventMouseMoved
from Quartz.CoreGraphics import kCGEventLeftMouseDown
from Quartz.CoreGraphics import kCGEventLeftMouseUp
from Quartz.CoreGraphics import kCGMouseButtonLeft
from Quartz.CoreGraphics import kCGHIDEventTap
from Quartz import CGDisplayBounds
from Quartz import CGMainDisplayID
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plot
import random
from threading import Thread
import matplotlib.image as img
from PIL import Image

# Defining Variables
global MSpeed;
global MDuration;
global MAngle;
global cyclesToComplete

# Inputs: Cycles, Angle, Mouse Speed, Mouse Duration, CyclesToClick
MSpeed = 100
MDuration = 10
MAngle = 43
cyclesToComplete = 10

horizontalSobelOperator = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
verticalSobelOperator = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Defining Features

# Forehead, hairline, eyebrows 1, eyebrows 2, nose, mouth/chin
featuresToTrain = [[[(27, 27), (81, 45)], "greater than", 195, 0.013], # Coordimates, orientation, threshold, say
            [[[(35, 1), (65, 13)], "less than", 40.5, 0.066], [[(35, 23), (65, 33)], "greater than", 193.0, 0.066]],
            [[[(15, 47), (35, 57)], "less than", 136.0, 0.308], [[(15, 37), (35, 47)], "greater than", 159.0, 0.308]],
            [[[(65, 37), (95, 47)], "greater than", 153.0, 0.150], [[(50, 72), (60, 77)], "less than", 166.0, 0.150]],
            [[[(40, 74), (47, 80)], "less than", 134.0, 0.125], [[(50, 72), (60, 77)], "greater than", 166.0, 0.125]],
            [[[(35, 87), (75, 93)], "less than", 170.0, 0.230], [[(35, 102), (75, 107)], "greater than", 182.0, 0.230]]]


features = [[[(27, 27), (81, 45)], "greater than"], # Coordimates, orientationy
            [[[(35, 1), (65, 13)], "less than"], [[(35, 23), (65, 33)], "greater than"]],
            [[[(15, 47), (35, 57)], "less than"], [[(15, 37), (35, 47)], "greater than"]],
            [[[(65, 37), (95, 47)], "greater than"], [[(50, 72), (60, 77)], "less than"]],
            [[[(40, 74), (47, 80)], "less than"], [[(50, 72), (60, 77)], "greater than"]],
            [[[(35, 87), (75, 93)], "less than"], [[(35, 102), (75, 107)], "greater than"]]]

trainingSetForTesting = [ [[True, (202)], [True, (45, 186)], [True, (142, 185)], [True, (183, 197)], [True, (153, 197)], [True, (176, 186)]], [[True, (198)], [True, (39, 200)],
                        [True, (172, 186)], [False, (182, 183)], [True, (175, 183)], [True, (170, 182)]], [[True, (202)], [True, (33, 203)], [True, (118, 177)], [False, (191, 195)],
                        [True, (148, 195)], [True, (168, 192)]], [[True, (195)], [True, (42, 207)], [False, (136, 159)], [False, (153, 166)], [False, (134, 166)], [False, (183, 153)]]]



# Forehead, hairline, eyebrow right, eyebrow left, nose, mouth
# Feature in the form: top left coordinate, bottom right coordinate, orientation. Complex features just are aggregates
# of normal features


# Defining Functions

def Enqueue(value, queue):
    queue.append(value)

    return queue


def Dequeue(queue):
    ReturnValue = queue[0]
    for i in range(len(queue) - 2):
        queue[i] = queue[i + 1]
    del queue[len(queue) - 1]

    return queue, ReturnValue


def factorial(value):
    totalVal = 1
    for i in range(1, value):
        totalVal *= totalVal * i

    return totalVal


def sin(x):
    # First we must convert our value in radians into degrees.
    x = (x / 180) * 3.141592653  # The value of pi
    sum_x = 0
    for i in range(0, 5):  # Further values are negligible
        sum_x += (((-1) ** i) * (x ** ((2 * i) + 1))) / factorial(
            (2 * i) + 1)  # The power series approximation of sin(x)

    return sum_x


def cos(x):
    # First convert x to radians
    x = (x / 180) * 3.141592653
    sum_x = 0
    for i in range(0, 5):
        sum_x += (((-1) ** i) * (x ** (2 * i))) / factorial(2 * i)  # The power series approximation of cos(x)

    return sum_x


def arcsin(x):
    if x == 0:
        return 0
    elif x == 1:
        return 3.141592653 / 2
    elif x < -1 or x > 1:
        print("This value was not within the domain of Arcsin(x)")
        return False
    else:
        theta = 0
        for i in range(0, 5):
            theta += (factorial(2 * i) * (x ** ((2 * i) + 1)) / ((((2 ** i) * factorial(i)) ** 2) + ((2 * i) + 1)))
        # This is the taylor series approximation of Theta for arcsin(x)
        return (theta * 180) / 3.141592653  # converting back into degrees


def arccos(x):
    if x == 1:
        return 3.141592653 / 2
    elif x == 0:
        return 0
    elif x < -1 or x > 1:
        return False
    else:
        theta = 3.141592653 / 2
        for i in range(0, 5):
            theta -= (factorial(2 * i) * (x ** (2 * i) + 1)) / ((4 ** i) * (factorial(i) ** 2) * ((2 * i) + 1))
        # This is the taylor series approximation of Theta for arccos(x)

        return (theta * 180) / 3.141592653


# These are the image related functions


def photoshot():
    camera = cv2.VideoCapture(0)
    frames = 0

    while frames <= 3:
        cap, frame = camera.read()
        frames += 1

    camera.release()
    cv2.destroyAllWindows()

    if cap:
        frame2 = cv2.resize(frame, (780, 360))
        frame3 = frame2.tolist()
        return frame3

    else:
        return False


def ColorToGray(original):
    imHeight = len(original);
    imWidth = len(original[0])
    changed = np.empty([imHeight, imWidth], dtype=np.uint8)

    for i in range(imHeight):

        for j in range(imWidth):
            changed[i][j] = int(
                (original[i][j][0] * 0.2126) + (0.7152 * original[i][j][1]) + (0.0722 * original[i][j][2]))
            # changed[i][j] = int((original[i][j][0] * 0.333) + (0.333 * original[i][j][1]) + ( 0.333* original[i][j][2]))

    return changed


def ColorPurityMap(Image):
    imageHeight = len(Image)
    imageWidth = len(Image[0])
    mapping = np.empty([imageHeight, imageWidth], dtype=np.uint8)
    for i in range(imageHeight):

        for j in range(imageWidth):
            ColorSum = (Image[i][j][0] + Image[i][j][1] + Image[i][j][2])  # The sum of each of the three colors

            if ColorSum == 0:  # In case the value is zero, we don't want an undefined val...
                ColorSum = 0.1

            ColorPurity = (Image[i][j][0] / ColorSum, Image[i][j][1] / ColorSum, Image[i][j][2] / ColorSum)
            #  The proportion of the color sum that each color makes up
            Purity = ((ColorPurity[1] * ColorPurity[2]) - (ColorPurity[0] ** 2)) + (
                    (ColorPurity[1] * ColorPurity[0]) - (ColorPurity[2] ** 2)) + (
                             (ColorPurity[1] * ColorPurity[2]) - (ColorPurity[0] ** 2))

            #  My own formula  I "invented" to find a measure of the difference in each of the ColorPurities
            #  It's the multiple of two of the color purities, minus the square of the third, for each of a,b,c as the
            #  third. It's actually more of a measure of Impurity to be honest.
            #  Then the modulus will be taken, and we'll have some measure of this, with the less the difference between
            #  the three purity values, the less the total purity sum. It shouldn't exceed 1.

            if Purity < 0:
                Purity = Purity * -1
            if Purity == 0:
                Purity = 0.001
            mapping[i][j] = Purity * 100

    return mapping


def integralImage(Im):
    ImHeight = len(Im)
    ImWidth = len(Im[0])
    #integral = np.zeros(0, dtype='int64')
    integral = []
    integral.append([Im[0][0]])

    for i in range(ImWidth - 1):
        integral[0].append(int(integral[0][i]) + int(Im[0][i + 1]))

    for j in range(ImHeight - 1):
        integral.append([])

        for k in range(ImWidth):

            if k != 0:
                integral[j + 1].append(int(integral[j + 1][k - 1]) + int(integral[j][k]) - int(integral[j][k - 1]) + int(Im[j + 1][k]))

            else:
                integral[j + 1].append(int(integral[j][k]) + int(Im[j + 1][k]))

    return integral


def MatrixDotProduct(matrix, SubImage):  # Only takes EVEN and TWO DIMENSIONAL arrays
    product = 0

    if len(matrix) == len(SubImage) and len(matrix[0]) == len(SubImage[0]):
        for i in range(len(matrix)):
            for j in range(len(SubImage[0])):
                product += matrix[i][j] * SubImage[i][j]

    return product


def convolve(kernel, image):
    def reverseKernel(kernel):
        FlippedKernel = []
        ReversedKernel = []

        for i in range(len(kernel)):
            FlippedKernel.append([])
            for j in range(len(kernel[i])):
                FlippedKernel[i].append(kernel[i][len(kernel[i]) - (j + 1)])

        for x in range(len(FlippedKernel)):
            ReversedKernel.append([])
            for y in range(len(FlippedKernel[x])):
                ReversedKernel[x].append(FlippedKernel[len(FlippedKernel) - (x + 1)][y])

        return ReversedKernel

    kernel = reverseKernel(kernel)
    result = []
    index = 0

    for y in range((len(kernel[0]) - 1) // 2, len(image[0]) - ((len(kernel[0]) - 1) // 2)):
        result.append([])

        # I make the assumption that this is a regular kernel, as I do not need any other irregularly shaped kernel, not for edge detection or gausian blur
        for x in range(((len(kernel) - 1) // 2), len(image) - ((len(kernel) - 1) // 2) - 1):
            # result.append([])
            SubImage = []
            for z in range(0, len(kernel)):
                row = []
                for w in range(len(kernel[0])):
                    row.append(image[x - ((len(kernel) - 1) // 2) + z][y - ((len(kernel[0]) - 1) // 2) + w])
                SubImage.append(row)
            result[index].append(int(MatrixDotProduct(kernel, SubImage)))
        index += 1

    return reverseKernel(result)


def GaussianBlur(image, size, extent):
    def generateGaussianKernel(size, sigma):
        center = size // 2
        kernel = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                sides = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(sides ** 2) / (2 * sigma ** 2))  # According to the formula

        return kernel / np.sum(kernel)

    BlurMatrix = generateGaussianKernel(size, extent)

    return convolve(BlurMatrix, image)


def LaplacianConvolution(image):

    LaplacianKernel = [[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]]

    return convolve(LaplacianKernel, image)


def bootstrapDatabase(dataSet):  # The dataset here is a two-dimensional array
    bootstrappedDataSet = [];
    outOfTheBoxDataSet = []

    for i in range(len(dataSet)):
        randomIndex = random.randint(0, len(dataSet) - 1)
        bootstrappedDataSet.append(dataSet[random.randint(0, len(dataSet) - 1)])

    for j in dataSet:
        if j not in bootstrappedDataSet:
            outOfTheBoxDataSet.append(j)

    return bootstrappedDataSet, outOfTheBoxDataSet


'''
def WeightedBootstrapDatabase(datasetWithweights): #Where the weights are the final value
    totalWeight = 0
    weightDistribution = []
    length = len(datasetWithweights[0])
    returnValues = []

    for i in range(len(datasetWithweights)):
        totalWeight += datasetWithweights[length]
        weightDistribution.append(totalWeight)
    for j in range(len(datasetWithweights)):
        randomNum = random.uniform(0, totalWeight[len(totalWeight)])
        if randomNum <= totalWeight[0]:
            returnValues.append(datasetWithweights[0])
        for k in range(len(1, totalWeight)):
            if totalWeight[k] >= randomNum > totalWeight[k-1]:
                returnValues.append(datasetWithweights[k])

    return returnValues
'''


def WeightedBootstrapDatabase(datasetWithweights):
    length = len(datasetWithweights[0])
    newSet = []
    returnValues = []
    totalWeight = 0

    for i in range(len(datasetWithweights)):
        fractionalValue = (datasetWithweights[i][length - 1]).as_integer_ratio()
        roundedValue = round(fractionalValue[1] // (fractionalValue[0])) + 1
        # In case it rounds to 0, but a one bias upwards makes no difference
        for z in range(roundedValue):
            newSet.append(datasetWithweights[i])
        # This creates a database where there are n terms of each datapoint, allowing for easier weighted boostrapping.

    for j in range(len(datasetWithweights)):
        randomisedValue = random.randint(0, len(newSet) - 1)
        returnValues.append(newSet[randomisedValue])

    for k in range(len(returnValues)):
        totalWeight += returnValues[k][length - 1]

    WeightSum = totalWeight ** -1

    for w in range(len(returnValues)):
        returnValues[w][length-1] = returnValues[w][length-1] * WeightSum

    return returnValues


def evaluateIntegralFeature(image, feature):
    # image is integral, and in the format of a 2D array
    # feature is in format, [area specified IN two points of X and Y pairs, threshold, above or below]
    # feature area points should be considered before using the function

    pixelIntensity = image[feature[0][1][1]][feature[0][1][0]]  # Y-2, X-2
    pixelIntensity += image[feature[0][0][1]][feature[0][0][0]]  # Y-1, X-1
    pixelIntensity -= image[feature[0][0][1]][feature[0][1][0]]  # Y-1, X-2
    pixelIntensity -= image[feature[0][1][1]][feature[0][0][0]]  # Y-2, X-1

    if feature[2] == "greater than" and pixelIntensity >= feature[1]:
        return True
    if feature[2] == "less than" and pixelIntensity <= feature[1]:
        return True
    else:
        return False


def EvaluateImage(features, IntegralImage, SubImageDimensions):
    # features should be in the form [ [[[first coordinates, second coordinates], "greater or less than", threshold],  <-- orientation and threshodl are not in the same list
    # [repeat if necessary]] ]
    length = len(IntegralImage)
    width = len(IntegralImage[0])
    rotationCount = 0
    evaluatedFeatures = []
    setOfFaces = []
    for x in range(0, width - (SubImageDimensions[0] + 1), 5):  # Because we only want it to go up to the end of the
        # image, and not repeat after that/
        for y in range(0, length - (SubImageDimensions[1] + 1), 5):
            evaluatedFeatures.append([])
            evaluation = True
            for currentFeature in features:
                if type(currentFeature[1]) == list: # If where the orientation is, there's instead a list
                    counts = 0
                    averagePixelIntensity = []
                    while evaluation and counts <= len(currentFeature) - 1: # We already know it's complex so we evaluate each one  seperately
                        evaluation = True
                        PixelIntensitySum = IntegralImage[currentFeature[counts][0][0][1] + y][currentFeature[counts][0][0][0] + x]
                        PixelIntensitySum += IntegralImage[currentFeature[counts][0][1][1] + y][currentFeature[counts][0][1][0] + x]
                        PixelIntensitySum -= IntegralImage[currentFeature[counts][0][1][1] + y][currentFeature[counts][0][0][0] + x]
                        PixelIntensitySum -= IntegralImage[currentFeature[counts][0][0][1] + y][currentFeature[counts][0][1][0] + x]

                        area = (currentFeature[counts][0][1][0] - currentFeature[counts][0][0][0]) * (
                                currentFeature[counts][0][1][1] - currentFeature[counts][0][0][1])

                        averagePixelIntensity.append(PixelIntensitySum / area)

                        if (currentFeature[counts][1] == "greater than" and averagePixelIntensity[counts] >=
                            currentFeature[counts][2]) or (currentFeature[counts][1] == "less than" and
                                                           averagePixelIntensity[counts] <= currentFeature[counts][2]):
                            pass
                        else:
                            evaluation = False

                        counts += 1


                    evaluatedFeatures[rotationCount].append([evaluation, averagePixelIntensity, [x, y]]) # find a way to either evaluate if this is a face or extract the set of
                    # truths for the features to a list with the coordinates in the same sub list, for further analysis.
                    # ALSO, find the average pixel intensity and return it along with this!

                elif type(currentFeature[1]) == str:
                    # This means the feature is non-complex, so in the form [ [Coordinates set 1, Coordinates set 2],
                    # orientation ]

                    PixelIntensitySum  = IntegralImage[currentFeature[0][0][1] + y][currentFeature[0][0][0] + x]
                    PixelIntensitySum += IntegralImage[currentFeature[0][1][1] + y][currentFeature[0][1][0] + x]
                    PixelIntensitySum -= IntegralImage[currentFeature[0][1][1] + y][currentFeature[0][0][0] + x]
                    PixelIntensitySum -= IntegralImage[currentFeature[0][0][1] + y][currentFeature[0][1][0] + x]

                    area = (currentFeature[0][1][0] - currentFeature[0][0][0]) * (
                            currentFeature[0][1][1] - currentFeature[0][0][1])

                    averagePixelIntensity = PixelIntensitySum / area

                    if (currentFeature[1] == "greater than" and averagePixelIntensity >= currentFeature[2]) or (
                            currentFeature[1] == "less than" and averagePixelIntensity <= currentFeature[2]):
                        evaluation = True
                    else:
                        evaluation = False

                    evaluatedFeatures[rotationCount].append([evaluation, averagePixelIntensity, [x, y]])

            counter = 0
            for ImRunningOutOfNames in evaluatedFeatures[rotationCount]:
                if ImRunningOutOfNames[0]:
                    for ok in features:
                        if type(ok[1]) == list:
                            counter += ok[0][3]
                        else:
                            counter += ok[3]

            setOfFaces.append([[x, y], counter])

            rotationCount += 1

    return setOfFaces

def EvalComplexIntegral(image, feature):  # Deals with integral features that have more than one square
    # Features should be of the form [ [ [ [Left top, Right bottom, Left bottom, Right Top], above or below ], [repeat] ], boundary threshold, say ]
    evaluation = True
    counts = 0
    while evaluation == True and counts <= len(feature) - 1:
        evaluation = evaluateIntegralFeature(image, feature[counts])
        counts += 1
    if evaluation:
        return True
    else:
        return False


def findSimilarity(feature, subImage):
    def flatten(matrix):
        flattened = sum(matrix, [])
        return flattened

    def modulus(array):  # This one only takes one dimensional lists
        total = 0

        for i in array:
            total += i ** 2

        return m.sqrt(total)

    def dotProd(feature, subImage):
        total = 0

        if len(feature) == len(subImage):
            for i in range(len(feature)):
                total += feature[i] * subImage[i]

        else:
            return False

        return total

    feature, subImage = flatten(feature), flatten(subImage)
    modFeature, modSubImage = modulus(feature), modulus(subImage)
    dotProduct = dotProd(feature, subImage)

    if dotProduct:

        if modFeature != 0 or modSubImage != 0:  # If there is a block of absolute darkness for some reason
            angle = (dotProduct / (modFeature * modSubImage))

            return angle + 0.0000000000000001  # Prevents identical division error
        else:
            print("There seems to be a block of absolute darkness here")
            return False
    else:
        print("Lengths don't match up")
        return False


def create_ForestOfRandomTrees(DataPoints, Stump, variablesToConsider, numberOfTrees):
    # The datapoints should be as such: [ [ [Feature 1 vals, is it true], [Feature 2 vals, is it true], etc...] , [
    # [Feature 1, is it true in face 2], etc... ]
    # The features correspond with the list of variables
    # The vals should be in the form of the average pixel intensity of this area

    def BranchGiniImpurity(ProbY,
                           ProbN):  # Of a branch, needs to be weighted and summed to get the Gini Impurity of that decision
        branch_GiniImpurity = 1 - (ProbY ** 2) - (ProbN ** 2)
        return branch_GiniImpurity

    def CalculateNumericGiniImpurity(NumericDataSet,
                                     trueOrFalse, complex,
                                     orientation):  # It also returns the best value to use as splitter and with which orientation
        # Numeric dataset should be in the form of a value with the average pixel intensity of the feature.
        # Complex refers to the fact that the variable we're dealing with has two different features

        def bubbleSort(Sortable):
            n = len(Sortable)
            changes = [0, 1, 2, 3]

            swapped = False

            for i in range(n - 1):

                for j in range(0, n - i - 1):


                    if Sortable[j] > Sortable[j + 1]:
                        swapped = True
                        Sortable[j], Sortable[j + 1] = Sortable[j + 1], Sortable[j]
                        changes[j], changes[j + 1] = changes[j + 1], changes[j]


            return Sortable, changes
        
        trueOrFalseSorted = []
        SortingSet = []
        sortedNums = []

        if not complex:
            for k in range(0, len(NumericDataSet)):  #
                SortingSet.append([NumericDataSet[k], trueOrFalse[k]])
            averages = []
            SortingSet = sorted(SortingSet, key=lambda x: (x[0])) # This won't work because of the changes I made in the data structure
            for h in range(0, len(SortingSet)):
                trueOrFalseSorted.append(SortingSet[h][1])
            trueOrFalse = trueOrFalseSorted
            for h in range(0, len(SortingSet)):
                sortedNums.append(SortingSet[h][0])

            for value in range(0, len(NumericDataSet) - 1):
                averages.append((sortedNums[value] + sortedNums[value + 1]) / 2)

            bestValuesSet = [float("inf"), 0, "greater than"]

            # Highest possible Gini Impurity; the average that causes it, 0 for default;
            # and the orientation of the sign for decision tree creation
            for x in range(0, len(averages)):
                number = averages[x]
                leftYes, leftNo = 0, 0
                rightYes, rightNo = 0, 0
                branchLeft, branchRight = 0, 0
                for i in range(0, len(sortedNums)):
                    if number >= sortedNums[i]:
                        if trueOrFalse[i]:
                            rightYes += 1
                        else:
                            rightNo += 1
                        branchRight += 1

                    elif number < sortedNums[i]:
                        if trueOrFalse[i]:
                            leftYes += 1
                        else:
                            leftNo += 1
                        branchLeft += 1

                if orientation[0] == "less than":
                    NumberOfTrue = leftYes
                    NumberOfFalse = branchRight + leftNo
                elif orientation[0] == "greater than":
                    NumberOfTrue = rightYes
                    NumberOfFalse = branchLeft + rightNo
                totalVals = NumberOfTrue + NumberOfFalse

                if NumberOfFalse == 0:
                    bestValuesSet = [0, averages, orientation]
                    skip = True
                elif NumberOfTrue == 0:
                    branchGini = 1
                    skip = False
                else:
                    skip = False
                    branchGini = BranchGiniImpurity(NumberOfTrue / totalVals, NumberOfFalse / totalVals)

                if not skip:
                    if branchGini < bestValuesSet[0]:
                        bestValuesSet = [branchGini, averages[x], orientation[0]]

            return bestValuesSet

        if complex:  # There are multiple features for each classifier
            seperated = []  # I'm seperated the features into different variables basically, and maximising them seperately
            seperatedTruth = []
            complexValueSet = []
            for i in range(len(NumericDataSet[0])):
                seperated.append(
                                     [[]])  # A feature list, and a list within that for the points, with the orientation in the wider list
                seperatedTruth.append([])
                for datapointer in range(len(NumericDataSet)):
                    seperated[i][0].append(NumericDataSet[datapointer][i])
                    seperatedTruth[i].append(trueOrFalse[datapointer])
                seperated[i].append(orientation[0][i])

            averages = []
            count = 0

            for feature in seperated:  # At this point we just maximize them separately
                averages.append([])
                indexForFurtherUse = seperated.index(feature)
                SortingSet = []
                for k in range(0, len(feature)):  #
                    SortingSet.append([feature[k], seperatedTruth[count]])
                    # after this point, they're changed into the normal names because they're jumbled
                    # through this sorting stuff

                changes = bubbleSort(SortingSet[0][0])

                SortingSet[0][0] = [changes[0][0], changes[0][1], changes[0][2], changes[0][3]]
                # We're sorting the sorting set pixel average values, then we'll sort the boolean evals according to the
                # Changes recorded
                SortingSet[0][1] = [SortingSet[0][1][changes[1][0]], SortingSet[0][1][changes[1][1]], SortingSet[0][1][changes[1][2]], SortingSet[0][1][changes[1][3]]]

                sortedNums = []
                for h in range(0, len(SortingSet[0][0])):
                    trueOrFalseSorted.append(SortingSet[0][1][h])
                trueOrFalse = trueOrFalseSorted
                for h in range(0, len(SortingSet[0][0])):
                    sortedNums.append(SortingSet[0][0][h])

                for value in range(len(feature[0]) - 1):
                    averages[indexForFurtherUse].append((sortedNums[value] + sortedNums[value + 1]) / 2)

                bestValuesSet = [float("inf"), 0, "greater than"]
                # Highest possible Gini Impurity; the average that causes it, 0 for default;
                # and the orientation of the sign for decision tree creation
                for x in range(0, len(averages)):
                    number = averages[indexForFurtherUse][x]
                    leftYes, leftNo = 0, 0
                    rightYes, rightNo = 0, 0
                    branchLeft, branchRight = 0, 0
                    for i in range(0, len(sortedNums) - 1):
                        if number >= sortedNums[i]:
                            if trueOrFalse[i]:
                                rightYes += 1
                            else:
                                rightNo += 1
                            branchRight += 1

                        elif number < sortedNums[i]:
                            if trueOrFalse[i]:
                                leftYes += 1
                            else:
                                leftNo += 1
                            branchLeft += 1

                    if orientation[0][indexForFurtherUse] == "less than":
                        NumberOfTrue = leftYes
                        NumberOfFalse = branchRight + leftNo
                    elif orientation[0][indexForFurtherUse] == "greater than":
                        NumberOfTrue = rightYes
                        NumberOfFalse = branchLeft + rightNo
                    totalVals = NumberOfTrue + NumberOfFalse

                    if NumberOfFalse == 0:
                        bestValuesSet = [0, averages[indexForFurtherUse][x], orientation[0][indexForFurtherUse]]
                        branchGini = 0
                        skip = True
                    elif NumberOfTrue == 0:
                        branchGini = 1
                        skip = False
                    else:
                        skip = False
                        branchGini = BranchGiniImpurity(NumberOfTrue / totalVals, NumberOfFalse / totalVals)

                    if branchGini < bestValuesSet[0] and not skip:
                        bestValuesSet = [branchGini, averages[indexForFurtherUse][x], orientation[0][indexForFurtherUse]]

                complexValueSet.append(bestValuesSet)
                count += 1

            return complexValueSet


    def changeSay(weights,
                  CorrectlyClassified):  # Correctly classified refers to whether the classifier successfully identified the presence of the feature in the image
        totalError = 0
        newWeights = []
        totalOfWeights = 0

        for i in range(0, len(weights)):
            if not CorrectlyClassified[i]:
                totalError += weights[i]

        AmountOfSay = 0.5 * (m.log(((1 - totalError) / (totalError + 0.001)) + 1, 10))

        for j in range(len(weights)):
            if not CorrectlyClassified[j]:
                newWeights.append(weights[j] * (m.e ** AmountOfSay))
            else:
                newWeights.append(weights[j] * (m.e ** (AmountOfSay * -1)))

            totalOfWeights += newWeights[j]

        for k in range(len(newWeights)):
            newWeights[k] = newWeights[k] * (1 / totalOfWeights)

        return AmountOfSay, newWeights

    def createRandomStump(variables, variableMatrixWithWeights, Weights):
        # Can later be integrated to be one, variables array should the same length
        # variables should be in the format [[(X1, Y1), (X2, Y2)], [potentially for complex features there'd be more]]
        # no evaluation of the integral within the training happens, this has already occurred
        # Format should be as in the overall function
        classifiersGini = []
        IsItTrue = []
        truth = []
        correctlyClassified = []

        for i in range(
                len(variables)):  # because we're passing through and choosing which variable is the best to classify
            currentSet = []
            variableTruth = []  # Is this true for this specific variable
            orientation = []
            for j in range(
                    len(variableMatrixWithWeights)):  # Go through the entire dataset and choose for each point whether that variable is true or not

                variableTruth.append(variableMatrixWithWeights[j][variables[i][1]][0])
                # Because each variable has original position in it, and whether it's true is at the first position
                currentSet.append(variableMatrixWithWeights[j][variables[i][1]][1])
                # Each variable has a number corresponding to its original position which is useful in this case

            if type(variables[i][0][1]) == list:
                complex = True
                orientationSet = []
                for subVariable in variables[i][0]:
                    orientationSet.append(subVariable[1])
                orientation.append(orientationSet)

            else:
                complex = False
                orientation.append(variables[i][0][1])

            # At this point, we should cross reference variable Truth (if the new classifier says it should be true) and
            # IsItTrue which says if it is actually true. ACTUALLY this is done within NumericGINI


            NumericGini = CalculateNumericGiniImpurity(currentSet, variableTruth, complex, orientation)

            if not complex:
                classifiersGini.append([NumericGini[0], [variables[i], [NumericGini[1], NumericGini[2]]]])
                # Gini, [variable, [threshold, orientation]]
            else:
                NewGini = 0
                thresholds = []
                for n in range(len(currentSet[0])): # We look at the first index because they all classify the same variable
                    NewGini += NumericGini[n][0]
                    thresholds.append(NumericGini[n][1])
                NewGini = NewGini // len(NumericGini[0])

                classifiersGini.append([NewGini, [variables[i], [thresholds, orientation]]])
                # This is of the form Average Gini, [ [variable, [all of the thresholds, all of the orientaitons]] ]

        leastGini = [1.1, ["the variable in list form", ["Best cut val in int form", "Orientation"]]]
        # Now that we've created the best classifier for every variable, we decide which variable is the best for the
        # stump

        for z in range(len(classifiersGini)):
            if classifiersGini[z][0] < leastGini[0]:
                leastGini = classifiersGini[z]

        for p in range(len(variableMatrixWithWeights)):
            variableIndex = leastGini[1][0][1]  # Because each variable is in the form [actual variable, index in
            # original list]
            if type(leastGini[1][0][0][0]) == list:  # In the variable, there is two coordinates, which are also lists,
                # if we simply checked that there are lists in the variable, it wouldn't work, so we check the first list and see
                # whether its contents would have been a numeric value, or a coordinate. If the latter, it means that the
                # variable is complex
                areTheyAllTrue = True
                for Features in range(len(leastGini[1][0][0][0])):  # How many features are there
                    FeatureEvaluation = variableMatrixWithWeights[p]
                    # Because leastGini[1][0][2] is the variable index
                    if leastGini[1][1][1][0][Features] == "greater than" and FeatureEvaluation[leastGini[1][0][1]][1][Features] >= \
                            leastGini[1][1][0][Features]:  # Orientation
                        pass
                    elif leastGini[1][1][1][0][Features] == "less than" and FeatureEvaluation[leastGini[1][0][1]][1][Features] <= \
                            leastGini[1][1][0][Features]:
                        pass  # So the orientation is less than, and the feature is less than, meaning it's true
                    else:
                        areTheyAllTrue = False  # Because they all need to be true for the feature to be true
                truth.append(areTheyAllTrue)

            else:
                if leastGini[1][1][1] == "greater than" and variableMatrixWithWeights[p][leastGini[1][0][1]][1] >= leastGini[1][1][0]:
                    truth.append(
                        True)  # If the avg pixel intensity is above the threshold, meaning this has been correctly identified
                elif leastGini[1][1][1] == "less than" and variableMatrixWithWeights[p][leastGini[1][0][1]][1] <= leastGini[1][1][0]:
                    truth.append(True)
                else:
                    truth.append(False)

        WeightsAndSay = changeSay(Weights, truth)

        return leastGini, WeightsAndSay

        # In the format, [the lowest Gini value, [the feature, [the cut off, orientation]], [AmountOfSay, weights()]

    def GenerateRandomStumpForest(variablesToConsider, resultsMatrix, NumberOfStumps):
        lengthOfMatrix = len(resultsMatrix)
        lengthOfDataPoint = len(resultsMatrix[0])
        ListOfUnprocessedStumps = []
        # A stump is of the format: [Say, [[classifier, cutOff], left correct percent, right correct percent]]

        for s in range(lengthOfMatrix):
            resultsMatrix[s].append(1 / lengthOfMatrix)  # Appending weights to a set of Datapoints without weights
        currentDataset = resultsMatrix

        for h in range(NumberOfStumps):
            passInVariables = []
            passedInVariables = []
            currentDataset = WeightedBootstrapDatabase(currentDataset)
            NumberOfVariables = 2

            while len(passInVariables) < NumberOfVariables:
                randomNumber = random.randint(0, len(variablesToConsider) - 1)
                if variablesToConsider[randomNumber] not in passInVariables:
                    passInVariables.append([variablesToConsider[randomNumber],
                                           randomNumber])  # So that we know the original index
                    passedInVariables.append(variablesToConsider[randomNumber])

            passedValues = []
            weightsToPassIn = []
            for k in range(len(currentDataset)):
                passedValues.append(currentDataset[k][0:len(currentDataset[k]) - 1])
                weightsToPassIn.append(currentDataset[k][len(currentDataset[k]) - 1])

            stumpToBe = createRandomStump(passInVariables, passedValues, weightsToPassIn)
            ListOfUnprocessedStumps.append([stumpToBe[0], stumpToBe[1][0]])

            for longness in range(len(currentDataset)):
                currentDataset[longness][len(currentDataset[0])-1] = stumpToBe[1][1][longness] # Changing the weights

        return ListOfUnprocessedStumps

    if Stump:
        UnprocessedStumpForest = GenerateRandomStumpForest(variablesToConsider, DataPoints, numberOfTrees)

    return UnprocessedStumpForest


import math as m


def findEyesValue(faceImage):

    BlurredFace = GaussianBlur(faceImage, 3, 1)  # To blend pupil and Iris
    HorizontalGradient = convolve(horizontalSobelOperator, BlurredFace)
    VerticalGradient = convolve(verticalSobelOperator, BlurredFace)

    GradientSum = []

    for i in range(len(HorizontalGradient)):
        GradientSum.append([])
        for j in range(len(HorizontalGradient[i])):
            GradientSum[i].append(HorizontalGradient[i][j] + VerticalGradient[i][j])

    GradientSum = np.array(GradientSum)
    CheckLocation = np.argwhere(GradientSum == np.min(GradientSum))[0]
    PupilLocation = np.argwhere(GradientSum == np.max(GradientSum))[0]
    plot.imshow(GradientSum, cmap='gray', vmin=0, vmax=255)
    plot.show()

    length = len(VerticalGradient)

    if (CheckLocation[0] - 10 <= PupilLocation[0] <= CheckLocation[0] + 15) and (CheckLocation[1] - 10 <= PupilLocation[1] <= CheckLocation[1] + 15):
        return (PupilLocation)
    else:
        return PupilLocation


def findPupilInformation(eyeImage, pupilLocation):
    MaxY = len(eyeImage)
    MaxX = len(eyeImage[0])

    done = False
    count = 1
    newImage = []

    for i in range(len(eyeImage)): # Because the values in the array are stored as uint8, so I am converting them to
        # integers
        newImage.append([])
        for j in range(len(eyeImage[0])):
            newImage[i].append(int(eyeImage[i][j]))

    eyeImage = newImage


    while not done: # This is for the top bound
        if count == 1:
            tester = eyeImage[pupilLocation[0]][pupilLocation[1]]
        elif count >= 8:
            topBound = 2
            done = True

        if abs(eyeImage[pupilLocation[0] + count][pupilLocation[1]] - tester) >= (0.1 * tester):
            topBound = count
            done = True

        count += 1

    done = False
    count = 1

    while not done: # This is for the bottom bound
        if count == 1:
            tester = eyeImage[pupilLocation[0]][pupilLocation[1]]
        elif count >= 8:
            bottomBound = 2
            done = True

        if abs(eyeImage[pupilLocation[0] - count][pupilLocation[1]] - tester) >= (0.1 * tester):
            bottomBound = count
            done = True

        count += 1
    done = False
    count = 1

    while not done: # This is for the right Iris bound
        if count == 1:
            tester = eyeImage[pupilLocation[0]][pupilLocation[1]]
        elif count >= 8:
            rightBoundIris = 2
            done = True

        if abs(eyeImage[pupilLocation[0]][pupilLocation[1] + count] - tester) >= (0.1 * tester):
            rightBoundIris = count
            done = True

        count += 1
    done = False
    count = 1

    while not done: # This is for the right pupil bound
        if count == 1:
            tester = eyeImage[pupilLocation[0]][pupilLocation[1] + rightBoundIris]
        elif count >= 8:
            rightBoundPupil = 2
            done = True

        if abs(eyeImage[pupilLocation[0]][pupilLocation[1] + rightBoundIris + count] - tester) >= (0.1 * tester):
            rightBoundPupil = count
            done = True

        count += 1
    rightBound = rightBoundPupil + rightBoundIris

    done = False
    count = 1

    while not done:  # This is for the left Iris bound
        if count == 1:
            tester = eyeImage[pupilLocation[0]][pupilLocation[1]]
        elif count >= 10:
            leftBoundIris = 2
            done = True

        if abs(eyeImage[pupilLocation[0]][pupilLocation[1] - count] - tester) >= (0.1 * tester):
            leftBoundIris = count
            done = True

        count += 1
    done = False
    count = 1

    while not done:  # This is for the left pupil bound
        if count == 1:
            tester = eyeImage[pupilLocation[0]][pupilLocation[1] - leftBoundIris]
        elif count >= 10:
            leftBoundPupil = 2
            done = True

        if abs(eyeImage[pupilLocation[0]][pupilLocation[1] - leftBoundIris - count] - tester) >= (0.1 * tester):
            leftBoundPupil = count
            done = True

        count += 1
    leftBound = leftBoundIris + leftBoundPupil

    return topBound, bottomBound, leftBound, rightBound

def EyeDirection(LeftX, RightX, TopY, BottomY):  # Referring the x and y distance of the Pupil from the sides of the eye
    TotalX = LeftX + RightX
    TotalY = TopY + BottomY
    print(TotalX, TotalY)
    OriginCoordinates = [TotalX // 2, TotalY // 2]  # The centre of the image
    PupilCoordinates = [(TotalX - RightX), (TotalY - TopY)]
    PupilCoordinates[0] -= OriginCoordinates[0]
    PupilCoordinates[1] -= OriginCoordinates[1]  # Now it's in a traditional Cartesian plane

    # Here are the checks for the Eye being pointed directly forward

    CheckCoordinates = [0, 0]

    if PupilCoordinates[0] < 0:
        CheckCoordinates[0] = PupilCoordinates[0] * -1
    else:
        CheckCoordinates[0] = PupilCoordinates[0]

    if PupilCoordinates[1] < 0:
        CheckCoordinates[1] = PupilCoordinates[1] * -1
    else:
        CheckCoordinates[1] = PupilCoordinates[1]

    if CheckCoordinates[0] < (TotalX // 2) // 5 and CheckCoordinates[1] < (TotalY // 2) // 5:
        return None

    if PupilCoordinates[0] != 0:
        Theta = m.atan(float((float(PupilCoordinates[1]) / (PupilCoordinates[0]))))  # CONSIDER VARIOUS QUADRANTS
    else:
        if PupilCoordinates[1] > 0:
            Theta = 90
        if PupilCoordinates[1] < 0:
            Theta = 270

    if PupilCoordinates[0] == 0 and PupilCoordinates[1] == 0:
        return [0]

    elif PupilCoordinates[0] < 0 and PupilCoordinates[1] >= 0:
        Theta += 90

    elif PupilCoordinates[0] < 0 and PupilCoordinates[1] < 0:
        Theta += 180

    elif PupilCoordinates[1] < 0 and PupilCoordinates[0] >= 0:
        Theta += 270

    print(Theta)

    return Theta


def smooth(values):
    Axis = "positive"  # It starts off with respect to the positive x-axis

    def takeAverage(values):
        sum = 0;
        girth = 0

        for i in values:
            sum += i
            girth += 1

        return sum / girth

    def standardDeviation(values):
        angleAverage = takeAverage(values)
        subAngles = [];
        subSum = 0;
        girth = 0

        for i in values:
            subAngles.append(i - angleAverage)
            girth += 1

        for z in subAngles:
            subSum += z ** 2

        variance = subSum / girth

        return m.sqrt(variance)

    def AxisCheck(datapoints):
        check = 0;
        girth = 0

        for i in datapoints:
            if i > 320 or i < 40:
                check += 1
            girth += 1

        if (check / girth) * 100 > 85:
            return False

        else:
            return True

    def Zscore(value):
        score = (value - AngleAverage) / standardeviation

        return m.fabs(score)

    if not AxisCheck(values):
        Axis = "negative"  # it is now with respect to the negative x-axis

        for i in values:

            if i > 180:
                values[i] = i - 180

            if i <= 180:
                values[i] = 180 - i

    standardeviation = standardDeviation(values)
    AngleAverage = takeAverage(values)
    passedValues = []

    for j in values:

        if Zscore(j) < 2:
            passedValues.append(j)

    if Axis == "negative":

        for i in passedValues:

            if i > 180:
                passedValues[i] = i + 180

            if i <= 180:
                passedValues[i] = 180 - i

    return takeAverage(passedValues)


def screenSize():
    screen = CGDisplayBounds(CGMainDisplayID())
    return screen.size.width, screen.size.height


def mouseEvent(type, X, Y):
    event = CGEventCreateMouseEvent(None, type, (X, Y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)


def mouseMove(X, Y):
    mouseEvent(kCGEventMouseMoved, X, Y)


def getMouseLocation():
    return Quartz.NSEvent.mouseLocation().x, Quartz.NSEvent.mouseLocation().y


def createMoveThread(MSpeed, MDuration, MAngle):
    print(MAngle, "Mangle")
    MouseController = MouseControl(MSpeed, MDuration, MAngle)
    CurrentMove = Thread(target=MouseController.MoveMouse())
    CurrentMove.daemon = True
    # Allows program to exit even if the "Daemon" thread is still running
    CurrentMove.start()
    t.sleep(1)


def screen_size():
    windowSize = CGDisplayBounds(CGMainDisplayID())
    return windowSize.size.width, windowSize.size.height


def createTypeThread(keyToPress, pressDuration):
    keyBoardControl = KeyboardMover(pressDuration, keyToPress)
    keyBoardControl = Thread(target=keyBoardControl.pressKey())
    keyBoardControl.daemon = True
    # Allows program to exit even if the "Daemon" thread is still running
    keyBoardControl.start()


def loadSettings(MSpeed, MDuration):
    try:

        settingsFile = open(r"settings.txt", "r")  # Format yet to be decided
        Lines = settingsFile.readlines()
        print(Lines)
        print("File Opened")

    # For this one, we now instead ask questions as to the settings initially wanted etc.

    except IOError:  # If there are no prior settings, let the new settings be the new default

        settingsFile = open(r"settings.txt", "w+")
        settingsFile.write(f"MSpeed:{MSpeed}, ")
        settingsFile.write(f"MDuration:{MDuration}, ")
        settingsFile.write(f"Trained:{False}, ")

        print("File loaded and written")

    settingsFile.close()


def editSettings(Mspeed, MDuration):
    try:
        settingsFile = open(r"settings.txt", "r+")

        Lines = settingsFile.readlines()

        settingsFile.write(f"MSpeed:{MSpeed}, ")
        settingsFile.write(f"MDuration:{MDuration}, ")
        if (Lines[0].split(","))[2] == f"Trained:{False}":  # Whether the model has been trained yet.
            settingsFile.write(f"Trained:{False}, ")
        else:
            settingsFile.write(f"Trained: {True}")

        print("File edited")

    except IOError:

        settingsFile = open(r"settings.txt", "w+")
        settingsFile.write(f"MSpeed:{MSpeed}, ")
        settingsFile.write(f"MDuration:{MDuration}, ")
        settingsFile.write(f"Trained:{False}, ")

        print("File not existent, new settings file created")


def Warn():
    os.system('afplay /System/Library/Sounds/Sosumi.aiff')
    os.system('say "Please check position."')


# Defining Classes

class MouseControl(Thread):
    def __init__(self, MSpeed, MDuration, MAngle):
        Thread.__init__(self)
        self.HypotenuseDistance = MDuration * MSpeed
        self.XTravel = MDuration * MSpeed * m.cos(MAngle)
        self.YTravel = MDuration * MSpeed * m.sin(MAngle)

    def MoveMouse(self):
        while True:
            try:
                HypotenuseDistance = MDuration * MSpeed
                XTravel = HypotenuseDistance * m.cos(MAngle)
                YTravel = HypotenuseDistance * m.sin(MAngle)
                currentPos = getMouseLocation()

                for i in range(0, MDuration):
                    currentPos = getMouseLocation()
                    # This takes mouse position from bottom left, while move takes it from top right, so the y coordinate is adjusted in calculations
                    mouseMove(currentPos[0] + (XTravel / m.sqrt(HypotenuseDistance)),
                              screenSize()[1] - currentPos[1] + (XTravel / m.sqrt(HypotenuseDistance)))
                    # The Square root is rather arbitrary as I found during experimentation this felt the most 'smooth' while using, personally.
                    t.sleep(
                        1)  # Currently a stop-gap measure for what will eventually require a whole program clock related to cycles
            finally:
                break


class KeyboardMover(Thread):
    def __init__(self, timeTaken, keyToPress):
        Thread.__init__(self)
        self.timeTaken = timeTaken
        self.keyToPress = keyToPress
        self.keyCodes = {  # This list is courtesy of Stack Overflow https://stackoverflow.com/q/3202629/55075
            'a': 0x00,
            's': 0x01,
            'd': 0x02,
            'f': 0x03,
            'h': 0x04,
            'g': 0x05,
            'z': 0x06,
            'x': 0x07,
            'c': 0x08,
            'v': 0x09,
            'b': 0x0B,
            'q': 0x0C,
            'w': 0x0D,
            'e': 0x0E,
            'r': 0x0F,
            'y': 0x10,
            't': 0x11,
            '1': 0x12,
            '2': 0x13,
            '3': 0x14,
            '4': 0x15,
            '6': 0x16,
            '5': 0x17,
            '=': 0x18,
            '9': 0x19,
            '7': 0x1A,
            '-': 0x1B,
            '8': 0x1C,
            '0': 0x1D,
            ']': 0x1E,
            'o': 0x1F,
            'u': 0x20,
            '[': 0x21,
            'i': 0x22,
            'p': 0x23,
            'l': 0x25,
            'j': 0x26,
            '\'': 0x27,
            'k': 0x28,
            ';': 0x29,
            '\\': 0x2A,
            ',': 0x2B,
            '/': 0x2C,
            'n': 0x2D,
            'm': 0x2E,
            '.': 0x2F,
            '`': 0x32,
            'k.': 0x41,
            'k*': 0x43,
            'k+': 0x45,
            'kclear': 0x47,
            'k/': 0x4B,
            'k\n': 0x4C,
            'k-': 0x4E,
            'k=': 0x51,
            'k0': 0x52,
            'k1': 0x53,
            'k2': 0x54,
            'k3': 0x55,
            'k4': 0x56,
            'k5': 0x57,
            'k6': 0x58,
            'k7': 0x59,
            'k8': 0x5B,
            'k9': 0x5C,

            '\n': 0x24,
            '\t': 0x30,
            ' ': 0x31,
            'del': 0x33,
            'delete': 0x33,
            'esc': 0x35,
            'escape': 0x35,
            'cmd': 0x37,
            'command': 0x37,
            'shift': 0x38,
            'caps lock': 0x39,
            'option': 0x3A,
            'ctrl': 0x3B,
            'control': 0x3B,
            'right shift': 0x3C,
            'rshift': 0x3C,
            'right option': 0x3D,
            'roption': 0x3D,
            'right control': 0x3E,
            'rcontrol': 0x3E,
            'fun': 0x3F,
            'function': 0x3F,
            'f17': 0x40,
            'volume up': 0x48,
            'volume down': 0x49,
            'mute': 0x4A,
            'f18': 0x4F,
            'f19': 0x50,
            'f20': 0x5A,
            'f5': 0x60,
            'f6': 0x61,
            'f7': 0x62,
            'f3': 0x63,
            'f8': 0x64,
            'f9': 0x65,
            'f11': 0x67,
            'f13': 0x69,
            'f16': 0x6A,
            'f14': 0x6B,
            'f10': 0x6D,
            'f12': 0x6F,
            'f15': 0x71,
            'help': 0x72,
            'home': 0x73,
            'pgup': 0x74,
            'page up': 0x74,
            'forward delete': 0x75,
            'f4': 0x76,
            'end': 0x77,
            'f2': 0x78,
            'page down': 0x79,
            'pgdn': 0x79,
            'f1': 0x7A,
            'left': 0x7B,
            'right': 0x7C,
            'down': 0x7D,
            'up': 0x7E
        }

    def findKeyCode(self):
        if self.keyToPress.isalpha() and not self.keyToPress.islower():
            self.keyToPress = self.keyToPress.lower()

        if self.keyToPress in self.keyCodes:
            return self.keyCodes[self.keyToPress]
        else:
            return False

    def pushKey(self):
        self.keyToPress = self.findKeyCode(self.keyToPress)

        if self.keyToPress != False:
            CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, self.keyToPress, True))
            t.sleep(0.0001)

        else:
            print("How did you even do that??")

    def releaseKey(self):
        self.keyToPress = self.findKeyCode(self.keyToPress)

        if self.keyToPress != False:
            CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, self.keyToPress, False))
            t.sleep(0.001)

        else:
            print(" ... how???")

    def pressKey(self):
        self.pushKey(self.keyToPress)
        t.sleep(self.timeTaken)
        self.releaseKey(self.keyToPress)


def Main():
    while True:
        cyclesToComplete = 2
        loadSettings(MSpeed, MDuration)
        count = 0
        sets = []
        for i in range(cyclesToComplete):
            #OriginalImage = photoshot()

            if count == 0:
                TrainingPhoto = Image.open('/Users/kostasdemiris/Pictures/Photos for training NEA/Training Photo five.jpg')
                OriginalImage = np.asarray(TrainingPhoto)
            if count == 1:
                TrainingPhoto = Image.open('/Users/kostasdemiris/Pictures/Photos for training NEA/Training Photo one.jpg')
                OriginalImage = np.asarray(TrainingPhoto)

            if count == 2:
                TrainingPhoto = Image.open('/Users/kostasdemiris/Pictures/Photos for training NEA/Training Photo two.jpg')
                OriginalImage = np.asarray(TrainingPhoto)


            gray = ColorToGray(OriginalImage)
            plot.imshow(gray, cmap='gray', vmin=0, vmax=255)
            plot.show()
            ResizedGray = cv2.resize(gray, (520, 240))

            FaceShot = integralImage(ResizedGray)

            setOfFace = EvaluateImage(featuresToTrain, FaceShot, (100, 120))

            bestFace = 0
            probablyFace = []
            for face in setOfFace:
                if face[1] == bestFace:
                    probablyFace.append(face)
                elif face[1] > bestFace:
                    probablyFace = [face]
                    bestFace = face[1]
                else:
                    pass

            faceLength = len(FaceShot)
            faceWidth = len(FaceShot[0])
            midpoint = [faceWidth // 2, faceLength // 2]

            lowestDifference = float('inf')
            TheFaceToUse = []

            for possibleFace in probablyFace:
                difference = m.sqrt(
                    (abs(possibleFace[0][1] - midpoint[1]) ** 2) + ((abs(possibleFace[0][0] - midpoint[0])) ** 2))
                if difference < lowestDifference:
                    lowestDifference = difference
                    TheFaceToUse = possibleFace

            TheFaceToUse = TheFaceToUse[0]
            print(TheFaceToUse, "This is the face location")
            FaceImage = []

            for kappa in range(0, 120):

                FaceImage.append(ResizedGray[kappa + TheFaceToUse[1]][TheFaceToUse[0]: TheFaceToUse[0] + 100])

            plot.imshow(FaceImage, cmap='gray', vmin=0, vmax=255)
            plot.show()

            ZoomedImage = []
            for kappa in range(30, 80):
                ZoomedImage.append(FaceImage[kappa][15: 50])
                # This will cut the edge of the image off to prevent problems with clothing and hair, as we know vaguely where the
                # eyes are on the face according to ratios.

            ZoomedImage = np.array(ZoomedImage)
            CopyImage = ZoomedImage
            CopyImage = CopyImage.tolist()
            ZoomedImage = cv2.resize(ZoomedImage, (180, 200))

            plot.imshow(ZoomedImage, cmap='gray', vmin=0, vmax=255)
            plot.show()

            InitialBlurImage = GaussianBlur(ZoomedImage, 3, 0.1)
            CorrectedBlurImage = GaussianBlur(InitialBlurImage, 1, 3)

            PupilLocation = findEyesValue(CorrectedBlurImage)

            print(PupilLocation[0] / 4, PupilLocation[1] / 4)

            bounds = findPupilInformation(CopyImage, (int(PupilLocation[0] / 4), int(PupilLocation[1] / 4)))

            plot.imshow(CopyImage, cmap='gray', vmin=0, vmax=255)
            plot.show()

            direction = EyeDirection(bounds[2], bounds[3], bounds[0], bounds[1])

            count += 1

            sets.append(direction)

        counter = 0
        lengthValue = len(sets)
        alternateSets = []
        for i in range(len(sets)):
            if type(sets[i]) == list:
                counter += 1
            else:
                alternateSets.append(sets[i])

        sets = alternateSets

        if counter >= lengthValue // 2:
            mouseLocation = getMouseLocation()
            posx = mouseLocation[0]
            posy = mouseLocation[1]
            mouseEvent(kCGEventLeftMouseDown, posx, posy)
            mouseEvent(kCGEventLeftMouseUp, posx, screenSize()[1] - posy)

        else:
            values = smooth(sets)

            createMoveThread(MSpeed, MDuration, values)





'''
gray = photoshot()
gray = ColorToGray(gray)

plot.imshow(gray, cmap='gray', vmin=0, vmax=255)
plot.show()

gray = GaussianBlur(gray, 11, 1)

plot.imshow(gray, cmap='gray', vmin=0, vmax=255)
plot.show()
'''

'''
TrainingPhoto = Image.open('/Users/kostasdemiris/Pictures/Photos for training NEA/Training Photo five.jpg')
FaceShot = np.asarray(TrainingPhoto)

gray = ColorToGray(FaceShot)
ResizedGray = cv2.resize(gray, (520, 240))


FaceShot = integralImage(ResizedGray)

setOfFace = EvaluateImage(featuresToTrain, FaceShot, (100, 120))

print(setOfFace)


#ForestClassifier = create_ForestOfRandomTrees(trainingSetForTesting, True, features, 5)
#print(ForestClassifier)

# rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
# plot.add_patch(rect)
bestFace = 0
probablyFace = []
for face in setOfFace:
    print(face)
    if face[1] == bestFace:
        probablyFace.append(face)
    elif face[1] > bestFace:
        probablyFace = [face]
        bestFace = face[1]
    else:
        print("Nah")

print(probablyFace)

faceLength = len(FaceShot)
faceWidth = len(FaceShot[0])
midpoint = [faceWidth // 2, faceLength // 2]

lowestDifference = float('inf')
TheFaceToUse = []

for possibleFace in probablyFace:
    difference = m.sqrt((abs(possibleFace[0][1] - midpoint[1])**2) + ((abs(possibleFace[0][0] - midpoint[0]))**2))
    if difference < lowestDifference:
        lowestDifference = difference
        TheFaceToUse = possibleFace[0]

print(TheFaceToUse)
FaceImage = []

for kappa in range(0, 120):
    print(kappa + TheFaceToUse[1], kappa, TheFaceToUse[1])
    FaceImage.append(ResizedGray[kappa + TheFaceToUse[1]][TheFaceToUse[0]: TheFaceToUse[0] + 100])

ZoomedImage = []
for kappa in range(40, 70):
    ZoomedImage.append(FaceImage[kappa][5: 90])
    # This will cut the edge of the image off to prevent problems with clothing and hair, as we know vaguely where the
    # eyes are on the face according to ratios.


ZoomedImage = np.array(ZoomedImage)
CopyImage = ZoomedImage
CopyImage = CopyImage.tolist()
copyListLength = len(CopyImage)
copyListWidth = len(CopyImage[0])
ZoomedImage = cv2.resize(ZoomedImage, (340, 120))



BlurredImage = GaussianBlur(ZoomedImage, 3, 0.1)
CorrectedBlurredImage = GaussianBlur(BlurredImage, 3, 0.1)

PupilPlace = findEyesValue(CorrectedBlurredImage)

print(PupilPlace)

plot.imshow(ZoomedImage, cmap='gray', vmin=0, vmax=255)
plot.show()

#print(EyePupilDistance(CorrectedBlurredImage, (PupilPlace[1], PupilPlace[0]))) # Because Numpy return [y, x] while we want [x, y]


print(PupilPlace[0]/ 3, "This")
print(copyListLength, copyListWidth)
print(findPupilInformation(CopyImage, (int(PupilPlace[0]/ 4), int(PupilPlace[1] / 4))))


plot.imshow(CopyImage, cmap='gray', vmin=0, vmax=255)
plot.show()
'''

Main()
