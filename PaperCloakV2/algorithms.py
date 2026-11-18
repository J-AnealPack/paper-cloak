import math
from skimage import filters
import numpy as np

# Blur Implementation
def blur(imageSequence, sigma, truncationSigma):
    blurredImages = np.zeros(imageSequence.shape)
    for imIndex in range(blurredImages.shape[0]):
        blurredImages[imIndex,:,:,:] = filters.gaussian(imageSequence[imIndex,:,:,:], sigma=sigma, truncate=truncationSigma)
    return blurredImages

# Closing object
# Assume object is a 2-D array of 0's and 1's where 1 represents where the object is
def close(object):
    dilatedObj = np.zeros(object.shape)
    closedObj = np.zeros(object.shape)

    # First perform dilation
    for y in range(object.shape[0]):
        for x in range(object.shape[0]):
                if (y == 0 and x == 0):
                    if (object[y,x] == 1 or object[y,x+1] == 1 or object[y+1,x] == 1 or object[y+1,x+1] == 1):
                        dilatedObj[y,x] = 1
                elif (y == 0 and x == object.shape[0] - 1):
                    if (object[y,x] == 1 or object[y,x-1] == 1 or object[y+1,x] == 1 or object[y+1,x-1] == 1):
                        dilatedObj[y,x] = 1
                elif (y == object.shape[0] - 1 and x == 0):
                    if (object[y,x] == 1 or object[y,x+1] == 1 or object[y-1,x] == 1 or object[y-1,x+1] == 1):
                        dilatedObj[y,x] = 1
                elif (y == object.shape[0] - 1 and x == object.shape[0] - 1):
                    if (object[y,x] == 1 or object[y,x-1] == 1 or object[y-1,x] == 1 or object[y-1,x-1] == 1):
                        dilatedObj[y,x] = 1
                elif (y == 0):
                    if (object[y,x-1] == 1 or object[y,x] == 1 or object[y,x+1] == 1 or object[y+1,x-1] == 1 or object[y+1,x] == 1 or object[y+1,x+1] == 1):
                        dilatedObj[y,x] = 1
                elif (y == object.shape[0] - 1):
                    if (object[y,x-1] == 1 or object[y,x] == 1 or object[y,x+1] == 1 or object[y-1,x-1] == 1 or object[y-1,x] == 1 or object[y-1,x+1] == 1):
                        dilatedObj[y,x] = 1
                elif (x == 0):
                    if (object[y-1,x] == 1 or object[y,x] == 1 or object[y+1,x] == 1 or object[y-1,x+1] == 1 or object[y,x+1] == 1 or object[y+1,x+1] == 1):
                        dilatedObj[y,x] = 1
                elif (x == object.shape[0] - 1):
                    if (object[y-1,x] == 1 or object[y,x] == 1 or object[y+1,x] == 1 or object[y-1,x-1] == 1 or object[y,x-1] == 1 or object[y+1,x-1] == 1):
                        dilatedObj[y,x] = 1
                elif object[y-1,x-1] == 1 or object[y,x-1] == 1 or object[y+1,x-1] == 1 or object[y-1,x] == 1 or object[y,x] == 1 or object[y+1,x] == 1 or object[y-1,x+1] == 1 or object[y,x+1] == 1 or object[y+1,x+1] == 1:
                    dilatedObj[y,x] = 1

    # Then perform erosion
    for y in range(dilatedObj.shape[0]):
        for x in range(dilatedObj.shape[0]):
            if (y == 0 and x == 0):
                if (dilatedObj[y,x] == 1 and dilatedObj[y,x+1] == 1 and dilatedObj[y+1,x] == 1 and dilatedObj[y+1,x+1] == 1):
                    closedObj[y,x] = 1
            elif (y == 0 and x == dilatedObj.shape[0] - 1):
                if (dilatedObj[y,x] == 1 and dilatedObj[y,x-1] == 1 and dilatedObj[y+1,x] == 1 and dilatedObj[y+1,x-1] == 1):
                    closedObj[y,x] = 1
            elif (y == dilatedObj.shape[0] - 1 and x == 0):
                if (dilatedObj[y,x] == 1 and dilatedObj[y,x+1] == 1 and dilatedObj[y-1,x] == 1 and dilatedObj[y-1,x+1] == 1):
                    closedObj[y,x] = 1
            elif (y == dilatedObj.shape[0] - 1 and x == dilatedObj.shape[0] - 1):
                if (dilatedObj[y,x] == 1 and dilatedObj[y,x-1] == 1 and dilatedObj[y-1,x] == 1 and dilatedObj[y-1,x-1] == 1):
                    closedObj[y,x] = 1
            elif (y == 0):
                if (dilatedObj[y,x-1] == 1 and dilatedObj[y,x] == 1 and dilatedObj[y,x+1] == 1 and dilatedObj[y+1,x-1] == 1 and dilatedObj[y+1,x] == 1 and dilatedObj[y+1,x+1] == 1):
                    closedObj[y,x] = 1
            elif (y == dilatedObj.shape[0] - 1):
                if (dilatedObj[y,x-1] == 1 and dilatedObj[y,x] == 1 and dilatedObj[y,x+1] == 1 and dilatedObj[y-1,x-1] == 1 and dilatedObj[y-1,x] == 1 and dilatedObj[y-1,x+1] == 1):
                    closedObj[y,x] = 1
            elif (x == 0):
                if (dilatedObj[y-1,x] == 1 and dilatedObj[y,x] == 1 and dilatedObj[y+1,x] == 1 and dilatedObj[y-1,x+1] == 1 and dilatedObj[y,x+1] == 1 and dilatedObj[y+1,x+1] == 1):
                    closedObj[y,x] = 1
            elif (x == dilatedObj.shape[0] - 1):
                if (dilatedObj[y-1,x] == 1 and dilatedObj[y,x] == 1 and dilatedObj[y+1,x] == 1 and dilatedObj[y-1,x-1] == 1 and dilatedObj[y,x-1] == 1 and dilatedObj[y+1,x-1] == 1):
                    closedObj[y,x] = 1
            elif dilatedObj[y-1,x-1] == 1 and dilatedObj[y,x-1] == 1 and dilatedObj[y+1,x-1] == 1 and dilatedObj[y-1,x] == 1 and dilatedObj[y,x] == 1 and dilatedObj[y+1,x] == 1 and dilatedObj[y-1,x+1] == 1 and dilatedObj[y,x+1] == 1 and dilatedObj[y+1,x+1] == 1:
                closedObj[y,x] = 1

    return closedObj

# Normalized Color histogram in specified object in image
# Assume object is a 2-D array of 0's and 1's where 1 represents where the object is
def normColorHistRegion(image, object, numBinsPerAxis):
    colorHist = np.zeros((numBinsPerAxis, numBinsPerAxis, numBinsPerAxis))
    numPixels = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            numPixels = numPixels + 1
            if object[y,x] == 1:
                colorHist[math.floor(numBinsPerAxis*(image[y,x][0]/256)),math.floor(numBinsPerAxis*(image[y,x][1]/256)),math.floor(numBinsPerAxis*(image[y,x][2]/256))] = colorHist[math.floor(numBinsPerAxis*(image[y,x][0]/256)),math.floor(numBinsPerAxis*(image[y,x][1]/256)),math.floor(numBinsPerAxis*(image[y,x][2]/256))] + 1
    colorHist = colorHist / numPixels
    return colorHist