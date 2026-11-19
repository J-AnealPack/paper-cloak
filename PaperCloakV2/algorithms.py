import math
from skimage import filters
import numpy as np
import scipy

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

# Finds corners of paper in unobstructed image using Harris (Adapted from HW 8) T = 10000 seems to work decently
def corners(img, T):
    # Convert image to grayscale
    gray_img = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3

    Gauss = gauss2D(1)
    Gx,Gy = gaussDeriv2D(0.7)
    alpha = 0.05

    Ix = scipy.ndimage.convolve(gray_img, -Gx, mode='nearest')
    Iy = scipy.ndimage.convolve(gray_img, -Gy, mode='nearest')
    Ix_2 = Ix * Ix
    Iy_2 = Iy * Iy
    IxIy = Ix * Iy
    gIx_2 = scipy.ndimage.convolve(Ix_2, Gauss, mode='nearest')
    gIy_2 = scipy.ndimage.convolve(Iy_2, Gauss, mode='nearest')
    gIxIy = scipy.ndimage.convolve(IxIy, Gauss, mode='nearest')

    R = gIx_2*gIy_2 - (gIxIy ** 2) - alpha * ((gIx_2 + gIy_2) ** 2)

    R[R < T] = 0

    # Search for 4 non-zero thresholds with closest Manhattan distance to corners of image
    minDistance = 1000000
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            if R[y,x] != 0:
                dist = x + y
                if dist < minDistance:
                    minDistance = dist
                    topLeftPoint = [y, x]
    
    minDistance = 1000000
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            if R[y,x] != 0:
                dist = (R.shape[1] - x) + y
                if dist < minDistance:
                    minDistance = dist
                    topRightPoint = [y, x]
    
    minDistance = 1000000
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            if R[y,x] != 0:
                dist = x + (R.shape[0] - y)
                if dist < minDistance:
                    minDistance = dist
                    bottomLeftPoint = [y, x]

    minDistance = 1000000
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            if R[y,x] != 0:
                dist = (R.shape[1] - x) + (R.shape[0] - y)
                if dist < minDistance:
                    minDistance = dist
                    bottomRightPoint = [y, x]

    return np.array([topLeftPoint, bottomLeftPoint, bottomRightPoint, topRightPoint])

# Taken from HW 2, used for corner detector
def gaussDeriv2D(sigma):
    gaussDerivX = lambda x, y: x/(2*np.pi*sigma**4) * np.exp(-(x**2+y**2)/(2*sigma**2)) # Flip sign so values are ascending instead of descinding in mask
    maskSize = int(np.ceil(3*sigma)*2+1) # Get size of Gx and Gy

    Values = np.linspace(int(-np.ceil(3*sigma)), int(np.ceil(3*sigma)), maskSize)
    Gx = np.zeros((maskSize, maskSize))

    for xIndex in range(len(Values)):
        for yIndex in range(len(Values)):
            Gx[yIndex][xIndex] = gaussDerivX(Values[xIndex], Values[yIndex]) # Calculate values

    # Modification from HW 2 so sum of abs derivative is 1
    Gx = Gx / np.sum(np.abs(Gx))

    Gy = np.transpose(Gx)

    return Gx, Gy

# Taken from HW 2, used for corner detector
def gauss2D(sigma):
    gaussFunc = lambda x, y: 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*sigma**2))
    maskSize = int(np.ceil(3*sigma)*2+1)

    Values = np.linspace(int(-np.ceil(3*sigma)), int(np.ceil(3*sigma)), maskSize)
    gauss = np.zeros((maskSize, maskSize))

    for xIndex in range(len(Values)):
        for yIndex in range(len(Values)):
            gauss[yIndex][xIndex] = gaussFunc(Values[xIndex], Values[yIndex]) # Calculate values

    gauss = gauss / np.sum(np.abs(gauss))

    return gauss