import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from importImage import getImageArray
import algorithms

def main():
    # Image sequences are in the format of (frame, y-position, x-position, color)
    imageSequence = getImageArray(input("Enter file name of writing video: "))

    # Blur is gaussian, 2nd parameter is the sigma, 3rd parameter is how many sigmas to truncate at
    blurImageSequence = algorithms.blur(imageSequence, 3, 3)

    # Identify paper from first frame in image sequence using corners

    # Identify object in subsequent frames

    # Use color histogram from paper from previous frame to ensure color doesn't change to much or else remove from object

    # Close object a few times to allow changes such as writing

    # Reconstruct paper based on motions observed, object identified in the frame and using paper from most recent frame if object is obstructed


# Testing lines for ensuring all algorithms interact properly
#    print(imageSequence.shape)
#    print(blurImageSequence.shape)
#    plt.figure() ; io.imshow(imageSequence[0,:,:,:])
#    io.show()
#    plt.figure() ; io.imshow(blurImageSequence[0,:,:,:])
#    io.show()

#    testObj = np.array([[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,1,1,1,0,0], [0,0,1,0,1,0,0], [0,0,1,1,1,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]])
#    closedObj = algorithms.close(testObj)

#    print(closedObj.shape)
#    print(closedObj)
#    plt.figure() ; io.imshow(testObj)
#    io.show()
#    plt.figure() ; io.imshow(closedObj)
#    io.show()
#
#    print(algorithms.normColorHistRegion(imageSequence[0,:,:,:], np.ones(imageSequence[0,:,:,0].shape), 4))


main()