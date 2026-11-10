import imageio
import numpy as np
from importImage import getImageArray

def main():
    # Image sequences are in the format of (frame, y-position, x-position, color)
    imageSequence = getImageArray(input("Enter file name of writing video: "))