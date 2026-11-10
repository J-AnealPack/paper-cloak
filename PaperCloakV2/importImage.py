import imageio
import numpy as np

def getImageArray(fileName):
    video = imageio.get_reader(fileName, 'ffmpeg')
    images = []
    for image in video:
        images.append(image)
    video.close()
    images = np.array(images)
    return images