from sklearn.cluster import KMeans
import random as rng
import cv2
import imutils
import argparse
from imutils import contours
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import *

def main():
    parser = argparse.ArgumentParser(description='Process an image to measure foot size.')
    parser.add_argument('image_path', type=str, help='Path to the image to be processed')
    args = parser.parse_args()

    ImgPath = args.image_path

    print(f"Reading image from: {ImgPath}")
    
    oimg = imread(ImgPath)

    if not os.path.exists('output'):
        os.makedirs('output')

    preprocessedOimg = preprocess(oimg)
    cv2.imwrite('output/preprocessedOimg.jpg', preprocessedOimg)

    clusteredImg = kMeans_cluster(preprocessedOimg)
    cv2.imwrite('output/clusteredImg.jpg', clusteredImg)

    edgedImg = edgeDetection(clusteredImg)
    cv2.imwrite('output/edgedImg.jpg', edgedImg)

    boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
    pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
    cv2.imwrite('output/pdraw.jpg', pdraw)

    croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
    cv2.imwrite('output/croppedImg.jpg', croppedImg)
    
    newImg = overlayImage(croppedImg, pcropedImg)
    cv2.imwrite('output/newImg.jpg', newImg)

    fedged = edgeDetection(newImg)
    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
    fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
    cv2.imwrite('output/fdraw.jpg', fdraw)

    print("feet size (cm): ", calcFeetSize(pcropedImg, fboundRect)/10)


if __name__ == '__main__':
    main()
