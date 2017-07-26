# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib

def reorder_matches(kp1, kp2, matches, k):
    """

    matches 中的点是  kp1 在 kp2 中的 k 次出现， 这个函数的目的是要把 每次 出现的点放在一起。

    :param kp1: 要找的小图的 SIFT keypoints
    :param kp2: 大图的 SIFT keypoints
    :param matches: 要重排的匹配
    :return:
    """
    pts = []
    pts_indices = []
    for m in matches:
        pts.append(kp2[m.trainIdx].pt)
        pts_indices.append(m.trainIdx)
    pts = np.float32(pts)
    Z = np.vstack((pts))

    criteria = (int(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER), 10, 1.0)
    # ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, int(cv2.KMEANS_RANDOM_CENTERS))

    ret, label, center = cv2.kmeans(Z, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    ravel = label.ravel()
    # Now separate the data, Note the flatten()
    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]

    ret = {}
    for i, c in enumerate(ravel):
        if not ret.has_key(c):
            ret[c] = []
        ret[c].append(matches[i])

 #   return matches
    return ret



def drawMatches(img1, kp1, img2, kp2, matches, color = (255, 0, 0)):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    i = 0
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        color = (255, 0, 0)
        if i % 2 == 1:
            color = (0, 255, 0)
        i += 1
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, color, 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, color, 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, 1)

    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def drawOrderedMatches(img1, kp1, img2, kp2, ordered_matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    i = 0
    for c in ordered_matches.keys():
        matches = ordered_matches[c]
        for mat in matches:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            color = (255, 0, 0)
            if c % 3 == 1:
               color = (0, 255, 0)

            if c % 3 == 2:
               color = (0, 0, 255)

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1), int(y1)), 4, color, 1)
            cv2.circle(out, (int(x2) + cols1, int(y2)), 4, color, 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), color, 1)

    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out




def test():
    # matplotlib.use('Agg')
    X = np.random.randint(25, 50, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    Z = np.vstack((X, Y))


    print "X:", X
    print "Y:", Y
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria and apply kmeans()
    criteria = (int(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER), 10, 1.0)
    # ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, int(cv2.KMEANS_RANDOM_CENTERS))

    ret, label, center = cv2.kmeans(Z, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    ravel = label.ravel()
    # Now separate the data, Note the flatten()
    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]
    # Plot the data
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()
    return

if __name__ == "__main__":
    test()