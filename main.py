# -*- coding: utf-8 -*-
import numpy as np
from funcs import reorder_matches, drawOrderedMatches, drawMatches, read_transparent_png
import cv2
from matplotlib import pyplot as plt
import gen_html


MIN_MATCH_COUNT = 10
MAX_OCCURS = 10  # 最多尝试探测这么多个。如果打图中小图出现的次数小于等于这个，那么希望可以探测对。

def do_sift(img1,img2 ):
    """

    :param img1:  小图
    :param img2:  大图
    :return:
    """
    probed_occurs = 0
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # img=cv2.drawKeypoints(img2,kp2)
    # plt.imshow(img),plt.show()


    print "len(des1):", len(des1)
    print "len(des2):", len(des2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=MAX_OCCURS + 1)
    print "len(matches):", len(matches)
    # store all the good matches as per Lowe's ratio test.
    good = []


    for g in matches:
        # if m.distance < 0.7*n.distance:
        for i in range(len(g) - 1):
            if g[i].distance < 45.0:
                good.append(g[i])

                if (g[i].distance < 0.95*g[i+1].distance):
                    # good.append(g[i])
                    if i + 1 > probed_occurs:
                        probed_occurs = i + 1
                    break
            else:
                break

    # probed_occurs=3

    print "probed_occurs:", probed_occurs

    print "len(good):", len(good)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape[0:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
    matchesMask = None


    reordered_good = reorder_matches(kp1, kp2, good, probed_occurs)
    return kp1, kp2, reordered_good


def main():
    filepath1 = 'button.png'
    filepath2 = 'button_in_scene5.png'

    filepath1 = 'test2//button.png'
    filepath2 = 'test2//layout.png'

    # img1 = cv2.imread(filepath1, 0)          # queryImage
    img1 = read_transparent_png(filepath1)
    img2 = cv2.imread(filepath2,0) # trainImage
    kp1, kp2, reordered_good = do_sift(img1, img2)
    img4 = drawOrderedMatches(img1, kp1, img2, kp2, reordered_good)
    plt.imshow(img4),plt.show()
    gen_html.genHTML(img1, kp1, filepath1, img2, kp2, filepath2, reordered_good, "gen1.html")

if __name__ == "__main__":
    main()
