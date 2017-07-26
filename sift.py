import numpy as np
from funcs import reorder_matches, drawOrderedMatches, drawMatches
import cv2
from matplotlib import pyplot as plt
import gen_html


MIN_MATCH_COUNT = 10

filepath1 = 'box.png'
filepath2 = 'box_in_scene3.png'
img1 = cv2.imread(filepath1,0)          # queryImage
# img1.filepath = filepath1
img2 = cv2.imread(filepath2,0) # trainImage
# img2.filepath = filepath2

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# img=cv2.drawKeypoints(img2,kp2)
# plt.imshow(img),plt.show()


print "len(des1):", len(des1)
print "len(des2):", len(des2)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=4)
print "len(matches):", len(matches)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n,o,p in matches:
    # if m.distance < 0.85*n.distance:
    # if m.distance < 0.7*n.distance:
    if o.distance < 0.7 * p.distance:
        good.append(m)
        good.append(n)
        good.append(o)
print "len(good):", len(good)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

reordered_good = reorder_matches(kp1, kp2, good, 3)
# img3 = drawMatches(img1, kp1, img2, kp2, good)
# plt.imshow(img3),plt.show()

img4 = drawOrderedMatches(img1, kp1, img2, kp2, reordered_good)
plt.imshow(img4),plt.show()


gen_html.genHTML(img1, kp1, filepath1, img2, kp2, filepath2, reordered_good, "gen.html")

# help(plt.imshow)
# plt.imshow(img3, 'gray'),plt.show()
# plt.imshow(img3),plt.show()





