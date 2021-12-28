"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2 as cv
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
from random import randint
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

def sift_detector(image):
    sifty = cv.xfeatures2d.SIFT_create()
    keypoint, descriptor = sifty.detectAndCompute(image, None)
    return keypoint, descriptor

def matcher(des_left, des_right, threshold):
    matches = []
    
    for i in range(len(des_left)):
        current = []
        for j in range(len(des_right)):
            ssd = np.sum(np.square(np.subtract(des_left[i],des_right[j])))
            temp = cv.DMatch(j,i,ssd)
            current.append(temp)
            
        current = sorted(current, key = lambda x:x.distance)
        if current[0].distance < threshold*current[1].distance:
            matches.append(current[0]) 
        
    return matches 
    

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    
    kp_left , des_left = sift_detector(left_img)
    kp_right , des_right = sift_detector(right_img)
    keypoints = [kp_left, kp_right]

    matches = matcher(des_left, des_right, 0.07) 

    max_inliers = 0
    best_H = []
    for i in range(2000):
        randoms = []
        while len(randoms) != 4:
            random = matches[randint(0, len(matches)-1)]
            if random not in randoms:
                randoms.append(random)

        Homo = []
        for m in matches:
            x, y = keypoints[0][m.trainIdx].pt
            u, v = keypoints[1][m.queryIdx].pt

            Homo.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            Homo.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        
        Homo = np.asarray(Homo)
        U, S, Vh = np.linalg.svd(Homo)
        H = np.ndarray.reshape(Vh[8], (3, 3))
        H = (1/H.item(8))*H
        
        inliers = 0
        for match in matches:
            x1,y1 = keypoints[0][match.trainIdx].pt
            x2,y2 = keypoints[1][match.queryIdx].pt
            p1 = np.transpose(np.matrix([x1, y1, 1]))
            p2 = np.transpose(np.matrix([x2, y2, 1]))
            
            
            tp = np.dot(H, p1)
            if (tp.item(2) and tp.item(2) != 0):
                tp = (1 / tp.item(2)) * tp

            d = np.linalg.norm(tp - p2)
            if d < 1:
                inliers += 1
        if inliers > len(matches)*0.90:
                best_H = H 
                break              
        if inliers > max_inliers:
                max_inliers = inliers
                best_H = H
          
    
    final = best_H

    left_height, left_width = left_img.shape[1], left_img.shape[0]
    right_height, right_width = right_img.shape[1], right_img.shape[0]

    frame1 = np.float32([[0, 0], [0, left_width], [left_height, left_width], [left_height, 0]]).reshape(-1, 1, 2)
    frame2 = np.float32([[0, 0], [0, right_width], [right_height, right_width], [right_height, 0]]).reshape(-1, 1, 2)
    frame2_transformed = cv.perspectiveTransform(frame2, final)
    final_frame = np.vstack((frame1, frame2_transformed))
    
    [x, y] = np.int32(final_frame.max(axis=0).flatten())
    [xi, yi] = np.int32(final_frame.min(axis=0).flatten())
    
    td = [-xi, -yi]
    h_translation = np.array([[1, 0, td[0]], [0, 1, td[1]], [0, 0, 1]])
    result = cv.warpPerspective(left_img, h_translation.dot(final), (x - xi, y - yi))
    result[td[1]:left_width + td[1], td[0]:left_height + td[0]] = right_img
        
    result_img = result
    
    return result_img
    



    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


