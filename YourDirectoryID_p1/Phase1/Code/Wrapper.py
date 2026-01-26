#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import os
from matplotlib import image
import numpy as np
import cv2
import glob

# Global variable for anms points - used in feature descriptor step
# global anms_pts
# anms_pts = []

# Add any python libraries here

#Function to find local maxima coordinates
def imreginalmax(input, nloxMax = 100, minDist = 10):
    temp_img = input.copy()
    locations = []
    for i in range(nloxMax):
        _, max_val, _, max_loc = cv2.minMaxLoc(temp_img)

        if max_val <= 0:
            break
        locations.append(max_loc)
        cv2.circle(temp_img, max_loc, minDist, 0, -1)

    return locations




def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    # Shakthi 
    img_path = "/home/alien/YourDirectoryID_p1/Phase1/Data/Train/Set1"
    out_dir = "/home/alien/YourDirectoryID_p1/Phase1/Outputs"

    # Aditya      
    # img_path = "/home/adipat/Documents/Spring 26/CV/P1/YourDirectoryID_p1/Phase1/Data/Train/Set1"
    # out_dir = "/home/adipat/Documents/Spring 26/CV/P1/YourDirectoryID_p1/Phase1Outputs"

    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(img_path, "*.jpg")))


    # Shi tomasi corner detection - good features to track

    for img in img_paths:	
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = np.float32(gray)
        dst = cv2.goodFeaturesToTrack(gray,1000,0.01,10)
        for p in dst:
            x, y = p.ravel()
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        img_name = os.path.basename(img)
        out_path = os.path.join(out_dir, f"corners_{img_name}")

        cv2.imwrite(out_path, image)
        
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    # TO get ANMS points for a single image used in feature matching step
    def anms_single_image(image, num_features=150):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        score_map = cv2.cornerMinEigenVal(gray, blockSize=2, ksize=3)
        local_maxima = imreginalmax(score_map, nloxMax=300, minDist=10)

        N = len(local_maxima)
        r = [float('inf')] * N

        for i in range(N):
            xi, yi = local_maxima[i]
            si = score_map[yi, xi]

            for j in range(N):
                xj, yj = local_maxima[j]
                sj = score_map[yj, xj]

                if sj > si:
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    r[i] = min(r[i], dist)

        maxima_with_r = list(zip(local_maxima, r))
        maxima_with_r.sort(key=lambda x: x[1], reverse=True)

        return [pt for pt, _ in maxima_with_r[:num_features]]



###################################################
#---Returns ANMS points for all images---#
###################################################
    # def anms(img_paths, out_dir, num_features=100):
    #     global anms_pts

    #     for img in img_paths:
    #         image = cv2.imread(img)
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    #         score_map = cv2.cornerMinEigenVal(gray, blockSize=2, ksize=3)

    #         local_maxima = imreginalmax(score_map, nloxMax=300, minDist=10)

    #         N = len(local_maxima)
    #         r = [float('inf')] * N

    #         for i in range(N):
    #             xi, yi = local_maxima[i]
    #             si = score_map[yi, xi]

    #             for j in range(N):
    #                 xj, yj = local_maxima[j]
    #                 sj = score_map[yj, xj]

    #                 if sj > si:
    #                     dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
    #                     if dist < r[i]:
    #                         r[i] = dist

    #         maxima_with_r = list(zip(local_maxima, r))
    #         maxima_with_r.sort(key=lambda x: x[1], reverse=True)

    #         top_n = min(num_features, len(maxima_with_r))
    #         anms_pts = [maxima_with_r[i][0] for i in range(top_n)]

    #         vis = image.copy()
    #         for x, y in anms_pts:
    #             cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    #         out_path = os.path.join(out_dir, f"anms_{os.path.basename(img)}")
    #         cv2.imwrite(out_path, vis)
    # anms(img_paths, out_dir, num_features=150)
    # print("Stage1: ANMS points:", len(anms_pts))


    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    def feature_descriptor(image, keypoints_xy, patch_size=41, out_size=8):
        assert patch_size % 2 == 1, "patch size must be odd"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        half = patch_size // 2

        descriptors = []
        valid_kps = []

        for (x, y) in keypoints_xy:
            x = int(x)
            y = int(y)

            # To skip border points
            if x - half < 0 or x + half >= W or y - half < 0 or y + half >= H:
                continue

            patch = gray[y-half:y+half+1, x-half:x+half+1]

            # Gaussian blur
            patch_blur = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.0)

            # Subsample to 8x8
            patch_small = cv2.resize(
                patch_blur, (out_size, out_size), interpolation=cv2.INTER_AREA
            )

            vec = patch_small.astype(np.float32).reshape(-1)

            # Standardization
            mu = vec.mean()
            sigma = vec.std()
            if sigma < 1e-5:
                continue

            vec = (vec - mu) / sigma

            descriptors.append(vec)
            valid_kps.append((x, y))
        if len(descriptors) == 0:
            return np.zeros((0, out_size*out_size), np.float32), np.zeros((0, 2), np.int32)

        return np.array(descriptors, dtype=np.float32), np.array(valid_kps, dtype=np.int32)
    
    image = cv2.imread(img_paths[0])

    # descriptors, keypoints = feature_descriptor(image, anms_pts) # Feature Descriptor obtained here
    # print("ANMS points:", len(anms_pts))
    # print(descriptors.shape)  # (N, 64)
    # print(descriptors[0].shape) # (64,)
    # print(keypoints.shape) 

    # To get features of individual images
    features = []
    for img_path in img_paths:
        img =cv2.imread(img_path)
        anms_pts = anms_single_image(img, num_features=300)
        desc, kps = feature_descriptor(img, anms_pts)
        features.append({
            "image" : img,
            "keypoints" : kps,
            "descriptors" : desc
        })



    """
        Feature Matching
        Save Feature Matching output as matching.png
    """
    # Feature matching step
    def feature_matching(desc1, desc2, kp1, kp2, ratio_thresh=0.85):
        matches = []

        for i, d1 in enumerate(desc1):
            dists = np.linalg.norm(desc2 - d1, axis = 1)
            if len(dists) < 2:
                continue
            idxs = np.argsort(dists)
            best, second_best = dists[idxs[0]], dists[idxs[1]]
            if best / second_best < ratio_thresh:
                matches.append((i, idxs[0]))
        return matches

    # for i in range(len(features) - 1):
    #     img1 = features[i]["image"]
    #     img2 = features[i+1]["image"]

    #     kps1 = features[i]["keypoints"]
    #     kps2 = features[i+1]["keypoints"]

    #     desc1 = features[i]["descriptors"]
    #     desc2 = features[i+1]["descriptors"]

    #     matches = feature_matching(desc1, desc2, kps1, kps2, ratio_thresh=0.7)

    #     kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps1]
    #     kp2_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps2]

    #     cv_matches = [
    #         cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0)
    #         for i, j in matches
    #     ]

    #     vis = cv2.drawMatches(
    #         img1, kp1_cv,
    #         img2, kp2_cv,
    #         cv_matches, None,
    #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    #     )

    #     cv2.imwrite(f"matches_{i}_{i+1}.png", vis)

    #Getting Feature Correspondence points are array
    def build_feature_correspondence(matches, kp1, kp2):
        pts1 = []
        pts2 = []
        for i, j in matches:
            pts1.append(kp1[i])
            pts2.append(kp2[j])
        return np.array(pts1), np.array(pts2)
    
        
    """
    Refine: RANSAC, Estimate Homography
    """
    
    def computer_homography(pts1, pts2):
        A= []
        for (x,y), (xp,yp) in zip(pts1, pts2):
            A.append([-x,-y,-1,0,0,0,x*xp,y*xp,xp])
            A.append([0,0,0,-x,-y,-1,x*yp,y*yp,yp])
        A = np.array(A)
        _, _, VT = np.linalg.svd(A)
        H = VT[-1].reshape(3,3)
        return H / H[2,2]
    
    def ssd(p2, Hp1):
        return np.sum((p2 - Hp1)**2)
    
    def apply_homography(H, pts):
        x, y = pts
        p = np.array([x, y, 1.0])
        Hp = H @ p
        Hp = Hp / Hp[2]
        return Hp[:2]
    
    def ransac(kps1, kps2, matches, Nmax=2000, tau=5.0):
        best_inliers = []
        best_H = None

        pts1_all = np.array([kps1[i] for i, _ in matches])
        pts2_all = np.array([kps2[j] for _, j in matches])
        M = len(matches)

        if M < 4:
            return None, []

        for _ in range(Nmax):
            idx = np.random.choice(M, 4, replace=False)
            pts_1 = pts1_all[idx]
            pts_2 = pts2_all[idx]

            H = computer_homography(pts_1, pts_2)
            inliers = []

            for i in range(M):
                Hp1 = apply_homography(H, pts1_all[i])
                error = np.linalg.norm(pts2_all[i] - Hp1)
                if error < tau:
                    inliers.append(i)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H

        if best_H is None or len(best_inliers) < 4:
            return None, []

        H_final = computer_homography(
            pts1_all[best_inliers],
            pts2_all[best_inliers]
        )
        return H_final, best_inliers

    
    # H, inliers = ransac(kps1, kps2, matches)

    # print("Inliers:", len(inliers))
    def warp(input_mat, H_transformation):
        out_put_image=[]
        
    

    for i in range(len(features) - 1):
        img1 = features[i]["image"]
        img2 = features[i+1]["image"]

        kps1 = features[i]["keypoints"]
        kps2 = features[i+1]["keypoints"]

        desc1 = features[i]["descriptors"]
        desc2 = features[i+1]["descriptors"]

        matches = feature_matching(desc1, desc2, kps1, kps2)

        H, inliers = ransac(kps1, kps2, matches)
        print("Inliers:", len(inliers))

        cv_matches = [
            cv2.DMatch(_queryIdx=matches[i][0],
                    _trainIdx=matches[i][1],
                    _distance=0)
            for i in inliers
        ]

        kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps1]
        kp2_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps2]

        vis = cv2.drawMatches(
            img1, kp1_cv,
            img2, kp2_cv,
            cv_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        out_path = os.path.join(
            out_dir, f"ransac_matches_{i}_{i+1}.png"
        )
        cv2.imwrite(out_path, vis)
    

if __name__ == "__main__":
    main()
