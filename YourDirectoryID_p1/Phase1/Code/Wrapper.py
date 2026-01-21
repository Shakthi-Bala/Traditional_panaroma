#!/usr/bin/evn python

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

import numpy as np
import cv2
import os

# Add any python libraries here


def imregionalmax(input, nLocMax=100, threshold=0.0, min_dist=10):
    "Iteratively finds global maxima and suppresses region around them"
    temp_mat= input.copy()
    locations=[]

    for i in range(nLocMax):
        _,max_val,_,max_loc=cv2.minMaxLoc(temp_mat)

        #check threshold
        if max_val<=threshold:
            break
        locations.append(max_loc)

        cv2.circle(temp_mat, max_loc, int(min_dist),0.0,-1)
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
    train_dir="Traditional_panaroma/YourDirectoryID_p1/Phase1/Data/Train/Set1"

    file_names=sorted(os.listdir(train_dir))



    image_list=[]
    for file in file_names:
        file_path=os.path.join(train_dir,file)
        img=cv2.imread(file_path, flags=cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            print("Image loaded")
            image_list.append(img)
        else:
            print("Error in loading image")
        
    
    
    



    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    for i,img in enumerate(image_list):
        # Harris corner detector 
        blockSize = 2
        k_size=3 # aperture param for Sobel 
        k=0.05
        # corners_image=cv2.cornerHarris(img, blockSize=blockSize,ksize=k_size,k=k)
        # corners_norm=cv2.normalize(corners_image, None, 0, 255, cv2.NORM_MINMAX)


        # Better corners (Shi-Tomasi)
        # not using as harder to use during ANMS
        # corners=cv2.goodFeaturesToTrack(img,maxCorners=1000,qualityLevel=0.01,minDistance=10)

        # bgr_img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # if corners is not None:

        #     corners=np.int32(corners)

        #     for corner in corners:
        #         x,y=corner.ravel()
                
        #         cv2.circle(bgr_img,(x,y),3,(0,0,255),-1)


        # Using shi tomasi heatmap
        corners=cv2.cornerMinEigenVal(img,blockSize=blockSize,ksize=k_size)
        corners_norm=cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)

        # save corner image
        save_file_path=f"Traditional_panaroma/YourDirectoryID_p1/Phase1/Code/Image_results/corners{i:03d}.png"
        cv2.imwrite(save_file_path,corners_norm)
        

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """

        local_maxima=imregionalmax(corners, nLocMax=1000,threshold=0.01, min_dist=5)

        N_strong=len(local_maxima)
        r=[float('inf')]*N_strong
        
        for k in range(N_strong):
            xi, yi= local_maxima[k]

            score_i=corners[yi,xi]

            for j in range(N_strong):
                xj,yj=local_maxima[j]
                score_j=corners[yj,xj]

                if score_j>score_i:
                    ED=(xj-xi)**2 + (yj-yi)**2

                    if ED<r[k]:
                        r[k]=ED
        

        maxima_with_radius=list(zip(r,local_maxima))

        maxima_with_radius.sort(key=lambda x:x[0],reverse=True)
        top_n=100
        anms_corners=maxima_with_radius[:top_n]

        anms_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        for radius, point in anms_corners:
            x,y =point
            cv2.circle(anms_img, (x,y),3,(0,0,255),-1)

        # save corner image
        save_file_path=f"Traditional_panaroma/YourDirectoryID_p1/Phase1/Code/Image_results/anms{i:03d}.png"
        cv2.imwrite(save_file_path,anms_img)





    

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """

        """
        Refine: RANSAC, Estimate Homography
        """

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """


if __name__ == "__main__":
    main()
