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


# def anms(image):
    
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

    file_names=os.listdir(train_dir)



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
        corners=cv2.goodFeaturesToTrack(img,maxCorners=100,qualityLevel=0.01,minDistance=10)

        bgr_img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if corners is not None:

            corners=np.int32(corners)

            for corner in corners:
                x,y=corner.ravel()
                
                cv2.circle(bgr_img,(x,y),3,(0,0,255),-1)

        # save corner image
        save_file_path=f"Traditional_panaroma/YourDirectoryID_p1/Phase1/Code/Image_results/corners{i:03d}.png"
        cv2.imwrite(save_file_path,bgr_img)
        

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    

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
