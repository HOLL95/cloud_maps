import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
img = cv2.imread('bg-clouds.jpg', cv2.IMREAD_GRAYSCALE )


#width= len(img[:,:,0][0])
#height=len(img[:,:,0])
#new_arr=np.zeros((height, width))
def white_scanner(image, threshold):
    width=len(image[:,:,0][0])
    height=len(image[:,:,0])
    for i in range(0, height):
        for j in range(0, width):
            RGB_pix=image[i,j,:]
            if np.std(RGB_pix)<threshold:
                continue
            else:
                image[i,j,:]=0
    return image
def blue_scanner(image, blue_threshold, other_threshold):
    width=len(image[:,:,0][0])
    height=len(image[:,:,0])
    for i in range(0, height):
        for j in range(0, width):
            blue_pix=image[i,j,0]
            other_pix=image[i,j,1:3]
            #print blue_pix, np.mean(other_pix)
            if blue_pix>blue_threshold and np.mean(other_pix)<other_threshold:
                image[i,j,:]=0
    return image

detector = cv2.SimpleBlobDetector()
image=copy.deepcopy(img)
#image=white_scanner(img2,8)
#laplacian = cv2.Laplacian(image[:,:,0],cv2.CV_64F)
#sobel=cv2.Sobel(image[:,:,0], cv2.CV_64F, 0, 1, ksize=5)
params = cv2.SimpleBlobDetector_Params()


# Detect blobs.
keypoints = detector.detect(img)
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(sobel)
plt.show()
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
