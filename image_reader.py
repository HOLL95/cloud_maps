import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import image_funcs
img = cv2.imread('bg-clouds.jpg', cv2.IMREAD_COLOR )


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
def blur_colours(image, kernel_size):
    blurred_image=np.zeros((len(image), len(image[0]), 3))
    for i in range(0, 3):
        blur_im=image[:,:,i]
        blurred_image[:,:,i]=(cv2.GaussianBlur(blur_im, (kernel_size, kernel_size), 0))
    return blurred_image

image=copy.deepcopy(img)
gauss_img=blur_colours(image, 5)
counter=1

#im_gauss=white_scanner(gauss_img, 5)
c_im_gauss=image_funcs.white_scanner(gauss_img, 30)

imgray=np.int8(c_im_gauss)

print "=="*20
print imgray.dtype



# Detect blobs.
mask_gray = cv2.normalize(src=imgray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

vis0=cv2.cvtColor(mask_gray, cv2.COLOR_RGB2GRAY)

# get contours




plt.imshow(image)
prop_cycle = plt.rcParams['axes.prop_cycle']

ret, thresh = cv2.threshold(vis0, 125, 255, 0)
# get contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color=tuple(abs(np.random.rand(3)))
x_area=[]
y_area=[]
for i in range(len(contours)):
    cimg = np.zeros_like(image)
    cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
    pts = np.where(cimg == 255)
    x_area.append(pts[0])
    y_area.append(pts[1])
#print lst_intensities

for con in contours:
    x=[]
    y=[]
    for i in range(0, len(con)):
        x.append(con[i][0][0])
        y.append(con[i][0][1])

    plt.plot(x,y, color="b", lw=2)
plt.legend()
for i in range(0, len(x_area)):
    for j in range(0, len(x_area[i])):
        plt.plot(x_area[i][j], y_area[i][j], linestyle=":", color="r")
plt.show()

#plt.show()



    # Create a mask image that contains the contour filled in



    # Access the image pixels and create a 1D numpy array then add to list


#image=white_scanner(img2,8)
#laplacian = cv2.Laplacian(image[:,:,0],cv2.CV_64F)
#sobel=cv2.Sobel(image[:,:,0], cv2.CV_64F, 0, 1, ksize=5)
