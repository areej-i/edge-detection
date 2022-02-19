import cv2
import numpy as np
from math import ceil, pi, tan, e, atan

DATATYPE = np.uint8

# Returns an array of stick kernels.
# It can have more tiles in the stick than the size, but it preserves the same behaviour.
def makeStickKernels(size,stick_count):
    angle_step = 2 * pi / stick_count
    kernels = []

    for step in range(stick_count):
        angle = step * angle_step
        
        kernel = [[0 for i in range(size)] for j in range(size)]

        slope = tan(angle)
        intercept = (size / 2) * (1 - slope)

        if (abs(slope) < size):
            x_intercepts = [slope * x + intercept for x in range(size)]
            for x in range(size):
                pos = x_intercepts[x]
                if pos < size and pos >= 0:
                    kernel[x][int(pos)] = 1
                    if (x > 0):
                        kernel[x - 1][int(pos)] = 1

        if (abs(slope) > 1 / size):
            y_intercepts = [(y - intercept) / slope for y in range(size)]
            for y in range(size):
                pos = y_intercepts[y]
                if pos < size and pos >= 0:
                    kernel[int(pos)][y] = 1
                    if (y > 0):
                        kernel[int(pos)][y - 1] = 1

        kernels.append(kernel)
    
    return kernels
    
# Applies the stick filter
def stickFilter(img0,stick_count,size):
    kernels = makeStickKernels(size,stick_count)

    shape = img0.shape
    filteredImage = np.zeros((shape[0] - size + 2,shape[1] - size + 2),DATATYPE) 

    for y in range(size // 2 + 1,shape[1] - size // 2 - 1):
        for x in range(size // 2 + 1,shape[0] - size // 2 - 1):
            max_delta = 0
            for kernel in kernels:
                average = 0
                stick_average = 0
                for conv_y in range(size):
                    for conv_x in range(size):
                        image_x = x - (size // 2) + conv_x
                        image_y = y - (size // 2) + conv_y
                        value = img0[image_x,image_y]
                        if (kernel[conv_x][conv_y]):
                            stick_average += value
                        else:
                            average += value
                average /= size * size
                stick_average /= sum([sum(row) for row in kernel])
                delta = average - stick_average
                if (delta > max_delta):
                    max_delta = delta
            filteredImage[x - size // 2 + 1][y - size // 2 + 1] = int(max_delta)
    return filteredImage

# A basic convolution function for 2d kernels
def simpleConvolve(img,kernel):
    shape = img.shape
    kernel_shape = kernel.shape
    filteredImage = np.zeros((shape[0] - kernel_shape[0] + 1,shape[1] - kernel_shape[1] + 1),DATATYPE) 
    for y in range((kernel_shape[1] - 1) // 2,shape[1] - (kernel_shape[1] - 1) // 2):
        for x in range((kernel_shape[0] - 1) // 2,shape[0] - (kernel_shape[0] - 1) // 2):
            out_x = x - (kernel_shape[0] - 1) // 2
            out_y = y - (kernel_shape[1] - 1) // 2
            accumulator = 0
            for conv_x in range(kernel_shape[0]):
                for conv_y in range(kernel_shape[1]):
                    image_x = x - (kernel_shape[0] // 2) + conv_x
                    image_y = y - (kernel_shape[1] // 2) + conv_y
                    if (image_x >= 0 and image_y >= 0 and image_x < shape[0] and image_y < shape[1]):
                        accumulator += img[image_x][image_y] * kernel[conv_x][conv_y]
            filteredImage[out_x][out_y] = int(min(255,max(accumulator,0)))
    return filteredImage

# Does a separated convolution using a gaussian filter
def gaussianConvolve(img, sigma):
    size = 2 * ceil(3 * sigma) + 1
    kernel = np.zeros((size,1))
    low  = -(size - 1) // 2
    high = (size - 1) // 2 + 1
    for i in range(low,high):
        kernel[i + (size - 1) // 2][0] = (1 / (((2 * pi) ** 0.5) * sigma)) * (e ** -((i * i) / ( 2 * sigma * sigma)))
    test = simpleConvolve(img,kernel)
    img2 = simpleConvolve(test,np.transpose(kernel))
    return img2

def gradientMagnitudeThreshholding(image):
    new_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2),dtype=DATATYPE)

    # The magnitudes as pixel intensities.
    magnitude_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2),dtype=np.float32)
    
    for y in range(1,image.shape[1] - 1):
        for x in range(1,image.shape[0] - 1):
            ix = int(-image[x][y]) + int(image[x + 1][y])
            iy = int(-image[x][y]) + int(image[x][y + 1])
            magnitude = (ix * ix + iy * iy) ** 0.5
            if (magnitude < 20):
                new_image[x - 1][y - 1] = 0
            else:
                new_image[x - 1][y - 1] = image[x][y]
            magnitude_image[x - 1][y - 1] = magnitude
    return new_image, magnitude_image

def nonMaximumSupression(image):
    new_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2),dtype=DATATYPE)

    orientation_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2),dtype=DATATYPE)
    
    for y in range(1,image.shape[1] - 1):
        for x in range(1,image.shape[0] - 1):

            # Approximation for direction
            ix = int(-image[x][y]) + int(image[x + 1][y])
            iy = int(-image[x][y]) + int(image[x][y + 1])
            if (ix == 0):
                direction = pi / 2
            else:
                direction = atan(iy/ix)

            orientation_image[x - 1][y - 1] = min(max(int(((direction + pi / 2) / pi) * 255),0),255)

            # Get closest angle and see if pixels on that angle are greater
            if (direction > 3 * pi / 8 or direction < - 3 * pi / 8):
                if (image[x][y] >= image[x][y - 1] and image[x][y] >= image[x][y + 1]):
                    new_image[x - 1][y - 1] = image[x][y]
                else:
                    new_image[x - 1][y - 1] = 0
            elif (direction > pi / 8):
                if (image[x][y] >= image[x + 1][y + 1] and image[x][y] >= image[x - 1][y - 1]):
                    new_image[x - 1][y - 1] = image[x][y]
                else:
                    new_image[x - 1][y - 1] = 0
            elif (direction < -pi / 8):
                if (image[x][y] >= image[x + 1][y - 1] and image[x][y] >= image[x - 1][y + 1]):
                    new_image[x - 1][y - 1] = image[x][y]
                else:
                    new_image[x - 1][y - 1] = 0
            else:
                if (image[x][y] >= image[x + 1][y] and image[x][y] >= image[x - 1][y]):
                    new_image[x - 1][y - 1] = image[x][y]
                else:
                    new_image[x - 1][y - 1] = 0
    return new_image, orientation_image
            
        
# Main function. Chains the other function to get the final result
def myEdgeFilter(img0,sigma):
    blurred = gaussianConvolve(img0, sigma)
    print("Blurring complete.")

    # Sobel Edge detection
    # I included the reverse directions as well since it wasn't detecting very well without
    kernel_nx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_nx = simpleConvolve(blurred, kernel_nx)
    print("Negative X Sobel filter complete.")

    kernel_px = np.array([[1,0,-1],[2,0,-2],[-1,0,1]])
    sobel_px = simpleConvolve(blurred, kernel_px)
    print("Positive X Sobel filter complete.")

    kernel_ny = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_ny = simpleConvolve(blurred, kernel_ny)
    print("Negative Y Sobel filter complete.")

    kernel_py = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel_py = simpleConvolve(blurred, kernel_py)
    print("Positive Y Sobel filter complete.")

    # Combines all of the sobel edge detections into one.
    combined = sobel_nx + sobel_px + sobel_ny + sobel_py
    print("Combining edge filtered images complete.")

    # Cuts off areas with minimal magnitude in the gradient
    magnitude_threshold, magnitude_image = gradientMagnitudeThreshholding(combined)
    print("Magnitude threshholding complete.")

    # Applying stick filter
    sticks_filtered = stickFilter(magnitude_threshold,5,8)
    print("Sticks filter applied")

    # Suppressing maximums
    final_sticks, _ = nonMaximumSupression(sticks_filtered)
    print("Supression complete.")

    final_no_sticks, orientation_image = nonMaximumSupression(magnitude_threshold)
    print("Supression complete.")
    
    return final_no_sticks, final_sticks, magnitude_image, orientation_image, combined

# Calling the above functions
file = "cat2.jpg"
image = cv2.imread(file,0)
no_sticks, sticks, magnitude, orientation,edges = myEdgeFilter(image,3)
cv2.imwrite(file[:-4]+"_Edges.jpg",edges)
cv2.imwrite(file[:-4]+"_No_sticks.jpg",no_sticks)
cv2.imwrite(file[:-4]+"_Sticks.jpg",sticks)
cv2.imwrite(file[:-4]+"_Orientation.jpg",orientation)
cv2.imwrite(file[:-4]+"_magnitude.jpg",magnitude)
