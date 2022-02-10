import cv2
import math
import numpy as np

img = cv2.imread("cat2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray cat (original) ",gray)

def myEdgeFilter(image, sigma):
    hsize = 2 * math.ceil(3 * sigma) + 1  #dimension of kernel
    h2 = hsize//2
    kernel = [[0] * hsize] * hsize

    #making the kernel
    for x in range(-h2, h2+1):
        for y in range(-h2, h2+1):
            e1 = 1/2*math.pi*(sigma**2)
            e2 = math.exp(-(x**2 + y**2)/2*sigma**2)
            kernel[x+h2][y+h2] = e1 * e2

    kernel = np.array(kernel)
    
    #convoluting it
    m, n = image.shape  #dimensions of the original image
    m = m - 1
    n = n - 1
    result = np.zeros((((m-hsize)+1),((n-hsize)+1)))

    for i in range(0, (m-hsize)+1):  #0 - 361
        for j in range(0, (n-hsize)+1): #0 - 410
            box_vals = image[i : hsize+i, j: hsize+j]
            newBox = box_vals * kernel
            boxSum = 0.0
            for b in range (0,hsize):
                for c in range (0,hsize):
                    boxSum = boxSum + newBox[b][c]
            boxSum = boxSum / hsize**2
            result[i][j] = boxSum
                    
    result.astype(np.uint8)
    m, n = result.shape
    m = m - 1
    n = n - 1

    # Sobel filter
    sobelx = np.array([[-1, 0, 1],[-2, 0 ,2],[-1, 0, 1]])
    sobely = np.array([[-1, -2, -1],[0, 0 ,0],[1, 2, 1]])
    ssize = 3

    # Convoluting with filter
    x_image = np.zeros((((m-hsize)+1),((n-hsize)+1)))
    y_image = np.zeros((((m-hsize)+1),((n-hsize)+1)))
    
    for i in range(0, (m-ssize)+1):  #0 - 351
        for j in range(0, (n-ssize)+1): #0 - 400
            box_vals2 = result[i : ssize+i, j: ssize+j]
            XBox = box_vals2 * sobelx
            YBox = box_vals2 * sobely
            boxSumx = 0.0
            boxSumy = 0.0
            for b in range (0,ssize):
                for c in range (0,ssize):
                    boxSumx = boxSumx + XBox[b][c]
                    boxSumy = boxSumy + YBox[b][c]

# this is commented out because i get an error
            # x_image[i][j] = boxSumx 
            # y_image[i][j] = boxSumy
            # f_image = boxSumy * boxSumx



    cv2.imshow("blur cat",result)
    # cv2.imshow("sobel cat",f_image)

    

myEdgeFilter(gray, 1)
cv2.waitKey(50000) #for some reason waitKey(0) doesnt work, so i set the program to end in 20 seconds
cv2.destroyAllWindows()
