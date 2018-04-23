import cv2
import numpy as np


image = cv2.imread("image/0802.png")

filter1 = np.array(
    [[0,0,0],
     [0,1,0],
     [0,0,0]
    ]
)

res1 = cv2.filter2D(image, -1 , filter1)
cv2.imwrite('image/1.png', res1)

filter2 = np.array(
    [[-1,-1,1],
     [-1,0,1],
     [0,1,1]
    ]
)

res2 = cv2.filter2D(image, -1 , filter2)
cv2.imwrite('image/2.png', res2)

filter3 = np.array(
    [[1,1,1],
     [1,-7,1],
     [1,1,1]
    ]
)

res3 = cv2.filter2D(image, -1 , filter3)
cv2.imwrite('image/3.png', res3)

filter4 = np.array(
    [[-1,-1,-1,-1,-1],
     [-1, 2, 2, 2,-1],
     [-1, 2, 8, 2,-1],
     [-1, 2, 2, 2,-1],
     [-1,-1,-1,-1,-1]
    ]
)

res4 = cv2.filter2D(image, -1 , filter4)
cv2.imwrite('image/4.png', res4)
