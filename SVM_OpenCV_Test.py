import os, glob, sys
import cv2
import numpy as np

bin_n = 16 # Number of bins

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi)) # quantizing binvalues in (0,...,16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10,10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10,10:], mag[10:, 10:]   
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) # hist is a 64 bit vector
    return hist

def main():
    img = cv2.imread('./0.png', 0)    
    
    print()
    svm = cv2.ml.SVM_load('svm_data.json')
    hist = hog(img)
    hist = np.float32(hist).reshape(-1,bin_n*4)
    pred = svm.predict(hist)
    print(pred)
    result = pred[1]
    print(result)
    #for (i, img_path) in enumerate(test_img_paths):
    print(result[0][0])
    
    cv2.imshow("img", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()