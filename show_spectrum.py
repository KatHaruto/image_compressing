import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('my_picture.bmp', 0).astype(dtype=int)
    fimg = np.fft.fft2(img)
    fimg =  np.fft.fftshift(fimg)
    mag = 20*np.log(np.abs(fimg))
    plt.subplot(2,4,1)
    plt.title("original")
    plt.imshow(img, cmap = 'gray')
    plt.subplot(2,4,5)
    plt.imshow(mag, cmap = 'gray')
    plt.title("spectrum")


    img = cv2.imread('jpeg_converted.png', 0).astype(dtype=int)
    fimg = np.fft.fft2(img)
    fimg =  np.fft.fftshift(fimg)
    mag = 20*np.log(np.abs(fimg))
    plt.subplot(2,4,2)
    plt.title("jpeg")
    plt.imshow(img, cmap = 'gray')
    plt.subplot(2,4,6)
    plt.imshow(mag, cmap = 'gray')
    plt.title("spectrum")

    img = cv2.imread('new_jpeg__no__quantize.png', 0).astype(dtype=int)
    fimg = np.fft.fft2(img)
    fimg =  np.fft.fftshift(fimg)
    mag = 20*np.log(np.abs(fimg))
    plt.subplot(2,4,3)
    plt.title("new jpeg (no quantize)")
    plt.imshow(img, cmap = 'gray')
    plt.subplot(2,4,7)
    plt.title("spectrum")
    plt.imshow(mag, cmap = 'gray')

    img = cv2.imread('new_jpeg.png', 0).astype(dtype=int)
    fimg = np.fft.fft2(img)
    fimg =  np.fft.fftshift(fimg)
    mag = 20*np.log(np.abs(fimg))
    plt.subplot(2,4,4)
    plt.title("new jpeg")
    plt.imshow(img, cmap = 'gray')
    plt.subplot(2,4,8)
    plt.title("spectrum")
    plt.imshow(mag, cmap = 'gray')
    plt.show()


if __name__ == "__main__":
    main()