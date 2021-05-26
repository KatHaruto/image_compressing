import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
from cvxpy import *
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import re
import sys
import os
N = 8

np.set_printoptions(suppress=True)


class image_JPEG:
    def __init__(self,N):
        self.quantizer_table = \
            np.array([ 16, 11, 10, 16,  24,  40,  51,  61,
                       12, 12, 14, 19,  26,  58,  60,  55,
                       14, 13, 16, 24,  40,  57,  69,  56,
                       14, 17, 22, 29,  51,  87,  80,  62,
                       18, 22, 37, 56,  68, 109, 103,  77,
                       24, 35, 55, 64,  81, 104, 113,  92,
                       49, 64, 78, 87, 103, 121, 120, 101,
                      72, 92, 95, 98, 112, 100, 103,  99 ])
        
        self.N = N
        self.phi = np.array([self.phi_k(k) for k in range(N)])
        self.sparse_num = 0
        
    def phi_k(self,k):
        if k == 0:
            return np.ones(self.N) / math.sqrt(self.N)
        else:
            return math.sqrt(2/self.N)*np.cos(((2*(np.arange(self.N))+1)*k*math.pi)/(2*self.N))
    
    def compress(self,img):
        img_N_division = []
        x,y = img.shape
        if x != y or x %8 != 0 or y %8 != 0:
            raise Exception("Error : image must be square, and its each size must be divided 8")
        
        for i in range(y//8):
            for j in range(x//8):
                img_N_division.append(img[self.N*i:self.N*(i+1),self.N*j:self.N*(j+1)])


        for i in range(y//8):
            for j in range(x//8):
                qdct_coef_N_division,compressed_img_N_division = self.calculate(img_N_division[i*(y//N)+j])
                if j == 0:
                    compressed_img_N_row = compressed_img_N_division
                    qdct_coef_N_row = qdct_coef_N_division
                else:
                    compressed_img_N_row = np.concatenate([compressed_img_N_row, compressed_img_N_division], 1)
                    qdct_coef_N_row = np.concatenate([qdct_coef_N_row, qdct_coef_N_division], 1)
            if i == 0:
                compressed_img = compressed_img_N_row
                qdct_coef = qdct_coef_N_row
            else:
                compressed_img = np.concatenate([compressed_img,  compressed_img_N_row])
                qdct_coef = np.concatenate([qdct_coef, qdct_coef_N_row])

        compressed_img = correct_abnormal_value(compressed_img.astype(dtype=int))
        qdct_coef = qdct_coef.astype(dtype=int)
        return qdct_coef, compressed_img

    def calculate(self,img):
        dct_coef = self.dct(img)
        
        quantized_dct = self.quantize(dct_coef)

        self.sparse_num += quantized_dct.size-np.count_nonzero(quantized_dct)
        inv_quantized_dct = self.inv_quantize(quantized_dct)

        invdct_coef = self.idct(inv_quantized_dct)
        
        return quantized_dct,invdct_coef

    def get_num_nonzero_in_arr(arr):
        arr = arr.flatten()
        return len(arr) - np.count_nonzero(arr)
    
    def dct(self,img):
        img -= 128
        return np.dot(np.dot(self.phi,img),self.phi.T)
    def idct(self,inquimg):
        return np.round(np.dot(np.dot(self.phi.T,inquimg),self.phi)) + 128
    
    def quantize(self,dctimg):
        return np.array(np.split(np.round(np.array(dctimg.flatten()) / self.quantizer_table),8))
    def inv_quantize(self,quimg):
        return np.array(np.split(np.array(quimg.flatten()) * self.quantizer_table,8))

class image_sparse:
    def __init__(self,N):
        self.N = N
        self.phi = np.zeros((N**2,N**2))
        self.sparse_num = 0
        for i in range(N**2):
            for j in range(N**2):
                self.phi[i][j] = self.phi_k_i(j//N,i//N)*self.phi_k_i(j%N,i%N)

    def phi_k_i(self,k,i):
        if k == 0:
            return 1 / math.sqrt(self.N)
        else:
            return math.sqrt(2/self.N)*math.cos(((2*i+1)*k*math.pi)/(2*self.N))
        
    
    def compress(self,img,lam):
        img_N_division = []
        x,y = img.shape
        if x != y or x %8 != 0 or y %8 != 0:
            raise Exception("Error : Error : image must be square, and its each size must be divided 8")
        
        for i in range(y//8):
            for j in range(x//8):
                img_N_division.append(img[self.N*i:self.N*(i+1),self.N*j:self.N*(j+1)])

        for i in range(y//8):
            for j in range(x//8):
                dct_coef_N_division,compressed_img_N_division = self.calculate(img_N_division[i*(y//N)+j],lam)

                if j == 0:
                    compressed_img_row = compressed_img_N_division
                    dct_coef_row = dct_coef_N_division
                else:
                    compressed_img_row = np.concatenate([compressed_img_row, compressed_img_N_division], 1)
                    dct_coef_row = np.concatenate([dct_coef_row, dct_coef_N_division], 1)

                print("\r","Progress,",i*(x//8)+j+1," / ",(y//8)*(x//8),"",end='')

            if i == 0:
                compressed_img = compressed_img_row
                dct_coef = dct_coef_row
            else:
                compressed_img = np.concatenate([compressed_img,  compressed_img_row])
                dct_coef = np.concatenate([dct_coef,  dct_coef_row])
            

        print("")
            
        compressed_img = correct_abnormal_value(compressed_img.astype(dtype=int))
        dct_coef = dct_coef.astype(dtype=int)
        return dct_coef,compressed_img

    
    def calculate(self,img,lam):
        dct_coef = self.opt_dct_coef(img,lam)
        self.sparse_num += self.get_num_nonzero_in_arr(dct_coef)
        invdct_coef = self.idct(dct_coef)
        
        return dct_coef,invdct_coef
    
    def get_num_nonzero_in_arr(self,arr):
        arr = arr.flatten()
        return len(arr) - np.count_nonzero(arr)

    def opt_dct_coef(self,img,lam):
        #行列Wを生成
        img -= 128
        W = self.phi

        #変数xを定義
        x = Variable(self.N ** 2)
        y = img.flatten()
        #ハイパーパラメータを定義
        lamda = Parameter(nonneg=True)
        lamda.value =lam

        #目的関数を定義
        objective = Minimize(sum_squares((y - W @ x))/(2*N)+ lamda*norm(x, 1))
        p = Problem(objective)

        #最適化計算
        result = p.solve()
        dct_coef = np.round(x.value)
        return np.array(np.split(dct_coef,8)) 
    
    def idct(self,dct_coef):
        invdct = np.dot(self.phi, dct_coef.flatten())
        return np.round(np.split(invdct,8)) + 128
    
def calc_entropy(img):
    l = img.shape[0]

    if np.min(img) < 0:
        img -= np.min(img)

    histgram = [0]*(np.max(img)+1)

    for i in range(l):
        for j in range(l):
            v =img[i, j]
            histgram[v] += 1

    size = img.shape[0] * img.shape[1]
    entropy = 0

    for i in range(len(histgram)):

        p = histgram[i]/size
        if p == 0:
            continue
        entropy -= p*math.log2(p)
    return entropy

def correct_abnormal_value(img):
    l = img.shape[0]
    for i in range(l):
        for j in range(l):
            if img[i,j] < 0:
                img[i,j] = 0
            if img[i,j] > 255:
                img[i,j] = 255

    return img
    

def readimg(filename):
    return cv2.imread("./original/"+filename, 0).astype(dtype=int)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not len(sys.argv) == 2:
        raise Exception('Error!: two arguments are required')

    filename = sys.argv[1]
    body = re.split(r'\.',filename)[0]
    img = readimg(filename)

    os.makedirs("./images/"+body,exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    plt.subplot(1,2,1)
    plt.imshow(img,vmin=0, vmax=255)
    plt.title("original")
    plt.gray()

    image = image_JPEG(N)
    qdct_coef, compressed_img = image.compress(img)
    print("Compress image in conventional JPEG")
    print("\nnumber of dct with zero as its coefficient",image.sparse_num,"/",img.size)
    plt.subplot(1,2,2)
    plt.imshow(compressed_img,vmin=0,vmax=255)
    plt.title("jpeg")
    print("\nentropy :",calc_entropy(qdct_coef))

    img = readimg(filename)
    print("PSNR :",psnr(img,compressed_img,data_range=255))
    print("SSIM :",ssim(img,compressed_img,data_range=255))

    cv2.imwrite('./images/'+body+"/"+body+'_jpeg_compress.bmp', compressed_img)


    print("\n\nCompress images in JPEG format with L1 regularization")

    lams= [0.01, 0.1 ,1,5]
    for i,lam in enumerate(lams):
        print("\n------------------------------------\n")
        sparse = image_sparse(N)
        dct_coef,compressed_img = sparse.compress(img,lam)
        print("lambda=",lam)
        print("number of L1 with zero as its coefficient",sparse.sparse_num,"/",img.size)

        plt.subplot(2,2,i+1)
        plt.imshow(compressed_img,vmin=0,vmax=255)
        plt.title("lambda={}".format(lam))

        print("\nentropy :",calc_entropy(dct_coef))

        img = readimg(filename)

        print("PSNR :",psnr(img,compressed_img,data_range=255))
        print("SSIM :",ssim(img,compressed_img,data_range=255))

        cv2.imwrite('./images/'+body+"/"+body+'_jpeg_compress_with_lasso(lambda='+str(lam)+').bmp', compressed_img)
    plt.suptitle("JPEG format with L1 regularization")
    plt.show()
    

if __name__ == "__main__":
    main()
    