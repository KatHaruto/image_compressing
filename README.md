# Image Compressing with L1 regularization
I use L1 regularization in compressing image in JPEG format.

## implementation
JPEG has a high image compression ratio by cutting off high frequency components of the image.
Generally, high frequency components are not important for human recognition of images because they are the boundary areas of an image where colors shift significantly.

However, there are cases where JPEG is not a good choice.

In the jpeg image compression process, the high frequency components obtained by the discrete cosine transform(DCT) are reduced to zero by quantization.

I represented the discrete cosine coefficients and frequency components as a product of matrices and vectors, and used L1 regularization to find discrete cosine coefficients that have essential information and are still sparse.

## getting start

The image to be compressed must be square and its size must be divisible by 8.

```
$ python jpeg_sparse_modeling.py <filename>
```

## Expected Results
- New 5 files will be saved.
  - \<filename\>_jpeg_compressed.bmp  
    compressed image using conventional JPEG 
  - \<filename\>_jpeg_compressed_with_lasso(lambda=\<lambda\>).bmp × 4  
    compressed image using JPEG + L1 regularization (ex. lambda = \[0.01, 0.1, 1, 5\])

- The following will be output to the console
  <details>
    <summary>console output</summary>



  ```
  $ python jpeg_sparse_modeling.py girl.bmp
  
  Compress image in conventional JPEG

  number of dct with zero as its coefficient 58485 / 65536

  entropy : 0.9288794657758167
  PSNR : 36.77750843254707
  SSIM : 0.9430862105560861


  Compress images in JPEG format with L1 regularization

  ------------------------------------

   Progress, 1024  /  1024 
  lambda= 0.01
  number of L1 with zero as its coefficient 13859 / 65536

  entropy : 4.366387036134496
  PSNR : 58.48523352870478
  SSIM : 0.999308261902294

  ------------------------------------

   Progress, 1024  /  1024 
  lambda= 0.1
  number of L1 with zero as its coefficient 28200 / 65536

  entropy : 3.820047667154392
  PSNR : 49.82128417094307
  SSIM : 0.9952311484637766

  ------------------------------------

   Progress, 1024  /  1024 
  lambda= 1
  number of L1 with zero as its coefficient 57331 / 65536

  entropy : 1.464101790220432
  PSNR : 36.59022349125702
  SSIM : 0.9412162315845249

  ------------------------------------

   Progress, 1024  /  1024 
  lambda= 5
  number of L1 with zero as its coefficient 63140 / 65536

  entropy : 0.5509376134781546
  PSNR : 28.26721187573728
  SSIM : 0.820069541878914
  ```  
  </details>

- 4 images will be plotted

![image](https://user-images.githubusercontent.com/74958594/119597604-5d620a00-be1c-11eb-899c-f0704f38aa1d.png)
