# Hybrid-Image
![image](https://user-images.githubusercontent.com/83058686/225349186-146866bc-45ce-4012-ae5f-f1a754bd585b.png)
![image](https://user-images.githubusercontent.com/83058686/225349328-eb73f52a-a556-44e1-82c6-28df15fdd201.png)
Based on the "Hybrid Images" article by Aude Oliva et al. (2006), we construct hybrid images by the sum of a low pass filtered version of a first image and a high pass filtered version of another image.  
We can make the low pass and high pass filtered images in both the frequency domain and spatial domain. 

## Image Filtering
Image filtering is a fundamental image processing tool. Although the SciPy package has efficient functions to perform image filtering, we wrote our own convolution filtering from scratch. The following steps were taken to perform this task:
- item Return an error message for even-dimension filters.
- Add a dimension to grayscale images to make them compatible with color images.
- Pad the input image with zeros.
- Convolve the image through three nested for loops.

The code is written in a function named my_imfilter(img, filter) which received the image and a matrix filter as its arguments. The function supports arbitrary shaped odd-dimension kernels and returns a filtered image that is the same resolution as the input image.

Applying an identity (impulse response) kernel, the result is shown below:
![image](https://user-images.githubusercontent.com/83058686/225347909-9d977b15-ad45-4290-b651-5e859c57d5f2.png)

The above result is independent of the size of the kernel and the filtered image should be identical to the original image.
Applying a 5x7 box kernel (which is a low pass kernel), the filtered blurred result is as follows:
![image](https://user-images.githubusercontent.com/83058686/225348036-b60a3191-14ed-4a34-9b22-6acb85163b35.png)

## Frequency Domain
The following steps should be taken to perform this task:
- Add a dimension to grayscale images to make them compatible with color images.
- Construct a Gaussian mask.
- Apply fft, fftshift, mask (low \& high filters), fftishift, and ifft respectively for each color channel.  
The results of the low pass and high pass filtering in the frequency domain for different sigma values are shown in the following figures.
![image](https://user-images.githubusercontent.com/83058686/225346678-74b4d2ff-2e4a-43a4-a075-14cd4f556863.png)
![image](https://user-images.githubusercontent.com/83058686/225346803-53b3e165-0fe4-4c8d-834f-7218b20c6555.png)
![image](https://user-images.githubusercontent.com/83058686/225346931-12988d71-1dc9-473c-b70b-47f8e926b45a.png)
![image](https://user-images.githubusercontent.com/83058686/225347167-f068c9e8-9ac7-4d17-8c7e-c1481fa9392f.png)

## Spatial Domain
Performing low pass and high pass filtering in the spatial domain is very straightforward. The low-frequency image is obtained using a Gaussian filter and the high frequencies image is obtained by subtracting the low frequencies image from the original image.

The results are shown below:
![image](https://user-images.githubusercontent.com/83058686/225347705-5af5a150-dba2-4d99-b6dd-ac22de7a4ec6.png)

## Pyramid of Images
By using the built-in function pyramid\_gaussian() from skimage, we can easily make the series of images.  
The results for selected images (using the frequency domain with sigma=10) are shown below: 
![image](https://user-images.githubusercontent.com/83058686/225351385-1cc53aac-f32b-40b1-a017-cb329fff40f3.png)
![image](https://user-images.githubusercontent.com/83058686/225351584-334c820c-5714-4b34-8009-577e6543edb0.png)


![image](https://user-images.githubusercontent.com/83058686/225351907-305bea0c-9785-4369-be20-8bcb06d5d4c8.png)
![image](https://user-images.githubusercontent.com/83058686/225352022-b3022915-b2de-4117-a1da-b54216d32019.png)


![image](https://user-images.githubusercontent.com/83058686/225352169-024c0ed0-38f2-4dd5-a901-3be5581e8c2c.png)
![image](https://user-images.githubusercontent.com/83058686/225352262-0099468a-ae08-4eac-b4dc-b0f58ace7768.png)





