# Hybrid-Image
Based on the "Hybrid Images" article by Aude Oliva et al. (2006), we construct hybrid images `by the sum of a low pass filtered version of a first image and a high pass filtered version of another image.  
We can make the low pass and high pass filtered images in both the frequency domain and spatial domain. 

## Image Filtering
Image filtering is a fundamental image processing tool. Although the SciPy package has efficient functions to perform image filtering, we wrote our own convolution filtering from scratch. The following steps were taken to perform this task:
- item Return an error message for even-dimension filters.
- Add a dimension to grayscale images to make them compatible with color images.
- Pad the input image with zeros.
- Convolve the image through three nested for loops.

The code is written in a function named my_imfilter(img, filter) which received the image and a matrix filter as its arguments. The function supports arbitrary shaped odd-dimension kernels and returns a filtered image that is the same resolution as the input image.

Applying an identity (impulse response) kernel, the result is shown below:
![image](https://user-images.githubusercontent.com/83058686/225345162-fdf5a29d-9d05-4d86-8f09-fefb9d65520e.png)
