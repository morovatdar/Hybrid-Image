from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import time

def my_imfilter(img, filter):
    # Check for even-dimension filters
    f_rows, f_cols = filter.shape
    if not(f_rows % 2 or f_cols % 2): 
        print('The filter array has incorrect shape! The same image is returned.')
        return img
    
    # Add a dimension to grayscale images to make them compatible with color images
    if len(img.shape) == 2: img = np.expand_dims(img, 2)
    I_rows, I_cols, I_chans = img.shape
    
    # Zero pad the image
    padded_I = np.pad(img, ((f_rows//2, f_rows//2), (f_cols//2, f_cols//2), (0,0)))
    
    # Performing the convolution with the kernel
    filtered_I = np.zeros((I_rows, I_cols, I_chans))
    for i in range(I_rows):
        for j in range(I_cols):
            for k in range(I_chans):
                filtered_I[i,j,k] = np.sum(padded_I[i:i+f_rows, j:j+f_cols, k] * filter)
    return filtered_I


def test_my_imfilter():
    # Load the image
    root = 'C:/SDSMT/Courses/Vision 514/Projects/Project1/data/'
    img_name = input('Please enter the image name: ')
    I = io.imread(root + img_name + '.bmp') / 255

    # Make the filter (blur/ identity)
    f_r, f_c = input('Please enter the filter size (rows  columns): ').split()
    f_rows, f_columns = eval(f_r), eval(f_c)
    filter = np.ones((f_rows, f_columns)) / (f_rows* f_columns)
    # filter = np.pad([[1]], ((f_rows//2, f_rows//2), (f_columns//2, f_columns//2)))

    # Call my_imfilter function and the Scipy convolution function to compare the performance
    my_filtered_I = my_imfilter(I, filter)
    if len(I.shape) == 3: filter = np.expand_dims(filter, 2)
    py_filtered_I = convolve(I, filter, mode='constant', cval=0.0)
    
    # Plot the results
    ax0 = plt.subplot(131)
    ax0.imshow(I)
    ax0.set_title('Original Image')
    plt.axis('off')
    ax1 = plt.subplot(132)
    ax1.imshow(my_filtered_I)
    ax1.set_title('My Filter')
    plt.axis('off')
    ax2 = plt.subplot(133)
    ax2.imshow(py_filtered_I)
    ax2.set_title('SciPy Filter')
    plt.axis('off')
    plt.show()


def Gaussian_Mask(size, sigma):
    rows, cols = size
    y_c, x_c = rows/2, cols/2 # Centers
    mask = np.zeros(size)
    for x in range(cols):
        for y in range(rows):
            mask[y,x] = np.exp((-((y-y_c)**2 + (x-x_c)**2)/(2*(sigma**2))))
    # mask = mask / np.sum(mask)
    return mask


def scale_image(I):
    return (I - np.min(I)) / (np.max(I) - np.min(I))
    
    
def frequency_filter(I, sigma, plot=False):
    # Add a dimension to grayscale images to make them compatible with color images
    Ishape = I.shape
    if len(Ishape) == 2: I = np.expand_dims(I, 2)
    
    size = 2 * sigma + 1 if sigma % 2 == 0 else 2 * sigma
    mask = Gaussian_Mask([size, size], sigma)
    mask = np.pad(mask, (((Ishape[0]-size)//2, Ishape[0]-size-(Ishape[0]-size)//2), ((Ishape[1]-size)//2, Ishape[1]-size-(Ishape[1]-size)//2)))
    
    # Initializing images and spectra
    I_fft = np.zeros(Ishape)
    I_ffts = np.zeros(Ishape)
    I_fftslf = np.zeros(Ishape)
    I_fftlf = np.zeros(Ishape)
    I_lf = np.zeros(Ishape)
    I_fftshf = np.zeros(Ishape)
    I_ffthf = np.zeros(Ishape)
    I_hf = np.zeros(Ishape)
    
    # Applying fft, fftshift, mask (low & high filters), fftishift, and ifft for each color channel
    for i in range(Ishape[2]):
        fft = np.fft.fft2(I[:,:,i])
        ffts = np.fft.fftshift(fft)
        fftslf = ffts * mask
        fftshf = ffts * (1 - mask)
        fftlf = np.fft.ifftshift(fftslf)
        ffthf = np.fft.ifftshift(fftshf)
        I_lf[:,:,i] = np.fft.ifft2(fftlf).real
        I_hf[:,:,i] = np.fft.ifft2(ffthf).real
        I_fft[:,:,i] = 20 * np.log(abs(fft)) 
        I_ffts[:,:,i] = 20 * np.log(abs(ffts)) 
        I_fftslf[:,:,i] = 20 * np.log(0.001+abs(fftslf)) 
        I_fftlf[:,:,i] = 20 * np.log(0.001+abs(fftlf)) 
        I_fftshf[:,:,i] = 20 * np.log(0.001+abs(fftshf)) 
        I_ffthf[:,:,i] = 20 * np.log(0.001+abs(ffthf)) 
        
    I_hf = scale_image(I_hf)
    
    if plot: 
        # Scale the values to the interval [0..1]
        I_fft = scale_image(I_fft)
        I_ffts = scale_image(I_ffts)
        I_fftslf = scale_image(I_fftslf)
        I_fftlf = scale_image(I_fftlf)
        I_fftshf = scale_image(I_fftshf)
        I_ffthf = scale_image(I_ffthf)
        
        # Plot the result
        cmap = 'gray' if I.shape[2] == 1 else None
        ax0 = plt.subplot(331)
        ax0.imshow(I, cmap=cmap)
        ax0.set_title('Original Image', fontsize = 7)
        plt.axis('off')
        ax1 = plt.subplot(332)
        ax1.imshow(I_fft, cmap=cmap)
        ax1.set_title('Spectrum', fontsize = 7)
        plt.axis('off')
        ax2 = plt.subplot(333)
        ax2.imshow(I_ffts, cmap=cmap)
        ax2.set_title('Centered Spectrum', fontsize = 7)
        plt.axis('off')
        ax3 = plt.subplot(334)
        ax3.imshow(I_fftslf, cmap=cmap)
        ax3.set_title('Low Filtered Centered Spectrum', fontsize = 7)
        plt.axis('off')
        ax4 = plt.subplot(335)
        ax4.imshow(I_fftlf, cmap=cmap)
        ax4.set_title('Low Filtered Spectrum', fontsize = 7)
        plt.axis('off')
        ax5 = plt.subplot(336)
        ax5.imshow(I_lf, cmap=cmap)
        ax5.set_title('Low Filtered Image', fontsize = 7)
        plt.axis('off')
        ax6 = plt.subplot(337)
        ax6.imshow(I_fftshf, cmap=cmap)
        ax6.set_title('High Filtered Centered Spectrum', fontsize = 7)
        plt.axis('off')
        ax7 = plt.subplot(338)
        ax7.imshow(I_ffthf, cmap=cmap)
        ax7.set_title('High Filtered Spectrum', fontsize = 7)
        plt.axis('off')
        ax8 = plt.subplot(339)
        ax8.imshow(I_hf, cmap=cmap)
        ax8.set_title('High Filtered Image', fontsize = 7)
        plt.axis('off')
        plt.show()
    else:
        return I_lf, I_hf
    

def test_frequency_filter():
    # Load the image
    root = 'C:/SDSMT/Courses/Vision 514/Projects/Project1/data/'
    img_name = input('Please enter the image name: ')
    I = io.imread(root + img_name + '.bmp') / 255
    
    sigma = eval(input('Please enter Sigma: '))
    
    # Call frequency_filter function
    frequency_filter(I, sigma, plot=True)
    # lf, hf = frequency_filter(I, sigma, plot=False)
    # img = lf+hf
    # img = scale_image(img)
    # plt.imshow(img)
    # plt.show()


def spatial_filter(I, sigma, plot=True):
    size = 3 * sigma + 1 if sigma % 2 == 0 else 3 * sigma
    kernel = Gaussian_Mask([size, size], sigma)
    
    # Call my_imfilter function
    lf = my_imfilter(I, kernel)
    lf = scale_image(lf)
    hf = scale_image(I - lf)
    
    if plot:
        ax0 = plt.subplot(131)
        ax0.imshow(I)
        ax0.set_title('Original Image', fontsize = 7)
        plt.axis('off')
        ax1 = plt.subplot(132)
        ax1.imshow(lf)
        ax1.set_title('Low pass filter', fontsize = 7)
        plt.axis('off')
        ax2 = plt.subplot(133)
        ax2.imshow(hf)
        ax2.set_title('High pass filter', fontsize = 7)
        plt.axis('off')
        plt.show()
    else:
        return(lf, hf)
    

def test_spatial_filter(plot=True):
    # Load the image
    root = 'C:/SDSMT/Courses/Vision 514/Projects/Project1/data/'
    img_name = input('Please enter the image name: ')
    I = io.imread(root + img_name + '.bmp') / 255
    
    sigma = eval(input('Please enter Sigma: '))
    
    spatial_filter(I, sigma, plot=True)
    
    
def hybrid_image(I1, I2, sigma, method='frequency'):
    if method == 'frequency':
        I1_lf, _ = frequency_filter(I1, sigma, plot=False)
        _, I2_hf = frequency_filter(I2, sigma, plot=False)
    else:
        I1_lf, _ = spatial_filter(I1, sigma, plot=False)
        _, I2_hf = spatial_filter(I2, sigma, plot=False)
    
    I_hybrid = I1_lf + I2_hf
    I_hybrid = scale_image(I_hybrid)
    
    rows, cols, _  = I1.shape
    
    pyramid = tuple(transform.pyramid_gaussian(I_hybrid, max_layer=4, downscale=2, channel_axis=-1))

    composite_image = np.ones((rows, 2 * cols, 3), dtype=np.double)

    composite_image[:rows, :cols, :] = pyramid[0]

    i_col = int(1.01 * cols)
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[-n_rows:, i_col:i_col + n_cols] = p
        i_col += int(n_cols + 0.01 * cols)

    plt.imshow(composite_image)
    plt.axis('off')
    plt.show()
    

def test_hybrid_image():
    # Load the images
    root = 'C:/SDSMT/Courses/Vision 514/Projects/Project1/data/'
    I1_name = 'submarine'
    I2_name = 'fish'
    I1 = io.imread(root + I1_name + '.bmp') / 255
    I2 = io.imread(root + I2_name + '.bmp') / 255
    
    sigma = 15
    
    hybrid_image(I1, I2, sigma, method='frequency')
    
    
def test_time():
    # Load the image
    root = 'C:/SDSMT/Courses/Vision 514/Projects/Project1/data/'
    img_name = input('Please enter the image name: ')
    I = io.imread(root + img_name + '.jpg') / 255

    times1, times2 = np.zeros(7), np.zeros(7)
    x1 = np.arange(3,16,2)
    I_small = transform.resize(I, (I.shape[0] // 4, I.shape[1] // 4), anti_aliasing=True)
    
    for i in range(7):
        filter = np.ones((x1[i], x1[i])) / (x1[i]* x1[i])
        
        t0 = time.time()
        my_filtered_I = my_imfilter(I_small, filter)
        times1[i] = time.time() - t0
        
        if len(I.shape) == 3: filter = np.expand_dims(filter, 2)
    
        t0 = time.time()
        py_filtered_I = convolve(I_small, filter, mode='constant', cval=0.0)
        times2[i] = time.time() - t0
        
    plt.plot(x1, times1, label='My Filter')
    plt.plot(x1, times2, label='SciPy Filter')
    plt.xlabel('Kernel Size')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    
    times3, times4 = np.zeros(6), np.zeros(6)
    x2 = [8, 4, 2, 1, 0.5, 0.25]
    filter = np.ones((3, 3)) / 9
    filter2 = np.expand_dims(filter, 2)
    
    for i in range(6):
        I_small = transform.resize(I, (I.shape[0] // np.sqrt(2)**i, I.shape[1] // np.sqrt(2)**i), anti_aliasing=True)
        
        t0 = time.time()
        my_filtered_I = my_imfilter(I_small, filter)
        times3[i] = time.time() - t0
        
        t0 = time.time()
        my_filtered_I = convolve(I_small, filter2, mode='constant', cval=0.0)
        times4[i] = time.time() - t0
        
    plt.plot(x2, times3, label='My Filter')
    plt.plot(x2, times4, label='SciPy Filter')
    plt.xlabel('Megapixel')
    plt.ylabel('time')
    plt.show()
    
    
if __name__ == '__main__':
    # test_my_imfilter()
    # test_frequency_filter()
    # test_spatial_filter()
    # test_hybrid_image()
    test_time()