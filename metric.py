import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def em_distance(pred, gt):
    def wasserstein_distance(x, y, b, n):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
        
        x = tf.reshape(x,(b,-1))
        y = tf.reshape(y,(b,-1))

        all_values = tf.concat([x,y], axis=-1)
        all_values = tf.sort(all_values)
        reverse_all_values = all_values[:,:-1]

        deltas = tf.math.subtract(all_values[:,1:], reverse_all_values)

        x = x[:,::-1] # 오름차순
        y = y[:,::-1]
        x_cdf_indices = tf.searchsorted(x, reverse_all_values, side="right")
        y_cdf_indices = tf.searchsorted(y, reverse_all_values, side="right")
        x_cdf_indices = tf.cast(x_cdf_indices, dtype=tf.float32)
        y_cdf_indices = tf.cast(y_cdf_indices, dtype=tf.float32)

        x_cdf = tf.math.divide(x_cdf_indices, n)
        y_cdf = tf.math.divide(y_cdf_indices, n)

        output = tf.math.abs(x_cdf - y_cdf)
        output = tf.math.multiply(output, deltas)

        output = tf.math.reduce_sum(output, axis=-1)
        output = tf.reshape(output, [-1,1,1,1])

        return output

    """
    Global luminance comparisn , NO Top-K
    """
    b,h,w,c  = pred.shape
    b2,h2,w2,c2 = gt.shape
    
    assert b == b2 and c == c2, "batch size of img1 and img2 must be equal"

    n = h*w # Total pixel number
    
    pred_blue, pred_green, pred_red = tf.split(pred, num_or_size_splits=3, axis=-1)
    gt_blue, gt_green, gt_red = tf.split(gt, num_or_size_splits=3, axis=-1)

    em_distance_blue = wasserstein_distance(pred_blue, gt_blue, b, n)
    em_distance_green = wasserstein_distance(pred_green, gt_green, b, n)
    em_distance_red = wasserstein_distance(pred_red, gt_red, b, n)

    # TODO 06/04 15:17   sum -> mean
    em_distance = (em_distance_blue + em_distance_green + em_distance_red) / 3.
    
    return em_distance

def DoG(img, kernel_size=3, sigma=1.2489996, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    # Difference of Gaussian
    _,h,w,_ = img.get_shape()
    img = tf.image.resize(img, (2*h, 2*w))
    base_image = tfa.image.gaussian_filter2d(img, filter_shape=(kernel_size,kernel_size), sigma=sigma)
    # overlap sigma values in order to subtract images
    gaussian_kernels1 = [1.2262735, 1.5450078, 1.9465878, 2.452547] # base sigma = 1.6
    gaussian_kernels2 = [1.5450078, 1.9465878, 2.452547, 3.0900156]
    gaussian_images1 = [tfa.image.gaussian_filter2d(base_image, filter_shape=(kernel_size,kernel_size), sigma=gaussian_kernel, padding="REFLECT") for gaussian_kernel in gaussian_kernels1]
    gaussian_images2 = [tfa.image.gaussian_filter2d(base_image, filter_shape=(kernel_size,kernel_size), sigma=gaussian_kernel, padding="REFLECT") for gaussian_kernel in gaussian_kernels2]
    dog_image1, dog_image2, dog_image3, dog_image4 = [tf.math.subtract(second_image, first_image) for first_image, second_image in zip(gaussian_images1, gaussian_images2)]

    return dog_image1, dog_image2, dog_image3, dog_image4


def kl_divergence(I: np.ndarray, J: np.ndarray) -> float:
    
    def prob_dist(I):
      return np.histogramdd(np.ravel(I), bins = 256)[0] / I.size

    epsilon = 1e-10
    P = prob_dist(I) + epsilon
    Q = prob_dist(J) + epsilon
    return np.where(P != 0, P * np.log2(P / Q), 0).sum()