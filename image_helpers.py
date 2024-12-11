import cv2
import numpy as np

def highlight_green(image):
    """
    Enhance the green channel of an image to make green stand out.

    Args:
        image: Input image (BGR format as typically used by OpenCV).

    Returns:
        Processed image with enhanced green channel.
    """
    # Split the channels
    b, g, r = cv2.split(image)

    # Amplify the green channel and suppress others
    enhanced_g = cv2.multiply(g, 1)  # Amplify green (factor of 2)
    suppressed_b = cv2.multiply(b, 0.5)  # Suppress blue (factor of 0.5)
    suppressed_r = cv2.multiply(r, 0.5)  # Suppress red (factor of 0.5)

    # Clip values to ensure they stay within [0, 255]
    enhanced_g = np.clip(enhanced_g, 0, 255).astype(np.uint8)
    suppressed_b = np.clip(suppressed_b, 0, 255).astype(np.uint8)
    suppressed_r = np.clip(suppressed_r, 0, 255).astype(np.uint8)

    # Merge the channels back into a BGR image
    enhanced_image = cv2.merge((suppressed_b, enhanced_g, suppressed_r))

    return enhanced_image

def increase_saturation(image, factor=1.2):
    """
    Increase the saturation of an image.

    Args:
        image: Input image (BGR format as typically used by OpenCV).
        factor: Factor to increase the saturation (default is 1.5).

    Returns:
        Processed image with increased saturation.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels
    h, s, v = cv2.split(hsv)

    # Amplify the saturation channel
    enhanced_s = cv2.multiply(s, factor)

    # Clip values to ensure they stay within [0, 255]
    enhanced_s = np.clip(enhanced_s, 0, 255).astype(np.uint8)

    # Merge the channels back and convert to BGR
    enhanced_hsv = cv2.merge((h, enhanced_s, v))
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_image

def filter_green(image, threshold=120):
    """ if a pixel has a green value lower than the threshold, set it to black (0, 0, 0) """
    # return np.where(image[:, :, 1] < threshold, 0, image)
    image[:, :, :][image[:, :, 1] < threshold] = 0
    return image