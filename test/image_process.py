import cv2
import numpy as np

class Pre_Process:
    image = None

    def __init__(self, file_path):
        self.image = cv2.imread(file_path)

    def nothing(self, e):
        pass

    # Threshold
    def create_threshold_slider(self):
        cv2.namedWindow('Image Processing')
        cv2.createTrackbar('Threshold', 'Image Processing', 0, 255, self.nothing)
        
    def threshold_handle(self):
        threshold = cv2.getTrackbarPos('Threshold', 'Image Processing')
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('Image Processing', self.image)
        cv2.imshow('Thresholded Image', thresh_image)

    # RGB
    def create_rgb_slider(self):
        cv2.namedWindow('Image Processing')
        cv2.createTrackbar('Brightness', 'Image Processing', 100, 200, self.nothing)
        cv2.createTrackbar('Contrast', 'Image Processing', 100, 300, self.nothing)
        cv2.createTrackbar('Blur', 'Image Processing', 0, 20, self.nothing)
        
    def rgb_handle(self):
        brightness = cv2.getTrackbarPos('Brightness', 'Image Processing')
        contrast = cv2.getTrackbarPos('Contrast', 'Image Processing')
        blur = cv2.getTrackbarPos('Blur', 'Image Processing')

        beta = brightness - 100  # Brightness adjustment (centered at 100)
        alpha = contrast / 100.0 # Contrast adjustment (scale between 1 and 3)
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)

        if blur > 0:
            adjusted_image = cv2.GaussianBlur(adjusted_image, (2 * blur + 1, 2 * blur + 1), 0)

        cv2.imshow('Image Processing', adjusted_image) 

    # Run
    def run(self):
        self.create_threshold_slider()
        # self.create_rgb_slider()
        while True:
            self.threshold_handle()
            # self.rgb_handle()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    mod = Pre_Process("im1.jpg")
    mod.run()
