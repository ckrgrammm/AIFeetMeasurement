try:
    import imutils
    import matplotlib
    import cv2
    import skimage
    import sklearn
    print("All packages are installed successfully.")
except ImportError as e:
    print(f"Error importing package: {e}")
