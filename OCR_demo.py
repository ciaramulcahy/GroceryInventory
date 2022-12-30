#Building a Document Scanner App using Python, OpenCV, and Computer VisionPython

# Import the necessary packages
from transformV2 import four_point_transform, outlineImage, rotClockw90, defaultRectangleContour, findContours
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from PIL import Image
import pytesseract
import os

# Construct the argument parser 
ap = argparse.ArgumentParser()
# Parse the arguments
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
'''ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")'''
args = vars(ap.parse_args())

# Load the receipt image
image = cv2.imread(args["image"])

ratio = image.shape[0] / 500.0					# Compute the ratio of the old height 
orig = image.copy()								# Save the original
image = imutils.resize(image, height = 500)		# Resize the image; first time

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	# grayscale the image: image --> gray
gray = cv2.GaussianBlur(gray, (5, 5), 0)		# blur the image
imageLined = outlineImage(gray)
edged = cv2.Canny(imageLined, 75, 200)				# find edges of blurred image: gray --> edged

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = findContours(edged.copy())
 
# loop over the contours # approximate the contour
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:			# if our approximated contour has four points,
		receiptCnt = approx 		# Assign 4-sided contour to receiptCnt
		break

# Check again if receiptCnt exists, else would throw an error
try: receiptCnt
except NameError:
	receiptCnt = defaultRectangleContour(edged)

 
# Show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [receiptCnt], -1, (0, 255, 0), 2)					# Contours drawn on image
'''cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# Four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)		# Use of orig again
# Threshold image to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Write the warped image to disk as a temporary file 
filename = "{}warp_only.png".format(os.getpid())
cv2.imwrite(filename, warped)

cv2.imshow("warped", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
T = threshold_local(warped, 11, offset = 5, method = "gaussian")				# This is only detecting edges of images, Must be addressed.
warped = (warped > T).astype("uint8") * 255

# Rotate warped if its length is greater than its height
(h, w) = warped.shape[:2]
if h < w:
	warpedUpright = rotClockw90(warped)
else:
	warpedUpright = warped
# Show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("warpedUpright", imutils.resize(warpedUpright, height = 650))

cv2.waitKey(0)
cv2.destroyAllWindows()

# Write the warped, grayscale image to disk as a temporary file 
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, warpedUpright)

# load the image as a PIL/Pillow image, apply OCR, and then delete the temp file
text = pytesseract.image_to_string(Image.open(filename))
print(args["image"])
name = args["image"]
prefix = name[:-4]
with open(prefix + "_output.TXT", 'a') as f:
	f.write(text)
os.remove(filename)
print(text)

 
# show the output images
'''cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# To run
# python3 OCR_demo.py -i /Users/ciaramulcahy/Desktop/FoodReceiptSolve/ReceiptTests/RT_Kroger/RT3.JPG

