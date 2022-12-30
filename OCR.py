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
import cv2

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

# Use the four_point_transform, which depends on order_points function
# defaultRectangleContour, findContour 
# rotClockwise90, 
# outlineImage, 
def order_points(pts):		# initialzie a list of ordered coordinates 
	# first: top-left,		second: top-right, 
	# fourth: bottom-left,	third: bottom-right 
	rect = np.zeros((4, 2), dtype = "float32")
 
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]		# top-left point will have the smallest sum
	rect[2] = pts[np.argmax(s)]		# bottom-right point will have the largest sum
	# Compute the difference between the points
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]	# top-right point will have the smallest difference,
	rect[3] = pts[np.argmax(diff)]	# whereas the bottom-left will have the largest difference
	return rect 	# return the ordered coordinates

def four_point_transform(image, pts): # What data type is pts??
	# https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
	# Obtain a consistent order of the points and unpack them individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# Compute the width of the new image:
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))	# maximum distance between bottom-right and bottom-left x-coordiates
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))	# or the top-right and top-left x-coordinates
	maxWidth = max(int(widthA), int(widthB))
 
	# Compute the height of the new image:
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))	# maximum distance between the top-right and bottom-right y-coordinates
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))	# or the top-left and bottom-left y-coordinates
	maxHeight = max(int(heightA), int(heightB))
	# Construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image
	# specifying points in the top-left, top-right, bottom-right, and bottom-left order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# Compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped			# return the warped image

def outlineImage (image):
	# Draw black lines top and bottom; Add to cnts the top and bottom of the image in case is not closed
	imageLined = image.copy()
	imageLined = cv2.line(image,(0,0),(500,0),(0,0,0),2) 		# Draw a black line along the top
	imageLined = cv2.line(image,(0,500),(500,500),(0,0,0),2) 		# Draw a black line along the bottom
	imageLined = cv2.line(image,(0,0),(0,500),(0,0,0),2) 		# Draw a black line down the left side
	imageLined = cv2.line(image,(500,0),(500,500),(0,0,0),2) 	# Draw a black line down the right side
	'''cv2.imshow("corners", imageLined)
				cv2.waitKey(0)'''
	return imageLined


def defaultRectangleContour(image):	#image comes in already outlined, grayscale, blurred, edged
	# https://stackoverflow.com/questions/43009923/how-to-complete-close-a-contour-in-python-opencv	Eliezer Bernart
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
	dilated = cv2.dilate(image, kernel)
	cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	edged = cv2.Canny(image, 75, 200)				# find edges: (image, minVal, maxVal aperture_size)
								
	'''contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 	# find the contours in the edged image
	contours = imutils.grab_contours(contours)''' 												# keeping only the largest ones
	cntMax = sorted(cnts, key = cv2.contourArea, reverse = True)[:5][0]					# keep the largest 5 contours
	cntMax = np.array(cntMax)
	
	peri = cv2.arcLength(cntMax, True)		# overal perimeter of max contour
	fractDefault = 0.02
	epsilon = peri*fractDefault 			# difference from corners that is acceptable to form polygon contour
	# Approximate the contour					# 
	c = cv2.approxPolyDP(cntMax, epsilon, True)		# draw polygon with contour, perturbation of fract of arglength, closed
	#print(str(c))
	# Determine the most extreme points along the contour - there definitely is a more efficient way, but whatever
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	# [col, row]
	leftTop = (extLeft[0], extTop[1])
	rightBottom = (extRight[0], extBot[1])
	# Draw rectangles
	white = (255, 255, 255)
	black = (0, 0, 0)	
	thickness = 1
	imageBoxed = cv2.rectangle(image, leftTop, rightBottom, black, thickness) 
	leftTopInner = (extLeft[0]+1, extTop[1]+1, )
	rightBottomInner = (extRight[0]-1, extBot[1]-1)
	imageBoxed2 = cv2.rectangle(image, leftTopInner, rightBottomInner, white, thickness)
	'''print("leftTop = "+ str(leftTop))
				print("rightBot = " + str(rightBottom))'''
	cv2.imshow("ImageBoxed", imageBoxed2)		# Displaying the image 
	cv2.waitKey(0)  
	# Return largest contour, should be the rectangular one now
	cnts = findContours(imageBoxed2.copy())
	return cnts[0]

def findContours(image):
	cnts = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 	# find the contours in the edged image
	cnts = imutils.grab_contours(cnts) 												# keeping only the largest ones
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]					# Top 5 largest contours, sorted large to small
	return cnts				

def rotClockw90(img): 		# later might want to extend this to more rotation angles, see linked below
	# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
	# Get image height, width
	(h, w) = img.shape[:2]
	# calculate the center of the image
	(cX, cY) = (w // 2, h // 2) # find center coordinates
	# Perform the counter clockwise rotation holding at the center, 90 degrees
	angle270 = 270
	M = cv2.getRotationMatrix2D((cX, cY), angle270, scale = 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	# Compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# Adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	rotated270 = cv2.warpAffine(img, M, (nW, nH))
	'''
	cv2.imshow('Image rotated by 90 degrees clockwise',rotated270)
		cv2.waitKey(0) # waits until a key is pressed'''
	return rotated270
	
# To run
# python3 OCR_demo.py -i /Users/ciaramulcahy/Desktop/FoodReceiptSolve/ReceiptTests/RT_Kroger/RT3.JPG

