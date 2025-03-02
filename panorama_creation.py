import os
import numpy as np
import cv2
import glob
import imutils


#this code uses the opencv stitcher class to stitch images together
#it also applies a mask to the stitched image to remove unwanted areas


#Set the image directory path (Assuming the script runs from VR_assignment)
image_dir = "panoImages"

#Get all image paths (JPEG files)
images_path = glob.glob(os.path.join(image_dir, "*.jpeg"))

#Check if images are found
if not images_path:
    print("No images found! Check the path or file extensions.")
    exit()

#Load images
images = []
for image in images_path:
    img = cv2.imread(image)
    if img is None:
        print(f"Failed to load: {image}")
        continue
    images.append(img)

print(f"Number of images loaded: {len(images)}")

#Stitch images
image_stitcher = cv2.Stitcher_create()
error, stitched_img = image_stitcher.stitch(images)

if not error:
    #Save the stitched image
    cv2.imwrite("stitchedOutput.png", stitched_img)
    #print("Stitched image saved as 'stitchedOutput.png'")
    output_dir = "stitching_output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stitchedOutput.png")
    cv2.imwrite(output_file, stitched_img)


    #Add a border
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    #Convert to grayscale & apply threshold
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    #Find the largest contour (Area of Interest)
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    #Create a mask around the stitched region
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    #Erode the mask to refine the region
    minRectangle = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)

    #Find contours again for the refined mask
    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    #Crop the stitched image
    x, y, w, h = cv2.boundingRect(areaOI)
    stitched_img = stitched_img[y:y + h, x:x + w]

    #save this to a folder name part2_stitching_method1_output
    #output_dir = "part2_stitching_method1_output"
    #os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stitchedOutputProcessed.png")
    cv2.imwrite(output_file, stitched_img)
    print(f"Stitched image saved as {output_file}")

else:
    print("Images could not be stitched!")
    print("Likely not enough keypoints detected.")
