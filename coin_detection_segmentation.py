import cv2
import numpy as np
import os
from pathlib import Path

# Configuration
CONFIG = {
    "BASE_DIR": Path(__file__).resolve().parent,
    "IMAGES_DIR": "images",
    "RESULTS_DIR": "results",
    "INPUT_IMAGE": "coins2.jpeg",
    "BLUR_SIZE": (5, 5),       # Gaussian blur kernel size (smaller for smaller coins)
    "MIN_DIST": 70,            # Minimum distance between circle centers
    "PARAM1": 80,             # Canny edge high threshold
    "PARAM2": 50,             # Hough accumulator threshold
    "MIN_RADIUS": 20,         # Minimum coin radius
    "MAX_RADIUS": 100,        # Maximum coin radius
}

# Paths
IMAGES_DIR = CONFIG["BASE_DIR"] / CONFIG["IMAGES_DIR"]
RESULTS_DIR = CONFIG["BASE_DIR"] / CONFIG["RESULTS_DIR"]
INPUT_IMAGE = IMAGES_DIR / CONFIG["INPUT_IMAGE"]
OUTPUT_DETECTED = RESULTS_DIR / "detected_coins.jpg"
OUTPUT_EDGES = RESULTS_DIR / "edges.jpg"  # For debugging

def detect_coins(image_path, config):
    """Detect coins using edge detection and Hough circles."""
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config["BLUR_SIZE"], 2)
    
    # Edge detection with adaptive thresholds
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)
    cv2.imwrite(str(OUTPUT_EDGES), edges)  # Debug edges
    
    # Circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=config["MIN_DIST"],
        param1=config["PARAM1"],
        param2=config["PARAM2"],
        minRadius=config["MIN_RADIUS"],
        maxRadius=config["MAX_RADIUS"]
    )
    
    if circles is None:
        print("No circles detected. Adjust parameters.")
        return img, [], edges
    
    circles = np.uint16(np.around(circles))
    print(f"Detected {len(circles[0])} circles: {circles[0]}")
    for circle in circles[0]:
        center_x, center_y, radius = circle
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), 3)
    
    return img, circles[0], edges

def segment_coins(image, circles, edges):
    """Segment individual coins using watershed algorithm."""
    if len(circles) == 0:
        return []
    
    # Distance transform and thresholding
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Background and unknown regions
    sure_bg = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Watershed
    img_copy = image.copy()
    cv2.watershed(img_copy, markers)
    
    # Extract coins
    segmented_coins = []
    for i in range(2, np.max(markers) + 1):
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[markers == i] = 255
        coin = cv2.bitwise_and(image, mask)
        segmented_coins.append(coin)
    
    return segmented_coins

def count_coins(circles):
    """Count the total number of detected coins."""
    return len(circles)

def main():
    # Ensure directories exist
    RESULTS_DIR.mkdir(exist_ok=True)
    
    try:
        # Detect coins
        detected_img, circles, edges = detect_coins(INPUT_IMAGE, CONFIG)
        cv2.imwrite(str(OUTPUT_DETECTED), detected_img)
        
        # Segment coins
        segmented_coins = segment_coins(detected_img, circles, edges)
        for i, coin in enumerate(segmented_coins, 1):
            output_path = RESULTS_DIR / f"segmented_coin_{i}.jpg"
            # cv2.imwrite(str(output_path), coin)
        
        # Count coins
        coin_count = count_coins(circles)
        print(f"Total number of coins detected: {coin_count}")
        
        # Display results
        cv2.imshow("Detected Coins", detected_img)
        cv2.imshow("Edges", edges)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
