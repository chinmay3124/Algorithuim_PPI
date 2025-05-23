import cv2
import numpy as np
import math
import sys

def get_float(prompt):
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print(" Please enter a number (e.g. 10 or 10.5).")

def detect_largest_red_blob(img):
    """Return the largest red contour in img, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 100, 50]), np.array([10,255,255])
    lower2, upper2 = np.array([160,100,50]), np.array([180,255,255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def fallback_detect_dark_object(img):
    """Detect the largest dark object *not touching* the border and <80% area."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)[0]
    h, w = gray.shape
    filtered = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5000 or area > 0.8 * w * h:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x+ww >= w-2 or y+hh >= h-2:
            continue
        filtered.append(c)

    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

def main():
    # 1) Load full image
    path = r"C:\Users\HP\Downloads\test samples\for_test\13 inch\WIN_20250522_15_52_32_Pro.jpg"
    img  = cv2.imread(path)
    if img is None:
        sys.exit(f" Could not load image: {path}")

    # 2) Try red blob
    cnt = detect_largest_red_blob(img)
    if cnt is not None:
        print(" Red blob detected.")
    else:
        print(" No red detected; falling back to dark-object detection.")
        cnt = fallback_detect_dark_object(img)
        if cnt is None:
            sys.exit(" Fallback also failed to find a valid object.")

    # 3) Compute bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"Bounding box → x={x}, y={y}, w={w}, h={h}")

    # 4) Compute PPI
    diag_in    = get_float("Enter known diagonal of the object (inches): ")
    pixel_diag = math.hypot(w, h)
    ppi        = pixel_diag / diag_in
    print(f"Calculated PPI: {ppi:.1f} pixels/inch")

    # 5) Draw on full image
    out = img.copy()
    cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 3)
    cv2.putText(out,
                f"PPI: {ppi:.1f}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0,255,0),
                2)

    # 6) Display & save
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 800, 600)
    cv2.imshow("Result", out)
    cv2.imwrite("boxed_output.jpg", out)
    print(" Saved full-frame result as boxed_output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
