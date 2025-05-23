import cv2
import math


img = cv2.imread(r"C:\Users\HP\Downloads\WIN_20250520_14_45_58_Pro.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gr, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find object outlines
shapes, gr = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Remove big shapes (like full background)
good_shapes = []
for s in shapes:
    area = cv2.contourArea(s)
    if area < 0.6 * img.shape[0] * img.shape[1]:
        good_shapes.append(s)

# If any good shape found
if good_shapes:
    # Pick the biggest one
    biggest = max(good_shapes, key=cv2.contourArea)

    # Get position and size of that shape
    x, y, w, h = cv2.boundingRect(biggest)

    # blue outline
    cv2.drawContours(img, [biggest], -1, (255, 0, 0), 2)

    #green
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    daigonal_size = 15.55 #(inches  as input)

    ppi = math.sqrt((w**2) + (h**2))/ daigonal_size
    print(f"Detected box: w={w}, h={h}, PPI={ppi}")

# Show the result
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
