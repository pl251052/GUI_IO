import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def process_frame(frame, display, compass, template_counter, template_counter2,
                  template_threshold, template_threshold2, compass_switch_threshold,
                  px1_below_threshold_count, px1_threshold, temp, W, H, temp2, thresh, thresh2):
    # Define perspective transformation points
    pts1 = np.float32([[600, 400], [1300,450], [450, 900], [1380, 900]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (1080, 1920))

    # Detect lines on transformed video
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 175)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=25, minLineLength=10, maxLineGap=250)

    # Reverse perspective transformation for lines
    positives = []
    negatives = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Finds slope and prevents division by 0 error
            try:
                m = (y2-y1)/(x2-x1)
            except ZeroDivisionError:
                m = 99999999999999999
            # Uses slope to split up lines, postives are left lane and negatives are right
            if (m < -2.5 and m > -18):
                negatives.append([x1, y1, x2, y2])
            elif (m > 1 and m < 4):
                positives.append([x1, y1, x2, y2])
        # Initialize all the values used to calculate line for each side
        px1 = 0
        py1 = 0
        px2 = 0
        py2 = 0
        nx1 = 0
        ny1 = 0
        nx2 = 0
        ny2 = 0
        for x in range(len(positives)//1):
            line = positives[x]
            px1 = px1 + line[0]
            py1 = py1 + line[1]
            px2 = px2 + line[2]
            py2 = py2 + line[3]
        count = 0
        for x in range(len(negatives)//2): # Divide list in half to stop some of the crazy flickering
            line = negatives[x]
            if line[0]<200: # stops randomly detected specs from ruining the average
                count += 1
                nx1 = nx1 + line[0]
                ny1 = ny1 + line[1]
                nx2 = nx2 + line[2]
                ny2 = ny2 + line[3]
        try:
            px1 = px1 // len(positives)
            py1 = py1 // len(positives)
            px2 = px2 // len(positives)
            py2 = py2 // len(positives)
            nx1 = nx1 // count
            ny1 = ny1 // count
            nx2 = nx2 // count
            ny2 = ny2 // count
            # Midline calculations are weighted to account a little for the perspective transform
            midx1 = (((2 * px2) + (3 * nx1))// 5)
            midy1 = (((2 * py1) + (3 * ny2))// 5)
            midx2 = (((4 * px1) + (3 * nx2))// 7)
            midy2 = (((2 * py2) + (3 * ny1))// 5)

            # Reverse perspective transformation for line endpoints and display lines
            if display == 0:
                pt1 = np.array([[[px1, py1]]], dtype='float32')
                pt2 = np.array([[[px2, py2]]], dtype='float32')
                reverse_pt1 = cv2.perspectiveTransform(pt1, np.linalg.inv(matrix))[0][0]
                reverse_pt2 = cv2.perspectiveTransform(pt2, np.linalg.inv(matrix))[0][0]
                cv2.line(frame, (int(reverse_pt1[0]), int(reverse_pt1[1])), (int(reverse_pt2[0]), int(reverse_pt2[1])), (0, 0, 0), 9)

                nt1 = np.array([[[nx1, ny1]]], dtype='float32')
                nt2 = np.array([[[nx2, ny2]]], dtype='float32')
                reverse_nt1 = cv2.perspectiveTransform(nt1, np.linalg.inv(matrix))[0][0]
                reverse_nt2 = cv2.perspectiveTransform(nt2, np.linalg.inv(matrix))[0][0]
                cv2.line(frame, (int(reverse_nt1[0]), int(reverse_nt1[1])), (int(reverse_nt2[0]), int(reverse_nt2[1])), (0, 0, 0), 9)

                mt1 = np.array([[[midx1, midy1]]], dtype='float32')
                mt2 = np.array([[[midx2, midy2]]], dtype='float32')
                reverse_mt1 = cv2.perspectiveTransform(mt1, np.linalg.inv(matrix))[0][0]
                reverse_mt2 = cv2.perspectiveTransform(mt2, np.linalg.inv(matrix))[0][0]
                cv2.line(frame, (int(reverse_mt1[0]), int(reverse_mt1[1])), (int(reverse_mt2[0]), int(reverse_mt2[1])), (0, 0, 0), 9)
        except:
            pass

    # Template matching for left arrow
    img_gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp2_gray = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)
    match2 = cv2.matchTemplate(image=img_gray2, templ=temp2_gray, method=cv2.TM_CCOEFF_NORMED)
    (y_points, x_points) = np.where(match2 >= thresh2)
    boxes2 = list()
    for (x, y) in zip(x_points, y_points):
        boxes2.append((x, y, x + W, y + H))
    boxes2 = non_max_suppression(np.array(boxes2))

    # Check if left arrow template is found
    if len(boxes2) > 0:
        template_counter += 1
        if template_counter2 >= template_threshold2:
            display += 1
            compass = "left"

    # Template matching for right arrow
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    match = cv2.matchTemplate(image=img_gray, templ=temp_gray, method=cv2.TM_CCOEFF_NORMED)
    (y_points, x_points) = np.where(match >= thresh)
    boxes = list()
    for (x, y) in zip(x_points, y_points):
        boxes.append((x, y, x + W, y + H))
    boxes = non_max_suppression(np.array(boxes))

    # Check if right arrow template is found
    if len(boxes) > 0:
        template_counter += 1
        if template_counter >= template_threshold:
            template_counter = -30
            display += 1
            compass = "right"
    if len(boxes2) > 0:
        template_counter2 += 1
        print("Left")
        if template_counter2 >= template_threshold2:
            display += 1
            compass = "left"

    # Check if px1 falls below threshold
    if px1 < px1_threshold:
        px1_below_threshold_count += 1
        if px1_below_threshold_count >= compass_switch_threshold:
            px1_below_threshold_count = 0
            display = 0
            compass = "forward"

    # Display compass value
    cv2.putText(frame, "Compass: " + compass, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

    # Show lines on original video and resizes to fit GUI
    frame = cv2.resize(frame, (450, 300))
    return frame


# Initialize template images and values
temp = cv2.imread('right_arrow.png')
W, H = temp.shape[:2]
temp2 = cv2.imread("../GUI with Image Overlay/left_arrow.png")
W2, H2 = temp.shape[:2]
thresh = 0.5
thresh2 = 0.62

# Initialize variables used in template matching and determining compass
display = 0
compass = "forward"
template_counter = 0
template_counter2 = 0
template_threshold = 5
template_threshold2 = 4
compass_switch_threshold = 400
px1_below_threshold_count = 0
px1_threshold = 250


#Works Cited:
# Perspective Transform: https://www.geeksforgeeks.org/perspective-transformation-python-opencv/
# lines 9-14 & 78-94
# Template Matching: https://www.geeksforgeeks.org/multi-template-matching-with-opencv/
# lines 99-134