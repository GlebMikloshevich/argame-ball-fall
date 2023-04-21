import cv2
import numpy as np
import matplotlib.pyplot as plt


# generate image with aruco markers on the edges
def generate_aruco(screen_shape: tuple[int, int], size: int = 100,
                   padding: int = 1, aruco_dict_name=cv2.aruco.DICT_4X4_50, marker_ids=[1, 2, 4, 8]) -> np.ndarray:
    width, height = screen_shape

    if len(marker_ids) != 4:
        raise AttributeError("marker_ids length should be exact 4")

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    tags = [np.zeros((size, size, 1), dtype="uint8") for i in range(len(marker_ids))]
    for i in range(len(marker_ids)):
        aruco_dict.generateImageMarker(marker_ids[i], size, tags[i], 1)

    screen = np.ones((height, width, 3), dtype='uint8')
    screen *= 255

    # 0
    screen[padding:size + padding, padding:size + padding] = tags[0]
    # 1
    screen[padding:size + padding, width - size - padding:width - padding] = tags[1]
    # 2
    screen[height - size - padding:height - padding, width - size - padding:width - padding] = tags[2]
    # 3
    screen[height - size - padding:height - padding, padding:size + padding] = tags[3]
    return screen


# detect aruco markers on the screens
def detect_screen_corners(screen: np.ndarray, aruco_dict_name=cv2.aruco.DICT_4X4_50, marker_ids=[1, 2, 4, 8]) -> list:
    screen_corners = []

    # create detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    marker_dict = {}
    while True:
        # detect markers

        marker_corners, markerIds, rejected_candidates = detector.detectMarkers(screen)
        if markerIds is None:
            print(marker_corners)
            continue
        print(f"len markerIds: {len(markerIds)}")

        for idx, corners in zip(markerIds, marker_corners):
            # print(str(idx[0]))
            marker_dict[idx[0]] = corners
        if len(marker_dict.keys()) == 4:
            break

    for i in range(len(marker_ids)):
        if marker_ids[i] in marker_dict:
            print("marker: ", marker_dict[marker_ids[i]])
            # print(marker_dict[marker_ids[i]][0][i])
            screen_corners.append(marker_dict[marker_ids[i]][0][i])
        else:
            print(f"no marker: {marker_ids[i]}")

    return screen_corners


def update_markers_dict(screen: np.ndarray, aruco_dict_name=cv2.aruco.DICT_4X4_50, marker_ids=[1, 2, 4, 8], marker_dict={}) -> dict:
    screen_corners = []

    # create detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # detect markers
    marker_corners, markerIds, rejected_candidates = detector.detectMarkers(screen)
    if markerIds is None:

        return marker_dict
    print("at least smth")
    for idx, corners in zip(markerIds, marker_corners):
        if idx not in marker_ids:
            continue
        marker_dict[idx[0]] = corners
    # print(marker_dict.keys())
    return marker_dict


def generate_screen_corners(marker_dict: dict, marker_ids=[1, 2, 4, 8]):
    screen_corners = []
    for i in range(len(marker_ids)):
        if marker_ids[i] in marker_dict:
            print("marker: ", marker_dict[marker_ids[i]])
            # print(marker_dict[marker_ids[i]][0][i])
            print(marker_dict)
            screen_corners.append(marker_dict[marker_ids[i]][0][i])
        else:
            print(f"no marker: {marker_ids[i]}")
    return screen_corners


# create warp matrix based on 4 points. [top_left, top_right, bottom_right, bottom_left]
def get_warp_matrix(points):
    global maxWidth, maxHeight
    maxWidth = 500
    maxHeight = 500
    width_AB = np.sqrt(((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    width_CD = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    maxWidth = max(int(width_AB), int(width_CD))

    height_AD = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    height_BC = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    maxHeight = max(int(height_AD), int(height_BC))

    input_pts = np.float32([points[0], points[3], points[2], points[1]])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    return cv2.getPerspectiveTransform(input_pts, output_pts), maxWidth, maxHeight


def binarize_image(img, blur=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = img

    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)

    # ret, mask = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 2)
    th = (255-th)
    # return remove_small_objects(th, 20)
    return th


def binarize_image2(img, blur=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = img

    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)

    # ret, mask = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    th = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return th


# remove objects from binary image with area less than area_th.
def remove_small_objects(bin_img_original, area_th=500):
    bin_img = bin_img_original.copy()
    cnts = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_th:
            cv2.drawContours(bin_img, [c], -1, (0, 0, 0), -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    return close


def setup():
    pass


if __name__ == "__main__":
    s = generate_aruco((1920, 1080))

    corners = detect_screen_corners(s, cv2.aruco.DICT_4X4_50, [1, 2, 4, 8])

    plt.subplot(131)
    plt.imshow(s)

    print(corners)

    plt.subplot(132)
    for corner in corners:
        plt.plot(int(corner[0]), int(corner[1]), marker='o', color="red")
    plt.imshow(s)

    warp_matrix = get_warp_matrix(corners)
    warped_image = cv2.warpPerspective(s, warp_matrix, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    plt.subplot(133)
    plt.imshow(warped_image)
    plt.show()

    print(warp_matrix)
