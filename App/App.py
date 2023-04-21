import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Game import Game
from Vector import Vector
import screen_functions


class App:
    def __init__(self, width, height, selected_camera=0):
        self.width = width
        self.height = height
        self.game = Game(width, height)
        self.camera_bin_image = np.zeros((self.height, self.width, 1), np.uint8)
        self.__drawing = False

        #
        self.camera = cv2.VideoCapture(selected_camera)
        cv2.namedWindow("Game")
        cv2.namedWindow("bin")
        cv2.namedWindow("Warped")
        cv2.namedWindow("drawn")
        cv2.namedWindow("handwritings")
        cv2.namedWindow("acc_handwritings")
        # cv2.namedWindow("acc_handwritings")

        cv2.setMouseCallback("Game", self.draw)

    def run(self):
        """Run the app loop"""
        # screen corners detection with ARUCO markers
        marker_dict = {}
        break_flag = False
        while True:
            f = self.__wait_key()
            if f:
                break_flag = True
            marker_image = screen_functions.generate_aruco((self.width, self.height), marker_ids=[1, 2, 3, 4])
            ret, frame = self.camera.read()
            cv2.imshow("Game", marker_image)
            image = cv2.GaussianBlur(frame, (0, 0), 3)
            image = cv2.addWeighted(frame, 1.5, image, -0.5, 0)
            cv2.imshow("Image", image)

            marker_dict = screen_functions.update_markers_dict(image, cv2.aruco.DICT_4X4_50, [1, 2, 3, 4], marker_dict)
            # debug output to control detected markers. After you see all 4 of them, press b
            print(marker_dict.keys())

            if len(marker_dict.keys()) == 4 and break_flag:
                corners = screen_functions.generate_screen_corners(marker_dict, [1, 2, 3, 4])
                print("corners: ", corners)
                warp_matrix, maxWidth, maxHeight = screen_functions.get_warp_matrix(corners)
                break


        handwritings = self.camera_bin_image.copy()
        is_inverted = True
        kernel = np.ones((3, 3), 'uint8')
        bkernel = np.ones((7, 7), 'uint8')
        accumulated_handwritings = np.zeros((self.height, self.width), np.float64)
        thresholded_handwritings = self.camera_bin_image.copy()

        # main loop
        while True:
            self.__wait_key()
            # receiving image with drawn object from the game
            drawn_objects = np.squeeze(self.game.get_drawn_objects())
            cv2.imshow("drawn", drawn_objects)

            # here we update the game module
            image_to_display = self.game.update(thresholded_handwritings)
            image_to_display = 255 - image_to_display
            cv2.imshow("Game", image_to_display)

            # reading, warping and binarizing
            ret, frame = self.camera.read()
            warped_image = cv2.warpPerspective(frame, warp_matrix, (maxWidth, maxHeight), flags=cv2.INTER_NEAREST)
            cv2.imshow("Warped", warped_image)
            bin_img = screen_functions.binarize_image(warped_image)

            scaled_bin_img = cv2.erode(cv2.resize(bin_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST),
                                       kernel, iterations=1)
            cv2.imshow("bin", scaled_bin_img)

            # subtracting drawn objects from the camera image
            # dilation makes it too huge. We can remove small objects by their area
            handwritings = scaled_bin_img - cv2.dilate(drawn_objects, bkernel, iterations=2)

            # here we accumulate handwritings to binarize it one more time and get the handwritings
            accumulated_handwritings += handwritings
            accumulated_handwritings *= 0.9
            thresholded_handwritings= (accumulated_handwritings / accumulated_handwritings.max() * 255).astype('uint8')
            thresholded_handwritings = cv2.threshold(thresholded_handwritings, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            thresholded_handwritings = 255 - thresholded_handwritings


            cv2.imshow("handwritings", thresholded_handwritings)
            cv2.imshow("acc_handwritings", accumulated_handwritings)

    def draw(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__drawing = True
            cv2.circle(self.camera_bin_image, (x, y), 3, 255, -1)
        if event == cv2.EVENT_LBUTTONUP:
            self.__drawing = False
        if event == cv2.EVENT_MOUSEMOVE:
            if self.__drawing: cv2.circle(self.camera_bin_image, (x, y), 3, 255, -1)

    def __wait_key(self):
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit(0)

        if key == ord('r'):
            self.camera_bin_image = np.zeros((self.height, self.width, 1), np.uint8)

        if key == ord('w'):
            self.game.balls[0].add_force(Vector(0, -1), 1)
        if key == ord('d'):
            self.game.balls[0].add_force(Vector(1, 0), 1)
        if key == ord('a'):
            self.game.balls[0].add_force(Vector(-1, 0), 1)
        if key == ord('s'):
            self.game.balls[0].add_force(Vector(0, 1), 1)
        if key == ord('b'):
            return True