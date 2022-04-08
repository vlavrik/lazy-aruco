"""Calibration of a camera.
"""

import os
import pickle
import cv2
from tqdm import tqdm


class Calibrate():
    """Calibration

    Parameters
    ----------
    aruco_dict : str
        One of the predefined ArUco dictionaries. Defaults to "DICT_5X5_100"
    """
    aruco_dict_list = [
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_4X4_1000",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_6X6_50",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_50",
        "DICT_7X7_100",
        "DICT_7X7_250",
        "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5",
        "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11"
    ]

    def __init__(self, aruco_dict, path_to_images, calibration_filename):
        self.aruco_dict = aruco_dict
        self.path_to_images = path_to_images
        self.calibration_filename = calibration_filename

    def _create_aruco_dict(self):

        if self.aruco_dict in self.aruco_dict_list:

            return cv2.aruco.Dictionary_get(getattr(cv2.aruco, self.aruco_dict))
        return None

    def calibrate_charuco(self, rows=4, cols=3, square_length=0.087, marker_length=0.043, headless=False):
        """Perform a calibration of a camera with a usage of chArUco board.

        Parameters
        ----------
        rows : int, optional
            Number of chessboard squares in Y direction, by default 4
        cols : int, optional
            Number of chessboard squares in X direction, by default 3
        square_length : float, optional
            Chessboard square side length (normally in meters), by default 0.087
        marker_length : float, optional
            Marker side length (same unit than squareLength), by default 0.043
        headless : bool, optional
            If True the board is not displayed, by default False
        """

        aruco_dictionary = self._create_aruco_dict()
        gridboard = cv2.aruco.CharucoBoard_create(squaresX=cols,
                                                  squaresY=rows,
                                                  squareLength=square_length,
                                                  markerLength=marker_length,
                                                  dictionary=aruco_dictionary)

        corners_all = list()
        ids_all = list()

        images = os.listdir(self.path_to_images)
        for image in tqdm(images):
            frame = cv2.imread(self.path_to_images + image)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(
                image=gray, dictionary=aruco_dictionary)
            image_size = gray.shape[::-1]
            if ids is not None:

                frame = cv2.aruco.drawDetectedMarkers(
                    image=frame, corners=corners)
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=gridboard)

                if response >= 6:
                    frame = cv2.aruco.drawDetectedCornersCharuco(image=frame,
                                                                 charucoCorners=charuco_corners,
                                                                 charucoIds=charuco_ids)
                    corners_all.append(charuco_corners)
                    ids_all.append(charuco_ids)

                    if not headless:
                        cv2.imshow('Charuco board', frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print(
                        '[INFO] Not able to detect a full charuco board in image: {}'.format(image))

        if len(ids_all) != 0:
            calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=corners_all,
                charucoIds=ids_all,
                board=gridboard,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None)
            f = open(self.calibration_filename + '.pckl', 'wb')
            pickle.dump((cameraMatrix, distCoeffs), f)
            f.close()
        else:
            print('[WARNING] No ids detected in images.')
