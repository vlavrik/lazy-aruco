"""Generating and saving ArUco markers, chessboard, gridboard.
"""

import cv2


class Generate():
    """A class for generating and saving ArUco markers, chessboards and gridboards.

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

    def __init__(self, aruco_dict="DICT_5X5_100"):
        self.aruco_dict = aruco_dict

    def generate_aruco_marker(self, marker_id, size, headless=False):
        """Generates a single ArUco marker with a specified marker_id and a size.

        Parameters
        ----------
        marker_id : int
            Id of a marker from an ArUco dictionary.
        size : int
            Size of a marker from an ArUco dictionary in pixels.
        headless : bool
            If True the marker is not displayed. Defaults to False.
        """
        aruco_dict = cv2.aruco.Dictionary_get(
            getattr(cv2.aruco, self.aruco_dict))
        img = cv2.aruco.drawMarker(aruco_dict, marker_id, size)
        if not headless:
            cv2.imshow('ArUco tag', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cv2.imwrite('marker_id_' + '{}.jpeg'.format(marker_id), img)

    def generate_aruco_gridboard(self, rows=7, cols=5,
                                 marker_length=0.04, marker_separation=0.01,
                                 first_marker_id=10, headless=False):
        """Generates an ArUco gridboard to be used for a camera calibration.

        Parameters
        ----------
        rows : int, optional
            Number of markers in Y direction, by default 7
        cols : int, optional
            Number of markers in X direction, by default 5
        marker_length : float, optional
            Marker side length (normally in meters), by default 0.04
        marker_separation : float, optional
            Separation between two markers (same unit as markerLength), by default 0.01
        first_marker_id : int, optional
            Id of first marker in dictionary to use on board, by default 10
        headless : bool, optional
            If True the board is not displayed, by default False
        """

        aruco_dict = cv2.aruco.Dictionary_get(
            getattr(cv2.aruco, self.aruco_dict))
        gridboard = cv2.aruco.GridBoard_create(
            markersX=cols,
            markersY=rows,
            markerLength=marker_length,
            markerSeparation=marker_separation,
            dictionary=aruco_dict,
            firstMarker=first_marker_id)
        img = gridboard.draw(outSize=(988, 1400))
        if not headless:
            cv2.imshow('ArUco gridboard', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cv2.imwrite("gridboard_{}x{}_starts_with_id_{}.jpeg"
                    .format(rows, cols, first_marker_id), img)

    def generate_charuco_gridboard(self, rows=4, cols=3,
                                   square_length=0.04, marker_length=0.02, headless=False):
        """Generates a chArUco gridboard to be used for a camera calibration.

        Parameters
        ----------
        rows : int, optional
            Number of chessboard squares in Y direction, by default 4
        cols : int, optional
            Number of chessboard squares in X direction, by default 3
        square_length : float, optional
            Chessboard square side length (normally in meters), by default 0.04
        marker_length : float, optional
            Marker side length (same unit than squareLength), by default 0.02
        headless : bool, optional
            If True the board is not displayed, by default False
        """
        gridboard = cv2.aruco.CharucoBoard_create(
            squaresX=cols,
            squaresY=rows,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=cv2.aruco.Dictionary_get(self.aruco_dict))
        img = gridboard.draw(outSize=(988, 1400))
        if not headless:
            cv2.imshow('ChArUco gridboard', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite("charuco_gridboard_{}x{}.jpeg".format(rows, cols), img)
