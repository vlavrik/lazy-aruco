import os
import cv2
import pickle
import numpy as np
import math
from .utils import estimate_area, rotation_matrix_to_euler_angles, is_rotation_matrix


class Detection():
    """ArUco markers detection.

    Parameters
    ----------
    aruco_dict : str
        One of the predefined ArUco dictionaries. Defaults to "DICT_5X5_100"
    calibration_filename : str
        A name of calibration file wothout extension.
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

    def __init__(self, aruco_dict="DICT_5X5_100", calibration_filename=None):
        self.aruco_dict = aruco_dict
        self.calibration_filename = calibration_filename

    def _initialize_detector(self):
        aruco_dict = cv2.aruco.Dictionary_get(
            getattr(cv2.aruco, self.aruco_dict))
        aruco_params = cv2.aruco.DetectorParameters_create()
        try:
            f = open(self.calibration_filename, 'rb')
            (cameraMatrix, distCoeffs) = pickle.load(f)
            f.close()
        except:
            print('[WARNING] No calibration file found.')
            (cameraMatrix, distCoeffs) = (None, None)

        return aruco_dict, aruco_params, cameraMatrix, distCoeffs

    def _detect(self, frame):

        aruco_dict, aruco_params, camera_matrix, dist_coeffs = self._initialize_detector()
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(image=gray_frame, dictionary=aruco_dict, parameters=aruco_params,
                                                             cameraMatrix=camera_matrix, distCoeff=dist_coeffs)
        except:
            corners, ids, rejected = list(), list(), list()

        return corners, ids, rejected

    def _detect_marker_area(self, frame):

        corners, ids, rejected = self._detect(frame)
        list_cX, list_cY, list_areas = list(), list(), list()

        if corners:
            list_cX, list_cY, list_areas = estimate_area(corners)

        else:
            (cX, cY, area) = (0, 0, 0)
            list_cX.append(cX)
            list_cY.append(cY)
            list_areas.append(area)

        return list_cX, list_cY, list_areas

    def detect_images(self, path_to_raw_frames, path_to_detected_frames, headless=False):
        """Performs the detection of ArUco markers.

        Parameters
        ----------
        path_to_raw_frames : str
            Path where the set of frames are stored.
        path_to_detected_frames : str
            Pathe where detected frames will be saved.
        headless : bool, optional
            If True the frame with detected markers is not displayed, by default False

        Returns
        -------
        tuple
            Every element of tuple is a list with detected markers' corners, ids and rejected points.
        """
        corners_list = list()
        ids_list = list()
        rejected_list = list()

        filenames = os.listdir(path_to_raw_frames)
        for filename in filenames:
            frame = cv2.imread(path_to_raw_frames + filename)
            corners, ids, rejected = self._detect(frame)
            if ids is not None:
                if not headless:
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.imwrite(path_to_detected_frames + filename, frame)
                    cv2.imshow('ArUco detection', frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:

                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.imwrite(path_to_detected_frames + filename, frame)
                corners_list.append(corners)
                ids_list.append(ids)
                rejected_list.append(rejected)
            else:
                print('[INFO] Check file: ' + path_to_frames + filename)

        return corners_list, ids_list, rejected_list

    def detect_marker(self, frame):
        """Detects markers on the input frame.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame where AruCo markers must be detected.

        Returns
        -------
        corners : list
            List of arrays. Every array is a vector of detected marker corners. For each marker, its four corners are provided.
        ids : list
            List of arrays. Every array is a vector of identifiers of the detected markers.
        rejected : list
            List of arrays. Every array contains the points of those squares whose inner code has not a correct codification.
        """
        if isinstance(frame, np.ndarray):

            try:
                corners, ids, rejected = self._detect(frame)
            except:
                corners, ids, rejected = None, None, None
        else:
            print('[WARNING] Check the input frame.')
            corners, ids, rejected = None, None, None

        return corners, ids, rejected

    def detect_area(self, frame, draw_annotations=False):
        """Calculates area of detected markers.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame where AruCo markers must be detected.
        draw_annotations : bool, optional
            Draws id and area of a detected marker in the frame, by default False

        Returns
        -------
        list_cX : list
            List of X coordinate of a detected marker.
        list_cY : list
            List of Y coordinate of a detected marker.
        list_areas : list
            List of areas calculated for a detected marker in pixels.
        frame : numpy.ndarray
            Frame with detections drawn. If parameter draw_annotations is True, the id and area of detected markers are also drawn.
        """

        list_cX, list_cY, list_areas = self._detect_marker_area(frame)
        corners, ids, rejected = self._detect(frame)
        try:
            cv2.aruco.drawDetectedMarkers(frame, corners)
        except:
            pass

        h = frame.shape[0]
        w = frame.shape[1]
        if draw_annotations:
            try:
                for i in range(len(list_areas)):
                    print(ids[i])
                    cv2.putText(frame, 'ID: ' + str(int(ids[i])) + ' Area: ' + str(int(
                        list_areas[i])), (5, h - 10 - i*25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
            except:
                pass

        return list_cX, list_cY, list_areas, frame

    def _detect_pose(self, frame, marker_size):
        aruco_dict, aruco_params, camera_matrix, dist_coeffs = self._initialize_detector()
        corners, ids, rejected = self._detect(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners)
        rvec_all, tvec_all, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs)
        try:
            # to do draw for all markers
            rvec = rvec_all[0][0]
            tvec = tvec_all[0][0]
            cv2.aruco.drawAxis(frame, camera_matrix,
                               dist_coeffs, rvec, tvec, marker_size)
        except:
            pass

        return rvec_all, tvec_all, frame

    def detect_distance(self, frame, marker_size, draw_annotations=False):
        """Estimates a distance to detected ArUco marker.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame where AruCo markers must be detected.
        marker_size : float
            Marker side length.
        draw_annotations : bool, optional
            Draws id and area of a detected marker in the frame, by default False.

        Returns
        -------
        rvec_all : numpy.ndarray
            Array of output rotation vectors.
        tvec_all : numpy.ndarray
            Array of output translation vectors.
        frame : numpy.ndarray
            Frame with detections drawn. If parameter draw_annotations is True, the id and area of detected markers are also drawn.
        """
        rvec_all, tvec_all, frame = self._detect_pose(frame, marker_size)
        h = frame.shape[0]
        w = frame.shape[1]
        if draw_annotations:
            try:
                for i in range(len(tvec_all)):
                    cv2.putText(frame, 'Dist.: ' + str(int(tvec_all[i][0][-1])), (
                        5, h - 10 - i*25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
            except:
                pass

        return rvec_all, tvec_all, frame

    def detect_angles(self, frame, marker_size, draw_annotations=False):
        """Estimates x, y and direction of a detected ArUco marker.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame where AruCo markers must be detected.
        marker_size : float
            Marker side length.
        draw_annotations : bool, optional
            Draws id and area of a detected marker in the frame, by default False.

        Returns
        -------
        x : float
            _description_
        y : float
            _description_
        direction : float
            _description_
        frame : numpy.ndarray
            Frame with detections drawn. If parameter draw_annotations is True, the x, y and direction of a detected marker is also drawn.
        """

        rvec_all, tvec_all, frame = self._detect_pose(frame, marker_size)
        h = frame.shape[0]
        w = frame.shape[1]

        try:
            rvec = rvec_all[0][0]
            tvec = tvec_all[0][0]
            rvec_flipped = rvec * -1
            tvec_flipped = tvec * -1

            rotation_matrix, jacobian = cv2.Rodrigues(rvec_flipped)
            realworld_tvec = np.dot(rotation_matrix, tvec_flipped)

            p, y, r = rotation_matrix_to_euler_angles(rotation_matrix)

            x, y, direction = realworld_tvec[0], realworld_tvec[1], math.degrees(
                y)

            # to do consider multiple poses
            if draw_annotations:

                cv2.putText(frame, 'x:' + str(round(x, 1)) + ' y:' + str(round(y, 1)) + ' dir.:' + str(round(direction, 1)), (
                    5, h - 10 - 0*25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            x, y, direction = None, None, None
        return x, y, direction, frame
