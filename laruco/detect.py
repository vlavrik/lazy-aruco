import os
import cv2
import pickle
import numpy as np
import math
from .utils import estimate_area, rotation_matrix_to_euler_angles, is_rotation_matrix


class Detection():
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

    def __init__(self, aruco_dict, calibration_filename=None):
        """The main class for ArUco markers detection.

        Parameters
        ----------
        aruco_dict : str
            The set of ArUco markers, known as a dictionary. 
        calibration_filename : str, optional
            The full path to a file where the camera calibration is stored, by default None
        """
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
            (cameraMatrix, distCoeffs) = (None, None)

        return aruco_dict, aruco_params, cameraMatrix, distCoeffs

    def _detect(self, frame, marker_size):

        aruco_dict, aruco_params, camera_matrix, dist_coeffs = self._initialize_detector()
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(image=gray_frame, dictionary=aruco_dict, parameters=aruco_params,
                                                             cameraMatrix=camera_matrix, distCoeff=dist_coeffs)
        except:
            corners, ids, rejected = list(), list(), list()

        return corners, ids, rejected

    def _detect_marker_area(self, frame, marker_size):

        corners, ids, rejected = self._detect(frame, marker_size)
        list_cX, list_cY, list_areas = list(), list(), list()

        if corners:
            list_cX, list_cY, list_areas = estimate_area(corners)

        else:
            (cX, cY, area) = (0, 0, 0)
            list_cX.append(cX)
            list_cY.append(cY)
            list_areas.append(area)

        return list_cX, list_cY, list_areas

    def detect_images(self, path_to_frames, path_to_detections, marker_size, headless=False):
        """Performs the detection of ArUco markers.

        Parameters
        ----------
        path_to_frames : str
            Path where the set of frames are stored.
        marker_size : float
            The size of marker to be detected. A unit has to be consistent with a unit used for a calibration.
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

        filenames = os.listdir(path_to_frames)
        for filename in filenames:
            frame = cv2.imread(path_to_frames + filename)
            corners, ids, rejected = self._detect(frame, marker_size)
            if ids is not None:
                if not headless or path_to_detections is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    try:
                        cv2.imwrite(path_to_detections+filename, frame)
                    except:
                        pass
                    cv2.imshow('ArUco detection', frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    pass
                corners_list.append(corners)
                ids_list.append(ids)
                rejected_list.append(rejected)
            else:
                print('[INFO] Check file: ' + path_to_frames + filename)

        return corners_list, ids_list, rejected_list

    def detect_marker(self, frame, marker_size):
        """_summary_

        Parameters
        ----------
        frame : _type_
            _description_
        marker_size : _type_
            _description_

        Returns
        -------
        corners : tuple
            _description_
        ids : ndarray
            _description_
        rejected : tuple
            _description_
        """

        try:
            corners, ids, rejected = self._detect(frame, marker_size)
        except:
            corners, ids, rejected = None, None, None

        return corners, ids, rejected

    def detect_area(self, frame, marker_size):
        """_summary_

        Parameters
        ----------
        frame : _type_
            _description_
        marker_size : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        list_cX, list_cY, list_areas = list(), list(), list()
        corners, ids, rejected = self._detect(frame, marker_size)
        try:
            cv2.aruco.drawDetectedMarkers(frame, corners)
        except:
            pass
        for i in range(len(corners)):
            list_cX, list_cY, list_areas = list(), list(), list()
            cX, cY, area = self._detect_marker_area(frame, marker_size)
            list_cX.append(cX)
            list_cY.append(cY)
            list_areas.append(area)

        return list_cX, list_cY, list_areas, frame

    def _detect_pose(self, frame, marker_size):
        aruco_dict, aruco_params, camera_matrix, dist_coeffs = self._initialize_detector()
        corners, ids, rejected = self._detect(frame, marker_size)
        cv2.aruco.drawDetectedMarkers(frame, corners)
        rvec_all, tvec_all, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs)
        try:
            rvec = rvec_all[0][0]
            tvec = tvec_all[0][0]
            cv2.aruco.drawAxis(frame, camera_matrix,
                               dist_coeffs, rvec, tvec, 5)

        except:
            pass

        return rvec_all, tvec_all, frame

    def detect_angles(self, frame, marker_size):
        """_summary_

        Parameters
        ----------
        frame : _type_
            _description_
        marker_size : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        rvec_all, tvec_all, frame = self._detect_pose(frame, marker_size)
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
        except:
            x, y, direction = None, None, None
        return x, y, direction, frame

    def detect_distance(self, frame, marker_size):
        """_summary_

        Parameters
        ----------
        frame : _type_
            _description_
        marker_size : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        rvec_all, tvec_all, _ = self._detect_pose(frame, marker_size)

        return rvec_all, tvec_all, frame
