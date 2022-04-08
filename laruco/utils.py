import cv2
import pickle
import math
import numpy as np


def draw_detected_markers(frame, corners):
    """Creates a frame with detected markers.

    Parameters
    ----------
    frame : ndarray
        The frame with ArUco markers
    corners : tuple
        Positions of marker corners on input image.

    Returns
    -------
    frame : ndarray
        The frame with detected markers.
    """
    cv2.aruco.drawDetectedMarkers(frame, corners)
    return frame


def is_rotation_matrix(R):
    """Checks if a matrix is a valid rotation matrix.

    Parameters
    ----------
    R : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R):
    """Calculates rotation matrix to euler angles. The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).

    Parameters
    ----------
    R : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    assert(is_rotation_matrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def estimate_area(corners):
    """Estimates area of a detected ArUco marker.

    Parameters
    ----------
    corner : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    list_cX, list_cY, list_areas = list(), list(), list()
    for i in range(len(corners)):
        corner = corners[i].reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corner
        width = abs(topRight[0] - topLeft[0])
        height = abs(topRight[1] - bottomRight[1])
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        list_cX.append(cX)
        list_cY.append(cY)
        list_areas.append(width*height)

    return (list_cX, list_cY, list_areas)


def record_video(path_to_save_video, capture_channel=0, resolution=(640, 480)):
    """Streams and saves video.

    Parameters
    ----------
    path_to_save_video : str
        _description_
    capture_channel : int, optional
        Video channel, by default 0
    resolution : tuple, optional
        Resolution of a camera, by default (640, 480)
    """
    cap = cv2.VideoCapture(capture_channel)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_to_save_video, fourcc, 30.0, resolution)
    while(True):
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('Video stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def slice_video(path_from, file_name, path_to):
    # videoFile = "../video/output.avi"
    videoFile = path_from + '{}.avi'.format(file_name)
    imagesFolder = path_to
    cap = cv2.VideoCapture(videoFile)
    # frameRate = cap.get(5)  # frame rate
    i = 0
    while(cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break

        if i % 300:  # (frameId < 100):
            filename = imagesFolder + "image_" + str(int(frameId)) + ".jpg"
            cv2.imwrite(filename, frame)
        i += 1
    cap.release()


def estimate_distance():
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters_create()

    frame = cv2.imread('./test/images/image_223.jpg')
    print(frame.shape)
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs) = pickle.load(f)
    f.close()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, dictionary, parameters=parameters)
    # markerSizeInCM = 4.3
    markerSizeInCM = 17
    print(corners, ids)
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerSizeInCM, cameraMatrix, distCoeffs)
    print(tvec)


#record_video('./videos/charuco/', 'new_4')
# slice_video(path_from='./test/videos/',
#              file_name='video', path_to='./test/images/')
# slice_video(path_from='./', path_to='./test/')
# estimate_distance()
