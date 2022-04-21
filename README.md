# A python package to lazy work with ArUco.

[![Documentation Status](https://readthedocs.org/projects/lazy-aruco/badge/?version=latest)](https://lazy-aruco.readthedocs.io/en/latest/?badge=latest)

```python
>>> from laruco.detect import Detection
>>> det = Detection(aruco_dict="DICT_5X5_100", calibration_filename='../calibrations/calibration_l.pckl')
>>> det.detect_angles(frame,marker_size=2.1)
```

## Installing lazy ArUco and Supported Versions

Lazy ArUco is available on PyPI: 

TO BE DATED
```console
$ python3 -m pip install lazy-aruco
```

## API Reference and User Guide available on [Read the Docs](https://lazy-aruco.readthedocs.io)
