# A python package to lazy work with ArUco.

[![Documentation Status](https://readthedocs.org/projects/lazy-aruco/badge/?version=latest)](https://lazy-aruco.readthedocs.io/en/latest/?badge=latest)

```python
>>> from flespi_gateway.gateway import Device
>>> dv = Device(device_number=device_number, flespi_token=flespi_token)
>>> telemetry = dv.get_telemetry()
>>> print(telemetry)
{'result': [{'id': xxxxxx,
   'telemetry': {'battery.current': {'ts': 1609521935, 'value': 0},
    'battery.voltage': {'ts': 1609521935, 'value': 4.049},
    'can.absolute.load': {'ts': 1609327396, 'value': 23}]
}
```

## Installing lazy ArUco and Supported Versions

Lazy ArUco is available on PyPI:

```console
$ python3 -m pip install lazy-aruco
```

## API Reference and User Guide available on [Read the Docs](https://lazy-aruco.readthedocs.io)
