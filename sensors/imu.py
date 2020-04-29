from math import degrees, sqrt
from weakref import ref

from carla import Transform
from carla.libcarla import IMUMeasurement

from sensors.recordable import Recordable


class IMUSensor(Recordable):
    def __init__(self, parent_actor, frequency: int = 50):
        super().__init__("imu")

        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()

        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_bp.set_attribute("sensor_tick", str(1 / frequency))
        self.sensor = world.spawn_actor(
            imu_bp, Transform(), attach_to=self._parent,
        )
        self.sensor.listen(lambda sensor_data: IMUSensor._on_event(
            ref(self), sensor_data,
        ))

    @staticmethod
    def _on_event(weak_self, sensor_data: IMUMeasurement):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        )
        self.gyroscope = (
            max(limits[0], min(limits[1], degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], degrees(sensor_data.gyroscope.z))),
        )
        self.compass = degrees(sensor_data.compass)

        if self.recording:
            self.buffers["timestamp"].append(sensor_data.timestamp)
            self.buffers["accelerometer"].append(self.accelerometer)
            self.buffers["gyro"].append(self.gyroscope)
            self.buffers["compass"].append(self.compass)

            velocity = self._parent.get_velocity()
            speed = sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            self.buffers["steer"].append(self._parent.get_control().steer)
            self.buffers["speed"].append(speed)  # in m/s
