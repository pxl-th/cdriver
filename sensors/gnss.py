from weakref import ref

from carla import Transform, Location
from carla.libcarla import GnssMeasurement

from sensors.recordable import Recordable


class GnssSensor(Recordable):
    def __init__(self, parent_actor, frequency: int = 5):
        super().__init__("gnss")

        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()

        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_bp.set_attribute("sensor_tick", str(1 / frequency))
        self.sensor = world.spawn_actor(
            gnss_bp, Transform(Location(x=1.0, z=2.8)), attach_to=self._parent,
        )
        self.sensor.listen(lambda event: GnssSensor._on_event(
            ref(self), event,
        ))

    @staticmethod
    def _on_event(weak_self, event: GnssMeasurement) -> None:
        self = weak_self()
        if not self:
            return

        self.lat = event.latitude
        self.lon = event.longitude
        if self.recording:
            self.buffers["timestamp"].append(event.timestamp)
            self.buffers["gnss"].append([
                event.latitude, event.longitude, event.altitude,
            ])
