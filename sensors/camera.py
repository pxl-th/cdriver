from os.path import join
from weakref import ref

from skimage.io import imsave
from numpy import frombuffer, reshape
from pygame.surfarray import make_surface

from carla import Transform, Location, AttachmentType, Rotation, ColorConverter
from carla.libcarla import Image, Vehicle

from sensors.recordable import Recordable


class CameraManager(Recordable):
    def __init__(
        self, parent_actor, hud, base_path: str = None,
        frequency: int = 20,
    ):
        super().__init__("cam")

        self._parent: Vehicle = parent_actor
        self._camera_transforms = (
            Transform(Location(x=0.4, z=1.3), Rotation(pitch=0.0)),
            AttachmentType.Rigid,
        )

        self.sensor = None
        self.surface = None
        self.hud = hud

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        self.sensor_blueprint = bp_library.find("sensor.camera.rgb")
        self.sensor_blueprint.set_attribute('image_size_x', str(hud.dim[0]))
        self.sensor_blueprint.set_attribute('image_size_y', str(hud.dim[1]))
        self.sensor_blueprint.set_attribute("sensor_tick", str(1 / frequency))

    def set_sensor(self, notify=True, force_respawn=False):
        if self.sensor is not None:
            self.sensor.destroy()
            self.surface = None
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensor_blueprint,
            self._camera_transforms[0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[1],
        )

        self.sensor.listen(lambda image: CameraManager._on_event(
            ref(self), image,
        ))
        if notify:
            self.hud.notification("Camera RGB")

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _on_event(weak_self, image: Image):
        self = weak_self()
        if not self:
            return

        image.convert(ColorConverter.Raw)  # BGRA
        image_raw = frombuffer(image.raw_data, dtype="uint8")
        image_raw = reshape(image_raw, (image.height, image.width, 4))
        image_raw = image_raw[:, :, :3]
        image_raw = image_raw[:, :, ::-1]  # RGB
        self.surface = make_surface(image_raw.swapaxes(0, 1))

        if self.recording:
            imsave(
                join(self.directory, "frames", f"{image.frame}.jpg"),
                image_raw,
            )
            self.buffers["timestamp"].append(image.timestamp)
