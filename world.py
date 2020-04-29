from sys import exit
from re import compile, match
from random import choice

from carla import WeatherParameters, World, WorldSettings
from sensors import CameraManager, GnssSensor, IMUSensor


def find_weather_presets():
    rgx = compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(WeatherParameters) if match('[A-Z].+', x)]
    return [(getattr(WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class WorldHolder:
    def __init__(self, carla_world, hud, args, fps):
        self.world: World = carla_world
        self.fps = fps

        settings: WorldSettings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / fps
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        print(f"Applied settings {settings}")

        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print(
                f'RuntimeError: {error}\n'
                'The server could not send the OpenDRIVE (.xodr) file:\n'
                'Make sure it exists, has the same name of your town, '
                'and is correct.',
            )
            exit(1)

        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._gamma = args.gamma

        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None

        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Get car blueprint.
        blueprint = self.world.get_blueprint_library().find(
            "vehicle.tesla.model3",
        )
        blueprint.set_attribute('role_name', self.actor_role_name)
        blueprint.set_attribute('color', "180,44,44")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                exit(1)

            spawn_point = choice(spawn_points)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        # Set up camera.
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.set_sensor(notify=False)
        self.hud.notification(get_actor_display_name(self.player))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.world.tick()
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor, self.gnss_sensor.sensor,
            self.imu_sensor.sensor, self.player,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()
