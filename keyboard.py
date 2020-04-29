from datetime import datetime
from os.path import join
from carla import VehicleLightState, VehicleControl

import pygame
from pygame.key import get_mods, get_pressed
from pygame.locals import (
    KMOD_CTRL, KMOD_SHIFT, K_0, K_9, K_BACKQUOTE, K_DOWN,
    K_ESCAPE, K_F1, K_LEFT, K_RIGHT, K_SLASH, K_SPACE, K_UP,
    K_a, K_c, K_d, K_h, K_n, K_p, K_q, K_r, K_s, K_w,
)


class KeyboardControl:
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot

        self._control = VehicleControl()
        self._lights = VehicleLightState.NONE
        world.player.set_autopilot(self._autopilot_enabled)
        world.player.set_light_state(self._lights)

        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        current_lights = self._lights

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_c and get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (get_mods() & KMOD_CTRL):
                    # When recording share same folder between sensors
                    dataset_path = r"C:\Users\tonys\projects\carla-dataset"
                    current_dir = (
                        f"recording-{datetime.now()}"
                        .replace(":", "-").replace(".", "-").replace(" ", "-")
                    )
                    base_dir = join(dataset_path, current_dir)

                    world.camera_manager.toggle_recording(base_dir)
                    world.gnss_sensor.toggle_recording(base_dir)
                    world.imu_sensor.toggle_recording(base_dir)

                    recording = world.camera_manager.recording
                    world.hud.notification(
                        f'Recording {"On" if recording else "Off"}',
                    )
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_p and not get_mods() & KMOD_CTRL:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        f"Autopilot {'On' if self._autopilot_enabled else 'Off'}"
                    )

        if not self._autopilot_enabled:
            self._parse_vehicle_keys(get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            # Set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= VehicleLightState.Brake
            else:  # Remove the Brake flag
                current_lights &= (
                    VehicleLightState.All ^ VehicleLightState.Brake
                )
            if self._control.reverse:
                current_lights |= VehicleLightState.Reverse
            else:  # Remove the Reverse flag
                current_lights &= (
                    VehicleLightState.All
                    ^ VehicleLightState.Reverse
                )
            # Change the light state only if necessary
            if current_lights != self._lights:
                self._lights = current_lights
                world.player.set_light_state(VehicleLightState(self._lights))

            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key) -> bool:
        return key == K_ESCAPE
