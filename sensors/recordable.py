from collections import defaultdict
from os import makedirs
from os.path import join

from numpy import savetxt, array


class Recordable:
    def __init__(self, name: str):
        self.name = name
        self.recording = False
        self.directory = None
        self.buffers = defaultdict(lambda: list())

    def save(self) -> None:
        for name in self.buffers.keys():
            buff = self.buffers[name]
            if buff and self.directory is not None:
                savetxt(
                    join(self.directory, name), array(buff, copy=False),
                    "%.08f",
                )
                self.buffers[name].clear()

    def toggle_recording(self, recording_dir: str) -> None:
        self.recording = not self.recording
        if self.recording:
            self.directory = join(recording_dir, self.name)
            makedirs(self.directory, exist_ok=True)
            if self.name == "cam":
                makedirs(join(self.directory, "frames"), exist_ok=True)
        else:
            self.save()
            self.directory = None
