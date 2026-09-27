from collections import deque
from threading import Condition
from time import monotonic


class FrameBuffer:
    def __init__(self, capacity=16):
        self.frames = deque(maxlen=capacity)
        self.condition = Condition()

    def publish(self, frame, value):
        with self.condition:
            self.frames.append((frame, value))
            self.condition.notify_all()

    def wait(self, frame, timeout):
        deadline = monotonic() + timeout
        with self.condition:
            while True:
                while self.frames:
                    current, value = self.frames.popleft()
                    if current == frame:
                        return value
                    if current > frame:
                        self.frames.appendleft((current, value))
                        raise RuntimeError(f"Sensor skipped frame {frame}; received {current}")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Sensor did not deliver frame {frame} within {timeout}s")
                self.condition.wait(remaining)
