"""Record time-matched RGB frames from a scripted pass behind the pillar."""
from collections import deque
from threading import Lock
from time import monotonic, sleep

from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node
from PIL import Image as PILImage, ImageChops

FRONT_TOPIC = "/occlusion/front/rgb"
OVERHEAD_TOPIC = "/occlusion/overview/rgb"


def decode(message):
    if message.step != message.width * 3 or len(message.data) != message.step * message.height:
        raise ValueError("Expected packed RGB8 frames")
    timestamp = message.header.stamp.sec + message.header.stamp.nsec * 1e-9
    frame = PILImage.frombytes("RGB", (message.width, message.height), message.data)
    return timestamp, frame


def red_pixels(frame):
    """Count pixels dominated by red, used only to select the demo snapshots."""
    red, green, blue = frame.split()
    contrast = ImageChops.subtract(red, ImageChops.lighter(green, blue))
    return sum(contrast.histogram()[60:])


def record_pass():
    node = Node()
    lock = Lock()
    front_frames, overhead_frames = deque(maxlen=100), deque(maxlen=100)

    def on_front(message):
        timestamp, frame = decode(message)
        with lock:
            front_frames.append((timestamp, frame, red_pixels(frame)))

    def on_overhead(message):
        sample = decode(message)
        with lock:
            overhead_frames.append(sample)

    if not node.subscribe(Image, FRONT_TOPIC, on_front):
        raise RuntimeError("Unable to subscribe to front RGB")
    if not node.subscribe(Image, OVERHEAD_TOPIC, on_overhead):
        raise RuntimeError("Unable to subscribe to overhead RGB")
    drive = node.advertise("/model/target/cmd_vel", Twist)

    deadline = monotonic() + 70
    start = None
    phases = {}
    last_command = 0.0
    last_frame_time = -1.0

    try:
        while monotonic() < deadline:
            with lock:
                fronts = list(front_frames)
                overheads = list(overhead_frames)
            if not fronts or not overheads:
                sleep(0.04)
                continue

            if start is None:
                stamp, frame, red_count = fronts[0]
                nearest = min(overheads, key=lambda sample: abs(sample[0] - stamp))
                if abs(nearest[0] - stamp) > 0.045:
                    with lock:
                        if front_frames and front_frames[0][0] == stamp:
                            front_frames.popleft()
                    continue
                if red_count < 12:
                    raise RuntimeError(f"Target is not visible at the starting pose ({red_count} red pixels)")
                start = last_frame_time = stamp
                phases["before"] = (stamp, frame, nearest[1], red_count)
                print(f"RGB streams ready at t={start:.2f} s", flush=True)

            now = monotonic()
            if now - last_command >= 0.22:
                command = Twist()
                command.linear.x = 0.95
                drive.publish(command)
                last_command = now

            for stamp, frame, red_count in fronts:
                if stamp <= last_frame_time:
                    continue
                nearest = min(overheads, key=lambda sample: abs(sample[0] - stamp))
                if abs(nearest[0] - stamp) > 0.045:
                    # Keep the sample eligible until the matching overhead frame arrives.
                    if overheads[-1][0] < stamp:
                        break
                    continue
                last_frame_time = stamp
                elapsed = stamp - start
                if 2.5 <= elapsed <= 5.5 and red_count == 0:
                    old = phases.get("hidden")
                    if old is None or abs(elapsed - 3.9) < abs(old[0] - start - 3.9):
                        phases["hidden"] = (stamp, frame, nearest[1], red_count)
                if 6.0 <= elapsed <= 8.2 and "hidden" in phases and red_count >= 12:
                    old = phases.get("after")
                    if old is None or abs(elapsed - 7.4) < abs(old[0] - start - 7.4):
                        phases["after"] = (stamp, frame, nearest[1], red_count)

            if fronts[-1][0] - start >= 7.6:
                break
            sleep(0.04)
    finally:
        drive.publish(Twist())
        node.unsubscribe(FRONT_TOPIC)
        node.unsubscribe(OVERHEAD_TOPIC)
        sleep(0.2)

    if set(phases) != {"before", "hidden", "after"}:
        last_rgb = "none" if not front_frames else f"{front_frames[-1][0]:.2f}s"
        observations = [(round(stamp - (start or 0), 1), count) for stamp, _, count in list(front_frames)[::5]]
        raise RuntimeError(f"Incomplete timeline {list(phases)}. Last frame {last_rgb}. Samples {observations}")
    for phase in ("before", "hidden", "after"):
        stamp, front, overhead, count = phases[phase]
        front.save(f"{phase}_front.png")
        overhead.save(f"{phase}_overhead.png")
        print(f"{phase}: t={stamp:.2f} s, target red pixels={count}", flush=True)


if __name__ == "__main__":
    record_pass()
