from PIL import Image, ImageDraw, ImageFont
import subprocess

from tracking_env import GOAL, TOLERANCE

RESAMPLING = getattr(Image, "Resampling", Image)


class Video:
    def __init__(self, filename, stage):
        self.process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "1280x664", "-r", "6", "-i", "-", "-c:v", "libx264", "-pix_fmt",
             "yuv420p", "-crf", "21", "-movflags", "+faststart", str(filename)],
            stdin=subprocess.PIPE,
        )
        self.stage = stage
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.large = ImageFont.truetype(font, 23)
        self.small = ImageFont.truetype(font, 16)

    def add(self, env, info, seed, history):
        frame = Image.new("RGB", (1280, 664), (244, 244, 237))
        draw = ImageDraw.Draw(frame)
        draw.text((24, 14), f"Stage {self.stage} | Seed {seed} | {info['time']:.1f} s | "
                  f"Gap {info['distance']:.2f} m | {'visible' if info['visible'] else 'lost'}",
                  fill="#19282b", font=self.large)
        draw.text((24, 48), "Pursuer onboard RGB", fill="#172a32", font=self.small)
        draw.text((680, 48), "Overhead camera", fill="#172a32", font=self.small)
        frame.paste(Image.fromarray(env.front_frame).resize((640, 360), RESAMPLING.BILINEAR), (24, 73))
        frame.paste(Image.fromarray(env.overhead_frame).resize((576, 432)), (680, 73))
        draw.text((24, 459), "Vehicle gap (m)", fill="#172a32", font=self.small)
        left, right, top, bottom = 86, 1240, 512, 626
        ceiling = 8.0
        chart_y = lambda gap: bottom - (bottom - top) * min(ceiling, gap) / ceiling
        seconds = env.duration / 6
        chart_x = lambda time: left + (right - left) * time / seconds
        draw.rectangle((left, top, right, bottom), fill="#fcfcf9", outline="#899593")
        draw.rectangle((left + 1, chart_y(GOAL + TOLERANCE), right - 1, chart_y(GOAL - TOLERANCE)),
                       fill="#d7eacb")
        draw.line((left, chart_y(GOAL), right, chart_y(GOAL)), fill="#4b7e60", width=1)
        for gap in (0, GOAL, ceiling):
            draw.text((54, chart_y(gap) - 9), str(gap), fill="#293733", font=self.small)
        for t in (0, int(seconds / 2), int(seconds)):
            draw.text((chart_x(t) - 9, bottom + 7), f"{t}s", fill="#293733", font=self.small)
        if len(history) > 1:
            draw.line([(chart_x(row["time"]), chart_y(row["distance"])) for row in history],
                      fill="#1d4767", width=3)
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("FFmpeg could not finalize the recording")
