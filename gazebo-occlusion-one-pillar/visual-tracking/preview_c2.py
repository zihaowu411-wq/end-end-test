"""Render a 4x-speed Gazebo preview of an actual scripted Stage C2 target route."""

import argparse
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from multi_pillar_env import MultiPillarQuadTracking


def preview(output, seed, world):
    output.mkdir(parents=True, exist_ok=False)
    env = MultiPillarQuadTracking(world, output/"sim", stage="C2", record=True, rtf=30,
                                  cube_target=True, camera_resolution=(256,144), target_gap_m=2.0,
                                  tracking_reward="safe-effective-tracking")
    ffmpeg = None
    try:
        obs, info = env.reset(seed=seed)
        assert obs.shape == (3,144,256) and info["visible"] and info["valid"]
        stats = env.layout
        config = env.layout_config(stats.kind)
        duration_s = config["durations_s"][stats.kind]
        assert len(stats.pillars)==16 and len(stats.encounters)==4
        fps, skip = 20, 4
        width, height = 960, 648
        frame_y, overhead_h = 54, 540
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        title_font = ImageFont.truetype(font, 22)
        small_font = ImageFont.truetype(font,16)
        ffmpeg=subprocess.Popen(
            ["ffmpeg","-nostdin","-v","error","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{width}x{height}","-r",str(fps),"-i","-","-c:v","libx264",
             "-preset","veryfast","-pix_fmt","yuv420p","-crf","19","-movflags","+faststart",str(output/"preview.mp4")],
            stdin=subprocess.PIPE)
        pixels_per_m = width/(2*math.tan(.86/2)*(107-env.spawn_z))

        def screen(x, y):
            return width/2 - y*pixels_per_m, frame_y + overhead_h/2 - x*pixels_per_m

        planned = [screen(*xy) for xy in stats.xy[::4]]
        traveled = []
        target_clearance = []
        first_pass_times = {}
        for tick in range(env.duration+1):
            if tick > 0:
                _, _, terminated, truncated, info = env.step(env.neutral_action)
                if terminated or truncated != (tick==env.duration):
                    raise RuntimeError(f"Unexpected end of the preview at step {tick}: {info}")
            expected_x, expected_y = stats.xy[tick]
            if math.hypot(info["target_x"]-expected_x,info["target_y"]-expected_y) > 1e-6:
                raise RuntimeError(f"Gazebo target pose differs from its sampled trajectory at step {tick}")
            target_clearance.append(info["target_clearance_m"])
            for encounter in stats.encounters:
                if tick == encounter.sample_index:
                    first_pass_times[encounter.pillar_index] = info["time"]
            if tick % skip:
                continue
            position = screen(info["target_x"], info["target_y"])
            traveled.append(position)
            canvas = Image.new("RGB",(width,height),"#17252c")
            canvas.paste(Image.fromarray(env.overhead_frame), (0,frame_y))
            draw = ImageDraw.Draw(canvas)
            draw.line(planned,fill="#153b3d",width=5)
            draw.line(planned,fill="#8ce0e0",width=2)
            if len(traveled) > 1:
                draw.line(traveled,fill="#203638",width=7)
                draw.line(traveled,fill="#f9d86b",width=4)
            x,y = position
            draw.ellipse((x-6,y-6,x+6,y+6),outline="white",width=2)
            for number, encounter in enumerate(stats.encounters,1):
                p=stats.pillars[encounter.pillar_index]
                px,py=screen(p.x,p.y)
                draw.ellipse((px-10,py-10,px+10,py+10),fill="#263944")
                draw.text((px,py),str(number),fill="white",font=small_font,anchor="mm")
            draw.text((22,14),f"C2 crossing route  |  Seed {seed}",fill="white",font=title_font)
            draw.text((width-22,16),f"{info['time']:05.1f} / {duration_s} s  |  4x",
                      font=small_font,fill="white",anchor="ra")
            draw.text((22,615),"Cyan: planned route    Yellow: target trail    1-4: bypass sequence",
                      font=small_font,fill="#edf7f7")
            ffmpeg.stdin.write(canvas.tobytes())
        if len(first_pass_times)!=4:
            raise RuntimeError("The target did not reach all four bypass stations")
        if min(target_clearance)<config["minimum_target_clearance_m"]:
            raise RuntimeError("The Gazebo target violated its clearance threshold")
        ffmpeg.stdin.close()
        if ffmpeg.wait()!=0:
            raise RuntimeError("FFmpeg failed to complete the preview")
        ffmpeg=None
        report=dict(seed=seed, config=config, speed_mps=stats.speed_mps,
                    conservative_minimum_pillar_separation_m=stats.minimum_pillar_separation_m,
                    conservative_minimum_target_clearance_m=stats.minimum_target_clearance_m,
                    gazebo_minimum_target_clearance_m=min(target_clearance),
                    minimum_turn_radius_m=stats.minimum_turn_radius_m,
                    pillars=[dict(x=p.x,y=p.y,yaw=p.yaw,half_size=list(p.half_size)) for p in stats.pillars],
                    encounters=[dict(pillar_index=e.pillar_index,side="right" if e.side>0 else "left",
                                     time_s=first_pass_times[e.pillar_index]) for e in stats.encounters],
                    route_xy_m=stats.xy[::4].tolist())
        (output/"layout.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k not in ("pillars","route_xy_m")}),flush=True)
    finally:
        if ffmpeg is not None:
            if not ffmpeg.stdin.closed:
                ffmpeg.stdin.close()
            ffmpeg.wait()
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--world",type=Path,default=Path("../one_pillar.sdf"))
    args=parser.parse_args()
    preview(args.output,args.seed,args.world)
