import argparse

from PIL import Image, ImageDraw

from config import dqn_env_cfg
from env_setup import make_env


def render_track_frame(seed: int, scale: int) -> Image.Image:
    env = make_env(dqn_env_cfg, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    env.close()
    image = Image.fromarray(frame)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    return image


def render_track_map(seed: int, size: int, line_width: int) -> Image.Image:
    env = make_env(dqn_env_cfg, render_mode=None)
    env.reset(seed=seed)
    track = getattr(env.unwrapped, "track", None)
    env.close()

    if not track:
        raise RuntimeError(f"Could not read track coordinates for seed {seed}")

    points = [(tile[2], tile[3]) for tile in track]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    margin = size * 0.08
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = min(
        (size - 2 * margin) / (max_x - min_x),
        (size - 2 * margin) / (max_y - min_y),
    )

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = margin + (x - min_x) * scale
        py = size - (margin + (y - min_y) * scale)
        return px, py

    projected = [project(point) for point in points]
    image = Image.new("RGB", (size, size), color=(242, 244, 247))
    draw = ImageDraw.Draw(image)

    closed_track = projected + [projected[0]]
    draw.line(closed_track, fill=(42, 48, 57), width=line_width + 10, joint="curve")
    draw.line(closed_track, fill=(94, 103, 116), width=line_width, joint="curve")
    draw.line(closed_track, fill=(225, 230, 235), width=max(2, line_width // 8), joint="curve")

    start = projected[0]
    radius = max(5, line_width // 3)
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill=(33, 150, 83),
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a vertical PNG showing three possible CarRacing tracks."
    )
    parser.add_argument("--seeds", nargs=3, type=int, default=[42, 27734, 54362])
    parser.add_argument("--output", default="car_racing_tracks_3x1.png")
    parser.add_argument("--label", action="store_true", help="Draw seed labels on each image.")
    parser.add_argument(
        "--mode",
        choices=["map", "frame"],
        default="map",
        help="map draws a clean top-down layout; frame saves the simulator camera view.",
    )
    parser.add_argument("--size", type=int, default=700, help="Track map size in pixels.")
    parser.add_argument("--line-width", type=int, default=34, help="Track map road width.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for frame mode.")
    args = parser.parse_args()

    images = []
    for seed in args.seeds:
        if args.mode == "map":
            image = render_track_map(seed, args.size, args.line_width)
        else:
            image = render_track_frame(seed, args.scale)
        if args.label:
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 8, 108, 34), fill=(0, 0, 0))
            draw.text((14, 14), f"seed {seed}", fill=(255, 255, 255))
        images.append(image)

    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    output = Image.new("RGB", (width, height), color=(255, 255, 255))

    y = 0
    for image in images:
        output.paste(image, (0, y))
        y += image.height

    output.save(args.output)
    print(f"Saved {args.output}")
    print(f"Seeds: {args.seeds}")
    print(f"Size: {output.width}x{output.height}")


if __name__ == "__main__":
    main()
