"""Video processing utilities for colorization.

Provides frame extraction, video assembly, and end-to-end video colorization.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python not installed. Video processing will not be available.")
    print("Install with: pip install opencv-python")


def extract_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: Optional[int] = None,
    fps: Optional[float] = None
) -> Tuple[List[Path], float, Tuple[int, int]]:
    """Extract frames from video file.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        max_frames: Optional limit on number of frames to extract
        fps: Optional target fps (None = use original fps)

    Returns:
        frame_paths: List of paths to extracted frames
        original_fps: Original video frame rate
        size: (width, height) of frames
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for video processing")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame sampling
    if fps and fps < original_fps:
        frame_skip = int(original_fps / fps)
    else:
        frame_skip = 1

    if max_frames and total_frames > max_frames * frame_skip:
        frame_skip = max(frame_skip, total_frames // max_frames)

    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    print(f"Extracting frames from: {video_path.name}")
    print(f"  Original: {original_fps:.2f} fps, {total_frames} frames, {width}x{height}")
    print(f"  Sampling: every {frame_skip} frame(s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Save frame
            frame_path = output_dir / f"frame_{saved_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_idx += 1

            if saved_idx % 100 == 0:
                print(f"  Extracted {saved_idx} frames...")

            if max_frames and saved_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"  Total extracted: {len(frame_paths)} frames")

    return frame_paths, original_fps, (width, height)


def frames_to_video(
    frame_paths: List[Path],
    output_path: Path,
    fps: float = 30.0,
    codec: str = 'mp4v'
):
    """Reassemble frames into video file.

    Args:
        frame_paths: List of paths to frame images (in order)
        output_path: Output video path
        fps: Target frames per second
        codec: Video codec (mp4v, avc1, etc.)
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for video processing")

    if not frame_paths:
        raise ValueError("No frames to assemble into video")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_path}")

    print(f"Assembling video: {output_path.name}")
    print(f"  {len(frame_paths)} frames at {fps} fps")
    print(f"  Resolution: {width}x{height}")

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  Warning: Could not read frame {i}: {frame_path}")
            continue

        out.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  Written {i + 1}/{len(frame_paths)} frames...")

    out.release()
    print(f"  Video saved: {output_path}")


def frames_to_gif(
    frame_paths: List[Path],
    output_path: Path,
    fps: float = 10.0,
    loop: int = 0,
    optimize: bool = True
):
    """Create animated GIF from frames.

    Args:
        frame_paths: List of paths to frame images
        output_path: Output GIF path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        optimize: Optimize GIF size
    """
    if not frame_paths:
        raise ValueError("No frames to create GIF")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating GIF: {output_path.name}")
    print(f"  {len(frame_paths)} frames at {fps} fps")

    # Load frames
    frames = [Image.open(fp).convert('RGB') for fp in frame_paths]

    # Calculate duration per frame (milliseconds)
    duration = int(1000 / fps)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize
    )

    print(f"  GIF saved: {output_path}")


def frames_to_comparison_gif(
    gray_frame_paths: List[Path],
    color_frame_paths: List[Path],
    output_path: Path,
    fps: float = 10.0,
    loop: int = 0,
    optimize: bool = True
):
    """Create animated GIF with grayscale and colorized frames side-by-side.

    Args:
        gray_frame_paths: List of paths to grayscale frame images
        color_frame_paths: List of paths to colorized frame images
        output_path: Output GIF path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        optimize: Optimize GIF size
    """
    if not gray_frame_paths or not color_frame_paths:
        raise ValueError("No frames to create GIF")

    if len(gray_frame_paths) != len(color_frame_paths):
        raise ValueError(f"Frame count mismatch: {len(gray_frame_paths)} gray vs {len(color_frame_paths)} color")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating comparison GIF: {output_path.name}")
    print(f"  {len(gray_frame_paths)} frames at {fps} fps")
    print(f"  Layout: Grayscale | Colorized")

    # Load and combine frames side-by-side
    combined_frames = []
    for gray_path, color_path in zip(gray_frame_paths, color_frame_paths):
        gray_img = Image.open(gray_path).convert('RGB')
        color_img = Image.open(color_path).convert('RGB')

        # Ensure both images have the same height
        if gray_img.size != color_img.size:
            # Resize to match the color image size
            gray_img = gray_img.resize(color_img.size, Image.BICUBIC)

        # Create side-by-side image
        w, h = color_img.size
        combined = Image.new('RGB', (w * 2, h))
        combined.paste(gray_img, (0, 0))
        combined.paste(color_img, (w, 0))

        combined_frames.append(combined)

    # Calculate duration per frame (milliseconds)
    duration = int(1000 / fps)

    # Save as GIF
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize
    )

    print(f"  GIF saved: {output_path}")


def colorize_video(
    video_path: Path,
    output_path: Path,
    inference_manager,
    temperature: float = None,
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
    create_gif: bool = False,
    keep_frames: bool = False
):
    """End-to-end video colorization pipeline.

    Args:
        video_path: Input video path
        output_path: Output video path
        inference_manager: ColorizationInference instance
        temperature: Annealing temperature (None = use config)
        max_frames: Limit number of frames to process
        target_fps: Target fps (None = use original)
        create_gif: Also create animated GIF
        keep_frames: Keep intermediate frame files
    """
    print("=" * 70)
    print("Video Colorization Pipeline")
    print("=" * 70)
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print()

    # Create working directory
    work_dir = output_path.parent / f"{output_path.stem}_frames"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_dir = work_dir / "input"
    gray_dir = work_dir / "grayscale"
    color_dir = work_dir / "colorized"

    try:
        # Step 1: Extract frames
        print("Step 1: Extracting frames")
        print("-" * 70)
        frame_paths, original_fps, (w, h) = extract_frames(
            video_path,
            input_dir,
            max_frames=max_frames,
            fps=target_fps
        )
        print()

        # Step 2: Colorize frames
        print("Step 2: Colorizing frames")
        print("-" * 70)
        gray_dir.mkdir(exist_ok=True)
        color_dir.mkdir(exist_ok=True)
        grayscale_paths = []
        colorized_paths = []

        for i, frame_path in enumerate(frame_paths):
            # Colorize
            result = inference_manager.colorize_image(
                frame_path,
                temperature=temperature
            )

            # Save grayscale L channel frame
            gray_path = gray_dir / frame_path.name
            L_uint8 = (result['L'] * 255).astype(np.uint8)
            # Convert single channel to RGB for GIF compatibility
            L_rgb = np.stack([L_uint8, L_uint8, L_uint8], axis=-1)
            Image.fromarray(L_rgb).save(gray_path)
            grayscale_paths.append(gray_path)

            # Save colorized frame
            color_path = color_dir / frame_path.name
            rgb_uint8 = (result['rgb'] * 255).astype(np.uint8)
            Image.fromarray(rgb_uint8).save(color_path)
            colorized_paths.append(color_path)

            if (i + 1) % 50 == 0:
                print(f"  Colorized {i + 1}/{len(frame_paths)} frames...")

        print(f"  Total colorized: {len(frame_paths)} frames")
        print()

        # Step 3: Reassemble video
        print("Step 3: Assembling output video")
        print("-" * 70)
        use_fps = target_fps or original_fps
        frames_to_video(colorized_paths, output_path, fps=use_fps)
        print()

        # Step 4: Optional GIF
        if create_gif:
            print("Step 4: Creating animated GIF")
            print("-" * 70)
            gif_path = output_path.with_suffix('.gif')
            gif_fps = min(use_fps, 15)  # Cap GIF fps for reasonable file size
            frames_to_comparison_gif(grayscale_paths, colorized_paths, gif_path, fps=gif_fps)
            print()

        print("=" * 70)
        print("Video colorization complete!")
        print(f"Output video: {output_path}")
        if create_gif:
            print(f"Output GIF: {gif_path}")
        print("=" * 70)

    finally:
        # Cleanup
        if not keep_frames:
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir)
                print(f"Cleaned up temporary frames: {work_dir}")
