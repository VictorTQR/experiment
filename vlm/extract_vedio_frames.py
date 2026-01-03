# -*- coding: utf-8 -*-
"""
Video Key Frame Extraction Module
提取视频关键帧的工具模块
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sink=lambda msg: print(msg, end=''),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/video_extraction_{time:YYYY-MM-DD}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
    encoding="utf-8"
)


def extract_key_frames(
    video_path: str,
    output_dir: str = "key_frames",
    threshold: float = 30.0,
    min_interval: int = 10,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Extract key frames from video based on frame difference
    从视频中提取关键帧（基于帧差异算法）

    Args:
        video_path: Path to video file / 视频文件路径
        output_dir: Directory to save key frames / 关键帧输出目录
        threshold: Frame difference threshold (0-100), lower is more sensitive / 帧差异阈值，越小越敏感
        min_interval: Minimum frame interval to avoid similar frames / 最小帧间隔，避免连续提取相似帧
        max_frames: Maximum number of frames to extract, None for unlimited / 最大提取帧数，None表示不限制
        resize: Resize frame dimensions (width, height), None to keep original / 调整帧大小，None表示保持原尺寸

    Returns:
        List of extracted key frame file paths / 提取的关键帧文件路径列表

    Example:
        >>> key_frames = extract_key_frames("video.mp4", "output", threshold=25)
        >>> print(f"Extracted {len(key_frames)} key frames")
    """
    logger.info(f"Starting key frame extraction from: {video_path}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    logger.success(f"Successfully opened video: {video_path}")

    try:
        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video Information:")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.debug(f"Extraction parameters - threshold: {threshold}, min_interval: {min_interval}, max_frames: {max_frames}")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            logger.error("Failed to read first frame from video")
            raise ValueError("Cannot read first frame")

        logger.success("Successfully read first frame")

        # Resize if specified
        if resize:
            prev_frame = cv2.resize(prev_frame, resize)
            logger.debug(f"Resized frame to {resize}")

        # Convert to grayscale for comparison
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Save first frame
        key_frame_paths = []
        frame_count = 0
        saved_count = 0

        first_frame_path = os.path.join(output_dir, f"keyframe_{saved_count:04d}.jpg")
        cv2.imwrite(first_frame_path, prev_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        key_frame_paths.append(first_frame_path)
        saved_count += 1
        logger.success(f"Saved first frame: {first_frame_path}")

        logger.info(f"Extracting key frames with threshold: {threshold}, min interval: {min_interval} frames")

        # Process all frames
        last_saved_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("Reached end of video")
                break

            frame_count += 1

            # Check minimum interval
            if frame_count - last_saved_frame < min_interval:
                continue

            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break

            # Resize if specified
            if resize:
                frame = cv2.resize(frame, resize)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, gray)
            diff_score = np.mean(diff)

            # Save as key frame if difference exceeds threshold
            if diff_score > threshold:
                key_frame_path = os.path.join(output_dir, f"keyframe_{saved_count:04d}.jpg")
                cv2.imwrite(key_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                key_frame_paths.append(key_frame_path)

                logger.info(f"Frame {frame_count}/{total_frames}: diff={diff_score:.2f} -> saved")

                # Update previous frame
                prev_gray = gray
                last_saved_frame = frame_count
                saved_count += 1

            # Periodically update reference frame
            elif frame_count % (min_interval * 2) == 0:
                prev_gray = gray
                logger.trace(f"Updated reference frame at position {frame_count}")

        logger.success(f"Extraction complete! Extracted {len(key_frame_paths)} key frames to: {output_dir}")
        return key_frame_paths

    finally:
        cap.release()
        logger.debug("Released video capture resource")


def extract_key_frames_by_time(
    video_path: str,
    output_dir: str = "key_frames",
    interval_seconds: float = 1.0,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Extract key frames at fixed time intervals
    按固定时间间隔从视频中提取关键帧

    Args:
        video_path: Path to video file / 视频文件路径
        output_dir: Directory to save key frames / 关键帧输出目录
        interval_seconds: Time interval between extractions in seconds / 提取间隔（秒）
        max_frames: Maximum number of frames to extract, None for unlimited / 最大提取帧数
        resize: Resize frame dimensions (width, height), None to keep original / 调整帧大小

    Returns:
        List of extracted key frame file paths / 提取的关键帧文件路径列表

    Example:
        >>> key_frames = extract_key_frames_by_time("video.mp4", interval_seconds=2.0)
        >>> print(f"Extracted {len(key_frames)} key frames")
    """
    logger.info(f"Starting time-based key frame extraction from: {video_path}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    logger.success(f"Successfully opened video: {video_path}")

    try:
        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video Information:")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.debug(f"Extraction parameters - interval: {interval_seconds}s, max_frames: {max_frames}")

        # Calculate frame interval
        frame_interval = int(interval_seconds * fps)
        logger.debug(f"Frame interval: {frame_interval} frames")

        key_frame_paths = []
        saved_count = 0

        logger.info(f"Extracting key frames at {interval_seconds}s intervals...")

        # Extract frames at intervals
        frame_count = 0
        while True:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            ret, frame = cap.read()
            if not ret:
                logger.debug("Reached end of video")
                break

            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break

            # Resize if specified
            if resize:
                frame = cv2.resize(frame, resize)

            # Save key frame
            key_frame_path = os.path.join(output_dir, f"keyframe_{saved_count:04d}.jpg")
            cv2.imwrite(key_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            key_frame_paths.append(key_frame_path)

            timestamp = frame_count / fps
            logger.info(f"Time {timestamp:.2f}s: saved key frame {saved_count + 1}")

            saved_count += 1
            frame_count += frame_interval

        logger.success(f"Extraction complete! Extracted {len(key_frame_paths)} key frames to: {output_dir}")
        return key_frame_paths

    finally:
        cap.release()
        logger.debug("Released video capture resource")


def extract_key_frames_histogram(
    video_path: str,
    output_dir: str = "key_frames",
    threshold: float = 0.3,
    min_interval: int = 10,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Extract key frames based on histogram difference (more accurate but slower)
    基于直方图差异从视频中提取关键帧（更准确但计算量更大）

    Args:
        video_path: Path to video file / 视频文件路径
        output_dir: Directory to save key frames / 关键帧输出目录
        threshold: Histogram difference threshold (0-1), lower is more sensitive / 直方图差异阈值
        min_interval: Minimum frame interval / 最小帧间隔
        max_frames: Maximum number of frames to extract, None for unlimited / 最大提取帧数
        resize: Resize frame dimensions (width, height), None to keep original / 调整帧大小

    Returns:
        List of extracted key frame file paths / 提取的关键帧文件路径列表

    Example:
        >>> key_frames = extract_key_frames_histogram("video.mp4", threshold=0.25)
        >>> print(f"Extracted {len(key_frames)} key frames")
    """
    logger.info(f"Starting histogram-based key frame extraction from: {video_path}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    logger.success(f"Successfully opened video: {video_path}")

    try:
        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video Information:")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.debug(f"Extraction parameters - threshold: {threshold}, min_interval: {min_interval}, max_frames: {max_frames}")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            logger.error("Failed to read first frame from video")
            raise ValueError("Cannot read first frame")

        logger.success("Successfully read first frame")

        # Resize if specified
        if resize:
            prev_frame = cv2.resize(prev_frame, resize)
            logger.debug(f"Resized frame to {resize}")

        # Calculate histogram
        prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
        logger.debug("Calculated initial histogram")

        # Save first frame
        key_frame_paths = []
        frame_count = 0
        saved_count = 0

        first_frame_path = os.path.join(output_dir, f"keyframe_{saved_count:04d}.jpg")
        cv2.imwrite(first_frame_path, prev_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        key_frame_paths.append(first_frame_path)
        saved_count += 1
        logger.success(f"Saved first frame: {first_frame_path}")

        logger.info(f"Extracting key frames (based on histogram) with threshold: {threshold}, min interval: {min_interval} frames")

        # Process all frames
        last_saved_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("Reached end of video")
                break

            frame_count += 1

            # Check minimum interval
            if frame_count - last_saved_frame < min_interval:
                continue

            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break

            # Resize if specified
            if resize:
                frame = cv2.resize(frame, resize)

            # Calculate histogram
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Calculate histogram difference (correlation)
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff_score = 1.0 - correlation  # Convert to difference score

            # Save as key frame if difference exceeds threshold
            if diff_score > threshold:
                key_frame_path = os.path.join(output_dir, f"keyframe_{saved_count:04d}.jpg")
                cv2.imwrite(key_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                key_frame_paths.append(key_frame_path)

                logger.info(f"Frame {frame_count}/{total_frames}: diff={diff_score:.3f} -> saved")

                # Update previous frame histogram
                prev_hist = hist
                last_saved_frame = frame_count
                saved_count += 1

        logger.success(f"Extraction complete! Extracted {len(key_frame_paths)} key frames to: {output_dir}")
        return key_frame_paths

    finally:
        cap.release()
        logger.debug("Released video capture resource")


# Example usage
if __name__ == "__main__":
    # Example 1: Extract based on frame difference
    logger.info("=== Method 1: Frame Difference ===")
    # key_frames = extract_key_frames(
    #     "your_video.mp4",
    #     output_dir="key_frames_diff",
    #     threshold=25.0,
    #     min_interval=15
    # )

    # Example 2: Extract by time interval
    logger.info("=== Method 2: Time Interval ===")
    # key_frames = extract_key_frames_by_time(
    #     "your_video.mp4",
    #     output_dir="key_frames_time",
    #     interval_seconds=2.0
    # )

    # Example 3: Extract based on histogram (more accurate)
    logger.info("=== Method 3: Histogram ===")
    # key_frames = extract_key_frames_histogram(
    #     "your_video.mp4",
    #     output_dir="key_frames_hist",
    #     threshold=0.3,
    #     min_interval=15
    # )

    logger.info("Please uncomment the code above and replace the video path to use")
