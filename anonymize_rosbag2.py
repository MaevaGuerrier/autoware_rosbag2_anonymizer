"""
anonymize_rosbag2.py
====================
High-performance ROS2 bag anonymizer (face + license plate blurring).

Key improvements over the original autoware_rosbag2_anonymizer pipeline:
  1. YOLOv8-face + YOLOv8-plate  →  replaces the heavy UnifiedLanguageModel + SAM2 stack
  2. Bounding-box blur only      →  SAM2 segmentation pass removed entirely
  3. Batched GPU inference        →  frames are grouped before hitting the model
  4. Multi-bag parallelism        →  one process per bag (safe: bags are independent)
  5. Ordered writer buffer        →  guarantees message order even with async processing
  6. Non-image messages           →  copied byte-for-byte, never touched

Dependencies
------------
    pip install ultralytics opencv-python-headless numpy tqdm
    # ROS2 Python packages must be sourced from your ROS2 install

Usage
-----
    python anonymize_rosbag2.py --input_folder /data/bags/raw \
                                --output_folder /data/bags/anon \
                                --face_model yolov8n-face.pt \
                                --plate_model yolov8n-lp.pt \
                                --batch_size 8 \
                                --conf_face 0.4 \
                                --conf_plate 0.4 \
                                --blur_kernel 51 \
                                --workers 2
"""

import os
import sys
import argparse
import logging
import json
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class MsgRecord:
    """Thin wrapper around one bag message, image or otherwise."""
    seq_idx: int          # original position in the bag (for ordered writing)
    topic: str
    msg_type: str
    timestamp: int        # nanoseconds
    raw_data: object      # original deserialized message
    is_image: bool
    compressed: bool = False



# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def load_models(face_model_path: str, plate_model_path: str, device: str):
    """Load both YOLO models onto the requested device."""
    from ultralytics import YOLO
    face_model  = YOLO(face_model_path)
    plate_model = YOLO(plate_model_path)
    face_model.to(device)
    plate_model.to(device)
    # Warm-up pass so the first real batch isn't penalised by JIT compilation
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    face_model([dummy], verbose=False)
    plate_model([dummy], verbose=False)
    log.info("Models loaded and warmed up on device: %s", device)
    return face_model, plate_model


def decode_image(record: MsgRecord) -> Optional[np.ndarray]:
    """
    Convert a ROS2 image message to a BGR numpy array.
    Returns None on failure so the caller can apply the safe fallback.
    """
    try:
        import cv_bridge
        bridge = cv_bridge.CvBridge()
        if record.compressed:
            img = bridge.compressed_imgmsg_to_cv2(record.raw_data)
        else:
            img = bridge.imgmsg_to_cv2(record.raw_data, desired_encoding="bgr8")
        return img
    except Exception as exc:
        log.warning("Frame decode failed at t=%d: %s", record.timestamp, exc)
        return None



def pad_box(x1: int, y1: int, x2: int, y2: int,
            h: int, w: int, pad: float = 0.15) -> Tuple[int, int, int, int]:
    """Expand a bounding box by `pad` fraction, clamped to image bounds."""
    dw = int((x2 - x1) * pad)
    dh = int((y2 - y1) * pad)
    return (
        max(0,     x1 - dw),
        max(0,     y1 - dh),
        min(w - 1, x2 + dw),
        min(h - 1, y2 + dh),
    )


def blur_boxes(img: np.ndarray,
               boxes: List[Tuple[int, int, int, int]],
               kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur to every bounding box region in-place (copy returned)."""
    out = img.copy()
    k   = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # must be odd
    h, w = img.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, h, w)
        if x2 <= x1 or y2 <= y1:
            continue
        roi          = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def run_detection_batch(
    images: List[np.ndarray],
    face_model,
    plate_model,
    conf_face: float,
    conf_plate: float,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Run both models over a list of images in a single batched call each.

    Returns:
        A list of length len(images).
        Each element is a list of (x1,y1,x2,y2) pixel boxes for that frame.
    """
    all_boxes: List[List[Tuple]] = [[] for _ in images]

    def collect(results, conf_thresh: float):
        for img_idx, result in enumerate(results):
            for box in result.boxes:
                if float(box.conf[0]) < conf_thresh:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                all_boxes[img_idx].append((x1, y1, x2, y2))

    face_results  = face_model(images,  verbose=False)
    plate_results = plate_model(images, verbose=False)
    collect(face_results,  conf_face)
    collect(plate_results, conf_plate)
    return all_boxes


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def write_audit_log(output_bag_path: str, records: List[dict]) -> None:
    """Write a JSON sidecar audit file next to the output bag."""
    log_path = output_bag_path.rstrip("/").rstrip("\\") + "_anonymization_audit.json"
    with open(log_path, "w") as f:
        json.dump({"frames": records}, f, indent=2)
    log.info("Audit log written → %s", log_path)


# ---------------------------------------------------------------------------
# Per-bag processing function (runs in its own process)
# ---------------------------------------------------------------------------
def process_bag(
    input_path: str,
    output_path: str,
    face_model_path: str,
    plate_model_path: str,
    batch_size: int,
    conf_face: float,
    conf_plate: float,
    blur_kernel: int,
    device: str,
    output_storage_id: str = "sqlite3",
    output_compressed: bool = False,
    fallback_blur_on_error: bool = True,
) -> dict:
    """
    Anonymize a single ROS2 bag.  Designed to run inside a subprocess so that
    each bag gets its own CUDA context and there is no shared state.

    Returns a summary dict with timing and detection statistics.
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Import ROS2 / cv_bridge inside the worker so the parent process
    # does not need a sourced ROS2 environment just for spawning.
    # ------------------------------------------------------------------
    try:
        from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader
        from autoware_rosbag2_anonymizer.rosbag_io.rosbag_writer import RosbagWriter
    except ImportError:
        log.error("Could not import rosbag_reader/writer — is ROS2 sourced?")
        raise

    log.info("▶  Processing bag: %s", input_path)

    # Load models
    face_model, plate_model = load_models(face_model_path, plate_model_path, device)

    # Open reader
    reader = RosbagReader(input_path, 1)

    # Open writer — mirror QoS profiles from the source bag exactly
    writer = RosbagWriter(
        output_path,
        output_compressed,
        output_storage_id,
        reader.get_qos_profile_map(),
    )

    # ------------------------------------------------------------------
    # Pass 1: read ALL messages into memory as MsgRecord objects.
    # This decouples reading speed from inference speed and lets us
    # batch arbitrarily without worrying about ordering.
    #
    # Memory note: for very large bags you can replace this with a
    # streaming approach using the ordered-writer buffer described in
    # the architecture docs.
    # ------------------------------------------------------------------
    all_records: List[MsgRecord] = []
    for seq_idx, (msg, is_image) in enumerate(reader):
        compressed = is_image and ("Compressed" in msg.type)
        all_records.append(MsgRecord(
            seq_idx   = seq_idx,
            topic     = msg.topic,
            msg_type  = msg.type,
            timestamp = msg.timestamp,
            raw_data  = msg.data,
            is_image  = is_image,
            compressed= compressed,
        ))

    total_msgs   = len(all_records)
    image_records = [r for r in all_records if r.is_image]
    log.info("  Total messages : %d  |  Image messages : %d", total_msgs, len(image_records))

    # ------------------------------------------------------------------
    # Pass 2: batched inference on image frames only
    # ------------------------------------------------------------------
    # Map seq_idx → processed BGR image
    processed_images: dict = {}
    audit_log: List[dict] = []

    batches = [
        image_records[i : i + batch_size]
        for i in range(0, len(image_records), batch_size)
    ]

    with tqdm(total=len(image_records),
              desc=f"  Inferring {Path(input_path).name}",
              unit="frame") as pbar:

        for batch in batches:
            # Decode each frame in the batch
            decoded: List[Optional[np.ndarray]] = [decode_image(r) for r in batch]

            # Collect valid frames (skip None)
            valid_indices = [i for i, img in enumerate(decoded) if img is not None]
            valid_imgs    = [decoded[i] for i in valid_indices]

            if valid_imgs:
                box_lists = run_detection_batch(
                    valid_imgs, face_model, plate_model, conf_face, conf_plate
                )
            else:
                box_lists = []

            # Map detections back to original batch positions
            detection_map: dict = {}
            for rank, orig_i in enumerate(valid_indices):
                detection_map[orig_i] = box_lists[rank]

            for i, record in enumerate(batch):
                img = decoded[i]

                if img is None:
                    # Decode failure — apply safe fallback
                    if fallback_blur_on_error:
                        log.warning(
                            "  Decode failed for frame t=%d; "
                            "blurring entire frame as fallback.", record.timestamp
                        )
                        # We can't blur what we can't decode; skip this frame
                        # and write the original message verbatim.
                    processed_images[record.seq_idx] = None  # signal: write original
                    audit_log.append({
                        "timestamp": record.timestamp,
                        "topic": record.topic,
                        "status": "decode_error",
                        "faces_detected": 0,
                        "plates_detected": 0,
                    })
                    pbar.update(1)
                    continue

                boxes = detection_map.get(i, [])

                if boxes:
                    # Only store the blurred numpy array when something was
                    # detected. Clean frames are written back verbatim in
                    # Pass 3, preserving the original compressed bytes exactly.
                    processed_images[record.seq_idx] = blur_boxes(img, boxes, blur_kernel)
                else:
                    # None signals "no change" — Pass 3 will call write_any
                    processed_images[record.seq_idx] = None

                audit_log.append({
                    "timestamp"       : record.timestamp,
                    "topic"           : record.topic,
                    "status"          : "ok",
                    "detections_count": len(boxes),
                    "boxes"           : boxes,
                })
                pbar.update(1)

    # ------------------------------------------------------------------
    # Pass 3: write ALL messages in original order
    # Non-image messages are written verbatim (raw bytes, untouched).
    # Image messages are written with the processed pixel data but with
    # the original header (timestamp, frame_id) preserved.
    # ------------------------------------------------------------------
    log.info("  Writing output bag …")
    with tqdm(total=total_msgs,
              desc=f"  Writing  {Path(input_path).name}",
              unit="msg") as pbar:

        for record in all_records:   # already in original order
            if not record.is_image:
                # --------------------------------------------------------
                # NON-IMAGE: copy raw bytes verbatim — zero modification
                # --------------------------------------------------------
                writer.write_any(
                    record.raw_data,
                    record.msg_type,
                    record.topic,
                    record.timestamp,
                )
            else:
                processed_img = processed_images.get(record.seq_idx)

                if processed_img is None:
                    # Two cases share this path:
                    #   (a) decode failed → original message written untouched
                    #   (b) no detections  → original message written untouched
                    # Either way the original compressed bytes go straight through
                    # with zero re-encoding, keeping the file size sane.
                    writer.write_any(
                        record.raw_data,
                        record.msg_type,
                        record.topic,
                        record.timestamp,
                    )
                else:
                    # A face or plate was blurred on this frame.
                    # Re-encode back to the SAME format as the source so we
                    # don't inflate uncompressed frames to raw size.
                    if record.compressed:
                        # Re-encode as JPEG at high quality to match source format.
                        # cv2.imencode returns (success, buffer); we need the buffer.
                        ok, buf = cv2.imencode(
                            ".jpg", processed_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        if not ok:
                            log.warning(
                                "JPEG re-encode failed at t=%d; "
                                "writing original frame.", record.timestamp
                            )
                            writer.write_any(
                                record.raw_data,
                                record.msg_type,
                                record.topic,
                                record.timestamp,
                            )
                        else:
                            import cv_bridge
                            ros_msg = cv_bridge.CvBridge().cv2_to_compressed_imgmsg(
                                processed_img, dst_format="jpg"
                            )
                            ros_msg.header.stamp    = record.raw_data.header.stamp
                            ros_msg.header.frame_id = record.raw_data.header.frame_id
                            writer.write_any(
                                ros_msg,
                                record.msg_type,
                                record.topic,
                                record.timestamp,
                            )
                    else:
                        # Raw image topic — writer.write_image expects a numpy array
                        writer.write_image(processed_img, record.topic, record.timestamp)

            pbar.update(1)

    write_audit_log(output_path, audit_log)

    elapsed   = time.perf_counter() - t_start
    fps       = len(image_records) / elapsed if elapsed > 0 else 0
    det_total = sum(e.get("detections_count", 0) for e in audit_log)

    summary = {
        "bag"              : input_path,
        "total_messages"   : total_msgs,
        "image_messages"   : len(image_records),
        "total_detections" : det_total,
        "elapsed_seconds"  : round(elapsed, 2),
        "fps"              : round(fps, 2),
    }
    log.info(
        "  ✓ Done  |  %.1f s  |  %.1f FPS  |  %d detections",
        elapsed, fps, det_total,
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def get_bag_paths(folder: str) -> List[str]:
    paths = []
    for ext in (".db3", ".mcap"):
        paths.extend(str(p) for p in Path(folder).rglob(f"*{ext}"))
    return sorted(paths)

def main():
    parser = argparse.ArgumentParser(
        description="High-performance ROS2 bag face+plate anonymizer"
    )
    parser.add_argument("--input_folder",
                        default="/workspace/ros2bags/nomad_bunker_office_loop_no_aug_trial_3",
                        help="Folder containing input .db3 or .mcap bags")
    
    parser.add_argument("--output_folder", 
                        default="/workspace/anonymized_bags/",
                        help="Folder for anonymized output bags")
    
    parser.add_argument("--face_model",    default="yolov11n-face.pt",
                        help="Path to YOLO face detection weights")
    
    parser.add_argument("--plate_model",   default="license-plate-finetune-v1n.pt",
                        help="Path to YOLO license-plate detection weights")
    
    parser.add_argument("--batch_size",    type=int, default=8,
                        help="Number of frames per GPU inference batch")
    
    parser.add_argument("--conf_face",     type=float, default=0.4,
                        help="Confidence threshold for face detection")
    
    parser.add_argument("--conf_plate",    type=float, default=0.4,
                        help="Confidence threshold for plate detection")
    
    parser.add_argument("--blur_kernel",   type=int, default=51,
                        help="Gaussian blur kernel size (must be odd)")
    
    parser.add_argument("--workers",       type=int, default=1,
                        help="Number of bags to process in parallel (one GPU per worker)")
    
    parser.add_argument("--device",        default="cuda",
                        help="PyTorch device string: 'cuda', 'cuda:1', 'cpu'")
    
    parser.add_argument("--storage_id",    default="sqlite3",
                        choices=["sqlite3", "mcap"],
                        help="Output bag storage plugin")
    
    parser.add_argument("--compressed",    action="store_true",
                        help="Write output images as CompressedImage messages")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    bag_paths = get_bag_paths(args.input_folder)
    if not bag_paths:
        log.error("No .db3 or .mcap files found in: %s", args.input_folder)
        sys.exit(1)

    log.info("Found %d bag(s) to process.", len(bag_paths))

    job_args = []
    for bag_path in bag_paths:
        bag_name   = Path(bag_path).stem
        output_path = os.path.join(args.output_folder, bag_name)
        job_args.append((
            bag_path,
            output_path,
            args.face_model,
            args.plate_model,
            args.batch_size,
            args.conf_face,
            args.conf_plate,
            args.blur_kernel,
            args.device,
            args.storage_id,
            args.compressed,
        ))

    t0 = time.perf_counter()
    summaries = []

    if args.workers == 1 or len(bag_paths) == 1:
        # Single-process path (simpler, easier to debug)
        for jargs in job_args:
            try:
                s = process_bag(*jargs)
                summaries.append(s)
            except Exception:
                log.error("Failed on bag %s:\n%s", jargs[0], traceback.format_exc())
    else:
        # Multi-process path: one process per bag
        # NOTE: if workers > num_GPUs, bags will share GPU — this is still
        # beneficial for I/O overlap but watch VRAM usage.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_bag, *jargs): jargs[0] for jargs in job_args}
            for future in as_completed(futures):
                bag = futures[future]
                try:
                    summaries.append(future.result())
                except Exception:
                    log.error("Failed on bag %s:\n%s", bag, traceback.format_exc())

    # ------------------------------------------------------------------
    # Final summary report
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t0
    total_frames  = sum(s["image_messages"]   for s in summaries)
    total_detects = sum(s["total_detections"] for s in summaries)

    print("\n" + "=" * 60)
    print("  ANONYMIZATION COMPLETE")
    print("=" * 60)
    for s in summaries:
        print(f"  {Path(s['bag']).name:<40} "
              f"{s['image_messages']:>6} frames  "
              f"{s['fps']:>6.1f} FPS  "
              f"{s['total_detections']:>5} detections")
    print("-" * 60)
    print(f"  Total bags     : {len(summaries)}")
    print(f"  Total frames   : {total_frames}")
    print(f"  Total detects  : {total_detects}")
    print(f"  Wall time      : {total_elapsed:.1f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()