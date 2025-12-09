import cv2
import numpy as np
import time
import argparse
import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from collections import deque
from ultralytics import YOLO
import threading
import queue

# Try to import text-to-speech (optional)
try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ============================================================================
# Configuration & Data Classes
# ============================================================================

@dataclass
class NavigationConfig:
    """Configuration for navigation system"""
    # Detection thresholds
    blind_path_ratio_threshold: float = 0.005
    crossing_ratio_threshold: float = 0.002
    crossing_bottom_ratio: float = 0.005  # Lowered from 0.01 for better detection

    # Obstacle detection
    obstacle_area_min_ratio: float = 0.003
    obstacle_overlap_threshold: float = 0.05
    ego_zone_height: float = 0.90

    # Timing thresholds (seconds)
    blocking_onpath_duration: float = 0.8
    blocking_front_duration: float = 2.5
    walk_signal_stable_duration: float = 1.0  # Walk signal must be stable for 1s

    # Message display durations
    guidance_message_duration: float = 2.0  # Quick guidance messages
    warning_message_sticky: bool = True  # Warnings stay until cleared

    # Navigation parameters
    orientation_threshold_deg: float = 10.0
    offset_threshold_ratio: float = 0.10
    nav_guidance_interval: float = 2.0
    nav_smoothing_frames: int = 3

    # Detection intervals
    obstacle_detection_interval: int = 3
    traffic_light_detection_interval: int = 2

    # UI parameters
    panel_width: int = 580
    max_quick_messages: int = 3
    max_warning_messages: int = 3

    # Voice settings
    voice_enabled: bool = TTS_AVAILABLE
    voice_rate: int = 150
    voice_volume: float = 0.9


class WorkflowState(Enum):
    """Main workflow states - mutually exclusive"""
    IDLE = "idle"
    BLIND_PATH_ACTIVE = "blind_path_active"
    BLIND_PATH_PAUSED = "blind_path_paused"
    ROAD_CROSSING_ACTIVE = "road_crossing_active"
    ROAD_CROSSING_PAUSED = "road_crossing_paused"


class ObstacleState(Enum):
    """Obstacle detection state"""
    CLEAR = "clear"
    DETECTED = "detected"


class TrafficSignalState(Enum):
    """Traffic signal detection states"""
    UNKNOWN = "unknown"
    WALK = "walk"
    DONT_WALK = "dont_walk"
    NO_SIGNAL = "no_signal"


class NavDirection(Enum):
    """Navigation directions"""
    STRAIGHT = "straight"
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class MessageType(Enum):
    """Message types for display"""
    QUICK_GUIDANCE = "quick_guidance"  # 1-2s iteration
    PERSISTENT_WARNING = "persistent_warning"  # Stays until cleared


@dataclass
class Message:
    """Display message with type and metadata"""
    text: str
    msg_type: MessageType
    timestamp: float
    category: str
    expires_at: float
    priority: int  # Higher = more important


@dataclass
class ObstacleInfo:
    """Obstacle detection information"""
    label: str
    bbox: Tuple[int, int, int, int]
    distance_score: float
    on_path: bool
    in_front: bool


# ============================================================================
# Voice Output Manager
# ============================================================================

class VoiceManager:
    """Manages text-to-speech output with queue"""

    def __init__(self, enabled: bool = True, rate: int = 150, volume: float = 0.9):
        self.enabled = enabled and TTS_AVAILABLE
        self.message_queue = queue.Queue()
        self.last_speak_time = 0
        self.min_speak_interval = 1.5

        if self.enabled:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)

            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

    def _worker(self):
        """Background worker to process voice queue"""
        while True:
            try:
                message, is_urgent = self.message_queue.get(timeout=0.1)
                current_time = time.time()

                if is_urgent or current_time - self.last_speak_time >= self.min_speak_interval:
                    self.engine.say(message)
                    self.engine.runAndWait()
                    self.last_speak_time = current_time

            except queue.Empty:
                continue
            except Exception:
                pass

    def speak(self, message: str, urgent: bool = False):
        """Queue a message for speaking"""
        if not self.enabled or not message:
            return

        if urgent:
            # Clear queue for urgent messages
            try:
                while not self.message_queue.empty():
                    try:
                        self.message_queue.get_nowait()
                    except queue.Empty:
                        break
            except:
                pass

        self.message_queue.put((message, urgent))


# ============================================================================
# Message Manager
# ============================================================================

class MessageManager:
    """Manages quick guidance and persistent warnings separately"""

    def __init__(self, voice_manager: Optional[VoiceManager] = None):
        self.quick_messages: List[Message] = []
        self.warning_messages: List[Message] = []
        self.voice_manager = voice_manager
        self.last_message_by_category: Dict[str, Message] = {}

    def add_quick_guidance(self, text: str, category: str, duration: float = 2.0):
        """Add a quick guidance message (direction, etc.)"""
        current_time = time.time()

        # Check if duplicate within short time
        if category in self.last_message_by_category:
            last = self.last_message_by_category[category]
            if last.text == text and current_time - last.timestamp < 1.5:
                return

        msg = Message(
            text=text,
            msg_type=MessageType.QUICK_GUIDANCE,
            timestamp=current_time,
            category=category,
            expires_at=current_time + duration,
            priority=1
        )

        self.quick_messages.append(msg)
        self.last_message_by_category[category] = msg

        if self.voice_manager:
            self.voice_manager.speak(text, urgent=False)

    def add_warning(self, text: str, category: str, urgent: bool = False):
        """Add a persistent warning message"""
        current_time = time.time()

        # Check if duplicate
        if category in self.last_message_by_category:
            last = self.last_message_by_category[category]
            if last.text == text and last.msg_type == MessageType.PERSISTENT_WARNING:
                return

        msg = Message(
            text=text,
            msg_type=MessageType.PERSISTENT_WARNING,
            timestamp=current_time,
            category=category,
            expires_at=float('inf'),  # Persistent until cleared
            priority=3 if urgent else 2
        )

        self.warning_messages.append(msg)
        self.last_message_by_category[category] = msg

        if self.voice_manager:
            self.voice_manager.speak(text, urgent=urgent)

    def clear_category(self, category: str):
        """Clear all messages in a category"""
        self.quick_messages = [m for m in self.quick_messages if m.category != category]
        self.warning_messages = [m for m in self.warning_messages if m.category != category]
        if category in self.last_message_by_category:
            del self.last_message_by_category[category]

    def cleanup_expired(self):
        """Remove expired quick messages"""
        current_time = time.time()
        self.quick_messages = [m for m in self.quick_messages if m.expires_at > current_time]

    def get_display_messages(self, max_quick: int = 3, max_warnings: int = 3) -> Tuple[List[Message], List[Message]]:
        """Get messages for display: (warnings, quick_guidance)"""
        self.cleanup_expired()

        # Sort warnings by priority and time
        warnings = sorted(self.warning_messages, key=lambda m: (-m.priority, -m.timestamp))[:max_warnings]

        # Sort quick messages by time (newest first)
        quick = sorted(self.quick_messages, key=lambda m: -m.timestamp)[:max_quick]

        return warnings, quick


# ============================================================================
# Workflow State Manager
# ============================================================================

class WorkflowManager:
    """Manages workflow state transitions and logic"""

    def __init__(self, config: NavigationConfig, message_manager: MessageManager):
        self.config = config
        self.message_manager = message_manager

        # Current states
        self.workflow_state = WorkflowState.IDLE
        self.obstacle_state = ObstacleState.CLEAR
        self.traffic_signal = TrafficSignalState.UNKNOWN
        self.nav_direction = NavDirection.STRAIGHT

        # Detection flags
        self.blind_path_detected = False
        self.crossing_detected = False
        self.crossing_at_position = False

        # Stability tracking for blind path (prevent flicker)
        self.blind_path_lost_time = 0.0
        self.blind_path_found_time = 0.0
        self.blind_path_stable_lost = False
        self.blind_path_stable_found = False
        self.blind_path_lost_warned = False

        # Stability tracking for crossing
        self.crossing_at_start_time = 0.0
        self.crossing_at_stable = False
        self.crossing_prompt_shown = False

        # Obstacle tracking
        self.blocking_obstacle_label = ""
        self.blocking_candidate_label = ""
        self.blocking_candidate_start = 0.0
        self.blocking_last_seen = 0.0

        # Traffic signal tracking (use consecutive detection instead of time-based)
        self.walk_consecutive_count = 0  # Count consecutive WALK detections
        self.dont_walk_consecutive_count = 0  # Count consecutive DON'T WALK detections
        self.walk_confirmed = False
        self.dont_walk_confirmed = False

        # Navigation smoothing
        self.nav_direction_frames = 0
        self.nav_last_time = 0.0

        # User interaction flag
        self.user_initiated_crossing = False

    def update_detections(self, blind_detected: bool, crossing_detected: bool,
                          crossing_at: bool, traffic_signal: TrafficSignalState):
        """Update detection flags"""
        self.blind_path_detected = blind_detected
        self.crossing_detected = crossing_detected
        self.crossing_at_position = crossing_at
        self.traffic_signal = traffic_signal

    def update_workflow_state(self, frame_time: float):
        """Main workflow state machine logic with stability checks"""
        old_state = self.workflow_state

        # === Stability check for blind path detection ===
        STABLE_DURATION = 2.0  # 2 seconds stability required

        if self.blind_path_detected:
            # Blind path is detected
            if not self.blind_path_stable_found:
                if self.blind_path_found_time == 0:
                    self.blind_path_found_time = frame_time

                duration = frame_time - self.blind_path_found_time
                if duration >= STABLE_DURATION:
                    self.blind_path_stable_found = True
                    self.blind_path_stable_lost = False
                    self.blind_path_lost_time = 0
                    self.blind_path_lost_warned = False

            # Clear any lost path warnings
            if self.blind_path_lost_warned:
                self.message_manager.clear_category("path_lost")
                self.blind_path_lost_warned = False
        else:
            # Blind path is NOT detected
            if not self.blind_path_stable_lost:
                if self.blind_path_lost_time == 0:
                    self.blind_path_lost_time = frame_time

                duration = frame_time - self.blind_path_lost_time
                if duration >= STABLE_DURATION:
                    self.blind_path_stable_lost = True
                    self.blind_path_stable_found = False
                    self.blind_path_found_time = 0

        # === Check crossing at position stability ===
        if self.crossing_at_position:
            if not self.crossing_at_stable:
                if self.crossing_at_start_time == 0:
                    self.crossing_at_start_time = frame_time

                duration = frame_time - self.crossing_at_start_time
                if duration >= 1.0:  # 1 second stability for crossing
                    self.crossing_at_stable = True
        else:
            # Only reset if we're not in active crossing mode
            if self.workflow_state != WorkflowState.ROAD_CROSSING_ACTIVE:
                self.crossing_at_stable = False
                self.crossing_at_start_time = 0
                self.crossing_prompt_shown = False

        # === State Machine Logic ===

        # State: IDLE
        if self.workflow_state == WorkflowState.IDLE:
            # User wants to cross at road crossing (HIGHEST PRIORITY)
            if self.user_initiated_crossing and self.crossing_at_stable:
                self.workflow_state = WorkflowState.ROAD_CROSSING_ACTIVE
                self.message_manager.clear_category("crossing_prompt")
                self.message_manager.add_quick_guidance(
                    "Road crossing mode activated",
                    "workflow_state",
                    duration=2.0
                )
            # Found blind path
            elif self.blind_path_stable_found:
                self.workflow_state = WorkflowState.BLIND_PATH_ACTIVE
                self.message_manager.add_quick_guidance(
                    "Blind path detected - following guide",
                    "workflow_state",
                    duration=3.0
                )
            # Prompt user about crossing opportunity
            elif self.crossing_at_stable and not self.crossing_prompt_shown:
                self.message_manager.add_warning(
                    "AT ROAD CROSSING - Press C to cross",
                    "crossing_prompt",
                    urgent=False
                )
                self.crossing_prompt_shown = True

        # State: BLIND_PATH_ACTIVE
        elif self.workflow_state == WorkflowState.BLIND_PATH_ACTIVE:
            # Lost blind path with stability
            if self.blind_path_stable_lost and not self.blind_path_lost_warned:
                self.workflow_state = WorkflowState.IDLE
                self.message_manager.clear_category("blind_path_nav")
                self.message_manager.add_warning(
                    "LOST BLIND PATH - PLEASE STOP",
                    "path_lost",
                    urgent=True
                )
                self.blind_path_lost_warned = True

            # Reached crossing and user wants to cross
            elif self.crossing_at_stable and self.user_initiated_crossing:
                self.workflow_state = WorkflowState.ROAD_CROSSING_ACTIVE
                self.message_manager.clear_category("blind_path_nav")
                self.message_manager.clear_category("crossing_prompt")
                self.message_manager.add_quick_guidance(
                    "Entering road crossing mode",
                    "workflow_state",
                    duration=2.0
                )

            # Prompt about crossing opportunity
            elif self.crossing_at_stable and not self.crossing_prompt_shown:
                self.message_manager.add_warning(
                    "📍 AT ROAD CROSSING - Press C to cross",
                    "crossing_prompt",
                    urgent=False
                )
                self.crossing_prompt_shown = True

        # State: ROAD_CROSSING_ACTIVE
        elif self.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
            # Stay in crossing mode until user manually exits
            # No automatic exit based on crossing detection
            pass

    def initiate_crossing(self):
        """Toggle road crossing workflow (press C to start/stop)"""
        # If already in crossing mode, exit it
        if self.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
            self.workflow_state = WorkflowState.IDLE
            self.user_initiated_crossing = False
            self.crossing_prompt_shown = False
            self.message_manager.clear_category("crossing_nav")
            self.message_manager.clear_category("traffic_signal")
            self.message_manager.clear_category("crossing_prompt")
            self.message_manager.add_quick_guidance(
                "Exited crossing mode",
                "workflow_state",
                duration=2.0
            )
            return

        # Otherwise, try to enter crossing mode
        if self.crossing_at_stable:
            self.user_initiated_crossing = True
            self.message_manager.clear_category("crossing_prompt")

    def update_obstacle_state(self, obstacles: List[ObstacleInfo],
                              blind_mask, crossing_mask, frame_time: float,
                              frame_width: int, frame_height: int):
        """Update obstacle detection and warnings (context-aware)"""

        # In ROAD_CROSSING mode, only warn about imminent collisions, not path blocking
        if self.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
            # During crossing, only detect very close obstacles (imminent danger)
            blocking_candidate = self._find_imminent_obstacle(
                obstacles, frame_width, frame_height
            )
        else:
            # In BLIND_PATH mode, detect obstacles blocking the path
            blocking_candidate = self._find_blocking_obstacle(
                obstacles, blind_mask, frame_width, frame_height
            )

        if blocking_candidate:
            candidate_label = blocking_candidate.label

            if candidate_label != self.blocking_candidate_label:
                self.blocking_candidate_label = candidate_label
                self.blocking_candidate_start = frame_time

            self.blocking_last_seen = frame_time

            # Check if stable enough to warn
            if self.obstacle_state == ObstacleState.CLEAR:
                duration = frame_time - self.blocking_candidate_start
                # Shorter threshold for imminent danger during crossing
                if self.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
                    threshold = 0.5  # 0.5s for crossing
                else:
                    threshold = (self.config.blocking_onpath_duration if blocking_candidate.on_path
                                 else self.config.blocking_front_duration)

                if duration >= threshold:
                    self.obstacle_state = ObstacleState.DETECTED
                    self.blocking_obstacle_label = candidate_label

                    # Different message for crossing vs blind path
                    if self.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
                        msg = f"{candidate_label.upper()} TOO CLOSE!"
                    else:
                        msg = f"{candidate_label.upper()} BLOCKING PATH!"

                    self.message_manager.add_warning(msg, "obstacle_warning", urgent=True)
        else:
            if self.obstacle_state == ObstacleState.DETECTED:
                if frame_time - self.blocking_last_seen > 3.0:
                    self.obstacle_state = ObstacleState.CLEAR
                    self.blocking_obstacle_label = ""
                    self.blocking_candidate_label = ""
                    self.message_manager.clear_category("obstacle_warning")
                    self.message_manager.add_quick_guidance(
                        "Path is clear",
                        "obstacle_status",
                        duration=2.0
                    )

    def _find_imminent_obstacle(self, obstacles: List[ObstacleInfo],
                                frame_width: int, frame_height: int) -> Optional[ObstacleInfo]:
        """Find obstacles that are imminently dangerous (very close) during crossing"""
        ego_y_min = int(frame_height * 0.85)  # Very close to bottom (stricter)
        total_area = frame_width * frame_height

        # Define critical zone (center area, very close)
        critical_x_min = int(frame_width * 0.25)
        critical_x_max = int(frame_width * 0.75)
        critical_y_min = int(frame_height * 0.60)  # Much closer than before
        critical_y_max = int(frame_height * 0.90)

        blocking = None
        min_distance = float('inf')

        for obs in obstacles:
            x1, y1, x2, y2 = obs.bbox

            # Skip if too far (below ego line)
            if y2 > ego_y_min:
                continue

            box_area = (x2 - x1) * (y2 - y1)
            if box_area / total_area < self.config.obstacle_area_min_ratio:
                continue

            obs.distance_score = (frame_height - y2) / frame_height

            # Check if in critical zone
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            in_critical_zone = (
                    critical_x_min <= cx <= critical_x_max and
                    critical_y_min <= cy <= critical_y_max
            )

            # Only warn about very close obstacles in critical zone
            if in_critical_zone and obs.distance_score < 0.3:  # Very close
                if obs.distance_score < min_distance:
                    min_distance = obs.distance_score
                    blocking = obs

        return blocking

    def _find_blocking_obstacle(self, obstacles: List[ObstacleInfo],
                                blind_mask, frame_width: int,
                                frame_height: int) -> Optional[ObstacleInfo]:
        """Find blocking obstacle if any"""
        ego_y_min = int(frame_height * self.config.ego_zone_height)
        total_area = frame_width * frame_height

        blocking = None
        min_distance = float('inf')

        for obs in obstacles:
            x1, y1, x2, y2 = obs.bbox

            if y2 > ego_y_min:
                continue

            box_area = (x2 - x1) * (y2 - y1)
            if box_area / total_area < self.config.obstacle_area_min_ratio:
                continue

            obs.distance_score = (frame_height - y2) / frame_height

            if obs.on_path and obs.distance_score < min_distance:
                min_distance = obs.distance_score
                blocking = obs
            elif obs.in_front and blocking is None and obs.distance_score < min_distance:
                min_distance = obs.distance_score
                blocking = obs

        return blocking

    def update_traffic_signal_state(self, frame_time: float):
        """Update traffic signal state with consecutive detection (3 times)

        Important: Once green light is confirmed, red light detection (countdown)
        will NOT override it, as countdown means you can still cross.
        """
        if self.workflow_state != WorkflowState.ROAD_CROSSING_ACTIVE:
            self.walk_consecutive_count = 0
            self.dont_walk_consecutive_count = 0
            self.walk_confirmed = False
            self.dont_walk_confirmed = False
            return

        CONSECUTIVE_REQUIRED = 3  # Need 3 consecutive detections

        # Don't Walk signal (RED LIGHT)
        if self.traffic_signal == TrafficSignalState.DONT_WALK:
            # IMPORTANT: If green light was already confirmed, ignore red light
            # (Red light detection includes countdown, which is still safe to cross)
            if self.walk_confirmed:
                # Keep green light status, ignore red detection
                return

            self.dont_walk_consecutive_count += 1
            self.walk_consecutive_count = 0  # Reset walk counter

            if self.dont_walk_consecutive_count >= CONSECUTIVE_REQUIRED:
                if not self.dont_walk_confirmed:
                    self.dont_walk_confirmed = True

                    self.message_manager.clear_category("traffic_signal")
                    self.message_manager.add_warning(
                        "RED LIGHT - PLEASE WAIT",
                        "traffic_signal",
                        urgent=True
                    )

        # Walk signal (GREEN LIGHT)
        elif self.traffic_signal == TrafficSignalState.WALK:
            self.walk_consecutive_count += 1
            self.dont_walk_consecutive_count = 0  # Reset don't walk counter

            if self.walk_consecutive_count >= CONSECUTIVE_REQUIRED:
                if not self.walk_confirmed:
                    self.walk_confirmed = True
                    self.dont_walk_confirmed = False

                    self.message_manager.clear_category("traffic_signal")
                    self.message_manager.add_warning(
                        "GREEN LIGHT - YOU MAY CROSS",
                        "traffic_signal",
                        urgent=False
                    )

        # No signal or other signal
        else:
            # Don't reset counters immediately, allow for brief detection gaps
            pass

    def update_navigation_guidance(self, nav_features, frame_time: float):
        """Update navigation guidance based on active workflow"""
        if self.obstacle_state == ObstacleState.DETECTED:
            return

        # Blind path navigation
        if self.workflow_state == WorkflowState.BLIND_PATH_ACTIVE:
            if nav_features is None:
                return

            direction, text = self._compute_blind_path_direction(nav_features)

            if direction == self.nav_direction:
                self.nav_direction_frames += 1
            else:
                self.nav_direction = direction
                self.nav_direction_frames = 1

            if self.nav_direction_frames >= self.config.nav_smoothing_frames:
                if frame_time - self.nav_last_time >= self.config.nav_guidance_interval:
                    self.nav_last_time = frame_time
                    self.message_manager.add_quick_guidance(
                        text,
                        "blind_path_nav",
                        duration=self.config.guidance_message_duration
                    )

        # Road crossing navigation - ONLY when green light is confirmed
        elif self.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
            # Don't provide navigation guidance if still waiting for green light
            if not self.walk_confirmed:
                return

            if nav_features is None:
                return

            direction, text = self._compute_crossing_direction(nav_features)

            if direction == self.nav_direction:
                self.nav_direction_frames += 1
            else:
                self.nav_direction = direction
                self.nav_direction_frames = 1

            if self.nav_direction_frames >= self.config.nav_smoothing_frames:
                if frame_time - self.nav_last_time >= self.config.nav_guidance_interval:
                    self.nav_last_time = frame_time
                    self.message_manager.add_quick_guidance(
                        text,
                        "crossing_nav",
                        duration=self.config.guidance_message_duration
                    )

    def _compute_blind_path_direction(self, nav_features) -> Tuple[NavDirection, str]:
        """Compute navigation direction for blind path"""
        angle_rad = nav_features["angle_rad"]
        offset_ratio = nav_features["center_offset_ratio"]
        offset_px = nav_features["center_offset_pixels"]

        orientation_thr = np.deg2rad(self.config.orientation_threshold_deg)
        offset_thr = self.config.offset_threshold_ratio

        if offset_ratio > offset_thr:
            if offset_px > 0:
                return NavDirection.SHIFT_RIGHT, "Shift right"
            else:
                return NavDirection.SHIFT_LEFT, "Shift left"

        if angle_rad > orientation_thr:
            return NavDirection.TURN_LEFT, "Turn left"
        elif angle_rad < -orientation_thr:
            return NavDirection.TURN_RIGHT, "Turn right"
        else:
            return NavDirection.STRAIGHT, "Continue straight"

    def _compute_crossing_direction(self, nav_features) -> Tuple[NavDirection, str]:
        """Compute navigation direction for crossing (perpendicular to crosswalk stripes)

        The nav_features["angle_rad"] represents the crosswalk stripe direction.
        We want to guide the user to walk PERPENDICULAR to the stripes.
        """
        # Crosswalk stripe angle
        stripe_angle = nav_features["angle_rad"]

        # We want to walk perpendicular to stripes
        # Since the centerline computed is along the stripes, we need to adjust
        # For crossing, we check lateral offset (parallel to stripes)

        offset_ratio = nav_features["center_offset_ratio"]
        offset_px = nav_features["center_offset_pixels"]

        offset_thr = self.config.offset_threshold_ratio
        orientation_thr = np.deg2rad(self.config.orientation_threshold_deg)

        # Check if user is drifting left/right while crossing
        if offset_ratio > offset_thr:
            if offset_px > 0:
                return NavDirection.SHIFT_RIGHT, "Shift right"
            else:
                return NavDirection.SHIFT_LEFT, "Shift left"

        # Check orientation relative to perpendicular direction
        # If stripe angle is large, user should adjust their walking direction
        # to stay perpendicular to stripes
        if abs(stripe_angle) > orientation_thr:
            if stripe_angle > 0:
                return NavDirection.TURN_LEFT, "Adjust left"
            else:
                return NavDirection.TURN_RIGHT, "Adjust right"
        else:
            return NavDirection.STRAIGHT, "Cross straight"


# ============================================================================
# Vision Processing Functions
# ============================================================================

def get_blind_and_crossing_masks(seg_results):
    """Extract blind path and crossing masks from segmentation"""
    h, w = seg_results.orig_img.shape[:2]
    blind_mask = np.zeros((h, w), dtype=np.uint8)
    crossing_mask = np.zeros((h, w), dtype=np.uint8)

    if seg_results.masks is None or seg_results.boxes is None:
        return blind_mask, crossing_mask

    small_masks = seg_results.masks.data.cpu().numpy()
    if small_masks.ndim != 3 or small_masks.shape[0] == 0:
        return blind_mask, crossing_mask

    classes = seg_results.boxes.cls.cpu().numpy().astype(int)
    Hm, Wm = small_masks.shape[1], small_masks.shape[2]

    blind_small = np.zeros((Hm, Wm), dtype=bool)
    crossing_small = np.zeros((Hm, Wm), dtype=bool)

    for i, cls in enumerate(classes):
        if cls == 1:  # Blind path
            blind_small |= (small_masks[i] > 0.5)
        elif cls == 0:  # Crosswalk
            crossing_small |= (small_masks[i] > 0.5)

    if blind_small.any():
        blind_resized = cv2.resize(
            blind_small.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        blind_mask[blind_resized > 0] = 255

    if crossing_small.any():
        crossing_resized = cv2.resize(
            crossing_small.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        crossing_mask[crossing_resized > 0] = 255

    return blind_mask, crossing_mask


def get_obstacles(det_results, blind_mask, frame_width, frame_height) -> List[ObstacleInfo]:
    """Extract and analyze obstacles from detection results"""
    OBSTACLE_CLASSES = {
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "bench", "trash can", "chair", "potted plant"
    }

    obstacles = []
    if det_results.boxes is None:
        return obstacles

    names = det_results.names
    boxes = det_results.boxes.xyxy.cpu().numpy()
    classes = det_results.boxes.cls.cpu().numpy().astype(int)

    roi_y1 = int(frame_height * 0.55)
    front_x_min = int(frame_width * 0.30)
    front_x_max = int(frame_width * 0.70)
    front_y_min = int(frame_height * 0.35)
    front_y_max = int(frame_height * 0.80)

    for box, cls in zip(boxes, classes):
        label = names.get(int(cls), str(int(cls)))
        if label not in OBSTACLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.astype(int)

        # Check if on path
        on_path = False
        if blind_mask is not None:
            yy1 = max(y1, roi_y1)
            yy2 = min(y2, frame_height)
            if yy2 > yy1:
                sub_mask = blind_mask[yy1:yy2, x1:x2]
                overlap_pixels = int(np.count_nonzero(sub_mask))
                box_area = (x2 - x1) * (y2 - y1)
                overlap_ratio = overlap_pixels / max(1, box_area)
                on_path = overlap_ratio > 0.05

        # Check if in front
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        in_front = (front_x_min <= cx <= front_x_max and
                    front_y_min <= cy <= front_y_max)

        obs = ObstacleInfo(
            label=label,
            bbox=(x1, y1, x2, y2),
            distance_score=0.0,
            on_path=on_path,
            in_front=in_front
        )
        obstacles.append(obs)

    return obstacles


def compute_navigation_features(mask, frame_width, frame_height):
    """Compute navigation centerline and features from mask (blind path or crossing)"""
    if mask is None or not np.any(mask):
        return None

    h, w = mask.shape[:2]

    bottom_y_min = int(h * 0.70)
    mid_y_min = int(h * 0.45)

    bottom_xs, bottom_ys = [], []
    mid_xs, mid_ys = [], []

    for y in range(h - 1, int(h * 0.35), -3):
        row = mask[y, :]
        x_pixels = np.where(row > 0)[0]
        if x_pixels.size < 10:
            continue

        x_min, x_max = x_pixels[0], x_pixels[-1]
        center_x = 0.5 * (x_min + x_max)

        if y >= bottom_y_min:
            bottom_xs.append(center_x)
            bottom_ys.append(y)
        elif y >= mid_y_min:
            mid_xs.append(center_x)
            mid_ys.append(y)

    if len(bottom_xs) < 3 or len(mid_xs) < 3:
        return None

    y1 = float(np.mean(bottom_ys))
    x1 = float(np.mean(bottom_xs))
    y2 = float(np.mean(mid_ys))
    x2 = float(np.mean(mid_xs))

    if abs(y2 - y1) < 1.0:
        return None

    m = (x2 - x1) / (y2 - y1)
    b = x1 - m * y1

    angle_rad = np.arctan(m)
    angle_deg = float(np.degrees(angle_rad))

    y_target = h * 0.6
    x_target = m * y_target + b

    y_bottom = h - 1
    y_top = int(h * 0.3)
    x_bottom = m * y_bottom + b
    x_top = m * y_top + b

    p_bottom = (int(np.clip(x_bottom, 0, w - 1)), int(y_bottom))
    p_top = (int(np.clip(x_top, 0, w - 1)), int(y_top))

    center_offset_pixels = x_target - frame_width / 2.0
    center_offset_ratio = abs(center_offset_pixels) / frame_width

    return {
        "slope": float(m),
        "intercept": float(b),
        "angle_rad": float(angle_rad),
        "angle_deg": angle_deg,
        "centerline_points": np.array([p_bottom, p_top], dtype=int),
        "target_point": (int(np.clip(x_target, 0, w - 1)), int(y_target)),
        "center_offset_pixels": float(center_offset_pixels),
        "center_offset_ratio": float(center_offset_ratio),
    }


def parse_traffic_signal(tl_results) -> TrafficSignalState:
    """Parse traffic light detection results
    Model labels: {0: 'Crosswalk', 1: "Don't Walk", 2: 'Traffic Signal', 3: 'Walk'}
    """
    if tl_results is None or tl_results.boxes is None:
        return TrafficSignalState.NO_SIGNAL

    if tl_results.boxes.cls is None or len(tl_results.boxes.cls) == 0:
        return TrafficSignalState.NO_SIGNAL

    classes = tl_results.boxes.cls.cpu().numpy().astype(int)
    scores = tl_results.boxes.conf.cpu().numpy()

    # Look for Walk or Don't Walk signals
    walk_indices = np.where(classes == 3)[0]
    dont_walk_indices = np.where(classes == 1)[0]

    if len(walk_indices) > 0:
        return TrafficSignalState.WALK
    elif len(dont_walk_indices) > 0:
        return TrafficSignalState.DONT_WALK
    else:
        return TrafficSignalState.NO_SIGNAL


# ============================================================================
# Enhanced Visualization Functions
# ============================================================================

def draw_text_with_bg(img, text, pos, font_scale=1.0,
                      text_color=(255, 255, 255), bg_color=(0, 0, 0),
                      thickness=2, padding=10):
    """Draw text with background"""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    cv2.rectangle(img,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_overlay(frame, blind_mask, crossing_mask, obstacles, nav_features=None,
                 workflow_state=WorkflowState.IDLE, tl_results=None):
    """Draw visual overlays with workflow-aware rendering and traffic signal detection"""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Crossing overlay (blue) - always show when detected
    if crossing_mask is not None and np.any(crossing_mask):
        overlay = vis.copy()
        overlay[crossing_mask > 0] = (255, 120, 0)
        vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    # Blind path overlay (green) - only show when blind path workflow is active
    if (blind_mask is not None and np.any(blind_mask) and
            workflow_state == WorkflowState.BLIND_PATH_ACTIVE):
        overlay = vis.copy()
        overlay[blind_mask > 0] = (0, 255, 100)
        vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    # Draw traffic signal detections (Walk/Don't Walk)
    if tl_results is not None and tl_results.boxes is not None:
        boxes = tl_results.boxes.xyxy.cpu().numpy()
        classes = tl_results.boxes.cls.cpu().numpy().astype(int)
        scores = tl_results.boxes.conf.cpu().numpy()
        names = tl_results.names

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = box.astype(int)
            label = names.get(int(cls), str(int(cls)))

            # Only draw Walk (3) and Don't Walk (1) signals
            if cls == 3:  # Walk
                color = (0, 255, 0)  # Green
                text = "WALK"
                thickness = 5
            elif cls == 1:  # Don't Walk
                color = (0, 0, 255)  # Red
                text = "DON'T WALK"
                thickness = 5
            else:
                continue  # Skip other classes

            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Draw label with background
            label_text = f"{text} {score:.2f}"
            draw_text_with_bg(vis, label_text, (x1 + 5, y1 - 15),
                              font_scale=1.0, text_color=(255, 255, 255),
                              bg_color=color, thickness=3, padding=10)

    # Obstacles
    for obs in obstacles:
        x1, y1, x2, y2 = obs.bbox
        color = (0, 0, 255) if obs.on_path else (0, 165, 255)
        thickness = 4 if obs.on_path else 3

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label_text = obs.label.upper()
        draw_text_with_bg(vis, label_text, (x1 + 5, y1 - 10),
                          font_scale=0.8, text_color=(255, 255, 255),
                          bg_color=color, thickness=2, padding=8)

    # Navigation centerline
    if nav_features is not None:
        pts = nav_features["centerline_points"]
        if len(pts) >= 2:
            # Different colors for different workflows
            if workflow_state == WorkflowState.BLIND_PATH_ACTIVE:
                line_color = (0, 255, 255)  # Yellow for blind path
            elif workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
                line_color = (255, 128, 0)  # Orange for crossing
            else:
                line_color = (128, 128, 128)  # Gray for inactive

            cv2.line(vis, tuple(pts[0]), tuple(pts[1]), line_color, 4)

        tx, ty = nav_features["target_point"]
        cv2.circle(vis, (tx, ty), 12, (0, 0, 255), -1)
        cv2.circle(vis, (tx, ty), 12, (255, 255, 255), 2)

        # Direction arrow
        cx, cy = w // 2, h - 80
        arrow_len = 90
        offset_ratio = nav_features["center_offset_ratio"]
        offset_px = nav_features["center_offset_pixels"]

        if offset_ratio > 0.05:
            end = (cx + arrow_len, cy) if offset_px > 0 else (cx - arrow_len, cy)
        else:
            end = (cx, cy - arrow_len)

        cv2.arrowedLine(vis, (cx, cy), end, (0, 255, 255), 5, tipLength=0.4)

    return vis


def compose_ui_frame(video_frame, workflow_manager: WorkflowManager,
                     message_manager: MessageManager, config: NavigationConfig):
    """Compose modern enhanced UI with workflow states and message panels"""
    h, w = video_frame.shape[:2]
    panel_width = config.panel_width
    panel = np.full((h, panel_width, 3), 40, dtype=np.uint8)

    # ===== Header =====
    header_h = 120
    cv2.rectangle(panel, (0, 0), (panel_width, header_h), (45, 45, 45), -1)

    # BLIND NAVIGATION in white
    draw_text_with_bg(panel, "BLIND NAVIGATION", (25, 50),
                      font_scale=1.4, text_color=(255, 255, 255),
                      bg_color=(45, 45, 45), thickness=3, padding=0)
    # ASSISTANT in gold/yellow
    draw_text_with_bg(panel, "ASSISTANT", (25, 90),
                      font_scale=1.2, text_color=(50, 200, 255),
                      bg_color=(45, 45, 45), thickness=2, padding=0)

    y_pos = header_h + 25

    # ===== Modern Workflow State Cards with Indicators =====
    card_margin = 20
    card_w = panel_width - 2 * card_margin
    card_h = 90
    indicator_radius = 18
    corner_radius = 8

    wf = workflow_manager.workflow_state

    # Helper function to draw rounded rectangle
    def draw_rounded_card(img, pt1, pt2, color, indicator_color, active=True):
        x1, y1 = pt1
        x2, y2 = pt2
        # Main rectangle
        cv2.rectangle(img, (x1 + corner_radius, y1), (x2 - corner_radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + corner_radius), (x2, y2 - corner_radius), color, -1)
        # Corners
        cv2.circle(img, (x1 + corner_radius, y1 + corner_radius), corner_radius, color, -1)
        cv2.circle(img, (x2 - corner_radius, y1 + corner_radius), corner_radius, color, -1)
        cv2.circle(img, (x1 + corner_radius, y2 - corner_radius), corner_radius, color, -1)
        cv2.circle(img, (x2 - corner_radius, y2 - corner_radius), corner_radius, color, -1)
        # Border
        border_color = (200, 200, 200) if active else (100, 100, 100)
        cv2.rectangle(img, pt1, pt2, border_color, 2)
        # Status indicator circle
        indicator_x = x1 + 45
        indicator_y = (y1 + y2) // 2
        cv2.circle(img, (indicator_x, indicator_y), indicator_radius, indicator_color, -1)
        cv2.circle(img, (indicator_x, indicator_y), indicator_radius, (255, 255, 255), 2)

    # Card 1: Blind Path Workflow
    if wf == WorkflowState.BLIND_PATH_ACTIVE:
        card_bg = (0, 130, 0)
        indicator_color = (255, 255, 255)
        title = "BLIND PATH"
        subtitle = "Active Navigation"
        active = True
    elif wf == WorkflowState.BLIND_PATH_PAUSED:
        card_bg = (80, 80, 0)
        indicator_color = (200, 200, 0)
        title = "BLIND PATH"
        subtitle = "Paused"
        active = True
    else:
        card_bg = (70, 70, 70)
        indicator_color = (120, 120, 120)
        title = "BLIND PATH"
        subtitle = "Inactive"
        active = False

    draw_rounded_card(panel, (card_margin, y_pos),
                      (card_margin + card_w, y_pos + card_h),
                      card_bg, indicator_color, active)

    # Title and subtitle
    text_x = card_margin + 80
    draw_text_with_bg(panel, title, (text_x, y_pos + 38),
                      font_scale=1.1, text_color=(255, 255, 255),
                      bg_color=card_bg, thickness=2, padding=0)
    draw_text_with_bg(panel, subtitle, (text_x, y_pos + 68),
                      font_scale=0.7, text_color=(180, 180, 180),
                      bg_color=card_bg, thickness=1, padding=0)

    y_pos += card_h + 15

    # Card 2: Road Crossing Workflow
    if wf == WorkflowState.ROAD_CROSSING_ACTIVE:
        card_bg = (0, 130, 0)
        indicator_color = (255, 255, 255)
        title = "ROAD CROSSING"
        subtitle = "Active Navigation"
        active = True
    elif wf == WorkflowState.ROAD_CROSSING_PAUSED:
        card_bg = (80, 80, 0)
        indicator_color = (200, 200, 0)
        title = "ROAD CROSSING"
        subtitle = "Paused"
        active = True
    else:
        card_bg = (70, 70, 70)
        indicator_color = (120, 120, 120)
        title = "ROAD CROSSING"
        subtitle = "Inactive"
        active = False

    draw_rounded_card(panel, (card_margin, y_pos),
                      (card_margin + card_w, y_pos + card_h),
                      card_bg, indicator_color, active)

    draw_text_with_bg(panel, title, (text_x, y_pos + 38),
                      font_scale=1.1, text_color=(255, 255, 255),
                      bg_color=card_bg, thickness=2, padding=0)
    draw_text_with_bg(panel, subtitle, (text_x, y_pos + 68),
                      font_scale=0.7, text_color=(180, 180, 180),
                      bg_color=card_bg, thickness=1, padding=0)

    y_pos += card_h + 15

    # Card 3: Obstacle Status
    obs_state = workflow_manager.obstacle_state
    if obs_state == ObstacleState.DETECTED:
        card_bg = (0, 0, 160)
        indicator_color = (255, 255, 255)
        title = "OBSTACLE"
        obs_label = workflow_manager.blocking_obstacle_label
        subtitle = obs_label.title() if obs_label else "Detected"
        active = True
    else:
        card_bg = (0, 130, 0)
        indicator_color = (255, 255, 255)
        title = "OBSTACLE"
        subtitle = "Clear"
        active = True

    draw_rounded_card(panel, (card_margin, y_pos),
                      (card_margin + card_w, y_pos + card_h),
                      card_bg, indicator_color, active)

    draw_text_with_bg(panel, title, (text_x, y_pos + 38),
                      font_scale=1.1, text_color=(255, 255, 255),
                      bg_color=card_bg, thickness=2, padding=0)
    draw_text_with_bg(panel, subtitle, (text_x, y_pos + 68),
                      font_scale=0.7, text_color=(180, 180, 180),
                      bg_color=card_bg, thickness=1, padding=0)

    y_pos += card_h + 45

    # ===== Message Section =====
    draw_text_with_bg(panel, "MESSAGES", (card_margin, y_pos),
                      font_scale=1.0, text_color=(200, 200, 200),
                      bg_color=(40, 40, 40), thickness=2, padding=0)
    y_pos += 45

    # Get messages
    warnings, quick = message_manager.get_display_messages(
        max_quick=config.max_quick_messages,
        max_warnings=config.max_warning_messages
    )

    # Display warnings first (persistent)
    for msg in warnings:
        if y_pos + 75 > h - 20:
            break

        if msg.priority >= 3:
            bubble_color = (30, 30, 180)
            icon_color = (100, 100, 255)
        else:
            bubble_color = (140, 70, 0)
            icon_color = (255, 180, 0)

        bubble_h = 70
        bubble_x1 = card_margin
        bubble_x2 = card_margin + card_w

        # Rounded message bubble
        cv2.rectangle(panel, (bubble_x1 + corner_radius, y_pos),
                      (bubble_x2 - corner_radius, y_pos + bubble_h), bubble_color, -1)
        cv2.rectangle(panel, (bubble_x1, y_pos + corner_radius),
                      (bubble_x2, y_pos + bubble_h - corner_radius), bubble_color, -1)
        cv2.circle(panel, (bubble_x1 + corner_radius, y_pos + corner_radius),
                   corner_radius, bubble_color, -1)
        cv2.circle(panel, (bubble_x2 - corner_radius, y_pos + corner_radius),
                   corner_radius, bubble_color, -1)
        cv2.circle(panel, (bubble_x1 + corner_radius, y_pos + bubble_h - corner_radius),
                   corner_radius, bubble_color, -1)
        cv2.circle(panel, (bubble_x2 - corner_radius, y_pos + bubble_h - corner_radius),
                   corner_radius, bubble_color, -1)

        # Border
        cv2.rectangle(panel, (bubble_x1, y_pos), (bubble_x2, y_pos + bubble_h),
                      (200, 200, 200), 2)

        # Icon circle
        icon_x = bubble_x1 + 35
        icon_y = y_pos + bubble_h // 2
        cv2.circle(panel, (icon_x, icon_y), 16, icon_color, -1)
        cv2.circle(panel, (icon_x, icon_y), 16, (255, 255, 255), 2)
        # Warning symbol "!"
        cv2.putText(panel, "!", (icon_x - 5, icon_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Message text
        text = msg.text[:45]
        draw_text_with_bg(panel, text, (bubble_x1 + 65, y_pos + bubble_h // 2 + 8),
                          font_scale=0.75, text_color=(255, 255, 255),
                          bg_color=bubble_color, thickness=2, padding=0)

        y_pos += bubble_h + 12

    # Display quick guidance messages
    for msg in quick:
        if y_pos + 65 > h - 20:
            break

        bubble_color = (60, 60, 60)
        icon_color = (255, 200, 50)
        bubble_h = 60
        bubble_x1 = card_margin
        bubble_x2 = card_margin + card_w

        # Rounded message bubble
        cv2.rectangle(panel, (bubble_x1 + corner_radius, y_pos),
                      (bubble_x2 - corner_radius, y_pos + bubble_h), bubble_color, -1)
        cv2.rectangle(panel, (bubble_x1, y_pos + corner_radius),
                      (bubble_x2, y_pos + bubble_h - corner_radius), bubble_color, -1)
        cv2.circle(panel, (bubble_x1 + corner_radius, y_pos + corner_radius),
                   corner_radius, bubble_color, -1)
        cv2.circle(panel, (bubble_x2 - corner_radius, y_pos + corner_radius),
                   corner_radius, bubble_color, -1)
        cv2.circle(panel, (bubble_x1 + corner_radius, y_pos + bubble_h - corner_radius),
                   corner_radius, bubble_color, -1)
        cv2.circle(panel, (bubble_x2 - corner_radius, y_pos + bubble_h - corner_radius),
                   corner_radius, bubble_color, -1)

        # Border
        cv2.rectangle(panel, (bubble_x1, y_pos), (bubble_x2, y_pos + bubble_h),
                      (120, 120, 120), 2)

        # Icon circle
        icon_x = bubble_x1 + 35
        icon_y = y_pos + bubble_h // 2
        cv2.circle(panel, (icon_x, icon_y), 14, icon_color, -1)
        cv2.circle(panel, (icon_x, icon_y), 14, (255, 255, 255), 2)

        # Message text
        text = msg.text[:45]
        draw_text_with_bg(panel, text, (bubble_x1 + 65, y_pos + bubble_h // 2 + 6),
                          font_scale=0.7, text_color=(220, 220, 220),
                          bg_color=bubble_color, thickness=2, padding=0)

        y_pos += bubble_h + 10

    # Combine frames
    combined = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    combined[:, :w] = video_frame
    combined[:, w:] = panel
    return combined


# ============================================================================
# Main Application
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Blind Navigation System")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--seg-weights", type=str, default="model/yolo-seg.pt")
    parser.add_argument("--det-weights", type=str, default="model/yolov8n.pt")
    parser.add_argument("--tl-weights", type=str, default="model/trafficlight_best.pt")
    parser.add_argument("--save-path", type=str, default="output.mp4")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Load config
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = NavigationConfig(**json.load(f))
        except:
            config = NavigationConfig()
    else:
        config = NavigationConfig()

    if args.no_voice:
        config.voice_enabled = False

    # Initialize managers
    voice_manager = VoiceManager(config.voice_enabled, config.voice_rate, config.voice_volume)
    message_manager = MessageManager(voice_manager)
    workflow_manager = WorkflowManager(config, message_manager)

    # Open video
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Load models
    seg_model = YOLO(args.seg_weights)
    det_model = YOLO(args.det_weights)
    tl_model = YOLO(args.tl_weights) if args.tl_weights else None

    # Video writer - include UI panel in output
    output_width = width + config.panel_width
    output_height = height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.save_path, fourcc, fps, (output_width, output_height))

    window_name = "Blind Navigation v2 - Q:Quit | SPACE:Pause | C:Toggle Crossing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_index = 0
    last_obstacles = []
    last_tl_results = None  # Store traffic light results for visualization
    paused = False
    last_ui_frame = None

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                now = time.time()

                # Segmentation
                seg_results = seg_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
                blind_mask, crossing_mask = get_blind_and_crossing_masks(seg_results)

                # Check detections
                blind_detected = np.count_nonzero(blind_mask) / (width * height) > config.blind_path_ratio_threshold
                crossing_pixels = np.count_nonzero(crossing_mask) / (width * height)
                crossing_detected = crossing_pixels > config.crossing_ratio_threshold

                # Check if at crossing (bottom portion of frame has crossing)
                bottom_cross = crossing_mask[int(height * 0.6):, :]
                bottom_ratio = np.count_nonzero(bottom_cross) / max(1, bottom_cross.size)
                crossing_at = bottom_ratio > config.crossing_bottom_ratio

                # Traffic signal detection
                traffic_signal = TrafficSignalState.NO_SIGNAL
                if tl_model and frame_index % config.traffic_light_detection_interval == 0:
                    tl_results = tl_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
                    last_tl_results = tl_results  # Save for visualization
                    traffic_signal = parse_traffic_signal(tl_results)

                # Update workflow detections
                workflow_manager.update_detections(
                    blind_detected, crossing_detected, crossing_at, traffic_signal
                )

                # Navigation features based on active workflow
                nav_features = None
                if workflow_manager.workflow_state == WorkflowState.BLIND_PATH_ACTIVE:
                    if np.any(blind_mask):
                        nav_features = compute_navigation_features(blind_mask, width, height)
                elif workflow_manager.workflow_state == WorkflowState.ROAD_CROSSING_ACTIVE:
                    if np.any(crossing_mask):
                        nav_features = compute_navigation_features(crossing_mask, width, height)

                # Obstacle detection (periodic)
                if frame_index % config.obstacle_detection_interval == 0:
                    det_results = det_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
                    last_obstacles = get_obstacles(det_results, blind_mask, width, height)

                # Update all workflow states
                workflow_manager.update_workflow_state(now)
                workflow_manager.update_obstacle_state(last_obstacles, blind_mask, crossing_mask, now, width, height)
                workflow_manager.update_traffic_signal_state(now)
                workflow_manager.update_navigation_guidance(nav_features, now)

                # Draw overlays (with traffic light detections)
                vis = draw_overlay(frame, blind_mask, crossing_mask, last_obstacles,
                                   nav_features, workflow_manager.workflow_state, last_tl_results)

                # Compose UI
                ui_frame = compose_ui_frame(vis, workflow_manager, message_manager, config)

                # Write UI frame to output video
                out.write(ui_frame)
                last_ui_frame = ui_frame
            else:
                ui_frame = last_ui_frame

            if ui_frame is not None:
                if frame_index == 1:
                    h_ui, w_ui = ui_frame.shape[:2]
                    cv2.resizeWindow(window_name, w_ui, h_ui)
                cv2.imshow(window_name, ui_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') or key == ord('p'):
                paused = not paused
            elif key == ord('c') or key == ord('C'):
                workflow_manager.initiate_crossing()

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()