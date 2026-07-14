"""
Defense COP v2.0 - Bird's Eye View (BEV) Spatial Mapping
Homography-based perspective transformation for tactical awareness.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("DefenseCOP.Spatial")


@dataclass
class BEVPoint:
    """Point in BEV coordinate system."""
    x: float  # meters
    y: float  # meters
    
    def to_pixel(self, meters_per_pixel: float) -> Tuple[int, int]:
        """Convert BEV coordinates to pixel coordinates."""
        px = int(self.x / meters_per_pixel)
        py = int(self.y / meters_per_pixel)
        return (px, py)


class BEVMapper:
    """
    Bird's Eye View mapper using homography transformation.
    Transforms pixel coordinates to real-world top-down view.
    """
    
    def __init__(
        self,
        calibration_points: List[Tuple[float, float, float, float]],
        map_width: int,
        map_height: int,
        meters_per_pixel: float,
        reference_latitude: float = 34.0522,
        reference_longitude: float = -118.2437
    ):
        """
        Initialize BEV mapper with calibration points.
        
        Args:
            calibration_points: List of (pixel_x, pixel_y, world_x, world_y) in meters
            map_width: BEV map width in pixels
            map_height: BEV map height in pixels
            meters_per_pixel: Real-world meters per pixel in BEV space
            reference_latitude: Reference GPS latitude for origin (0,0)
            reference_longitude: Reference GPS longitude for origin (0,0)
        """
        self.map_width = map_width
        self.map_height = map_height
        self.meters_per_pixel = meters_per_pixel
        self.reference_latitude = reference_latitude
        self.reference_longitude = reference_longitude
        self._calibration_points = calibration_points
        
        # Compute homography matrix
        self._homography_matrix = self._compute_homography(calibration_points)

    
    def _compute_homography(
        self,
        calibration_points: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        """
        Compute homography matrix from calibration points.
        
        Args:
            calibration_points: [(pixel_x, pixel_y, world_x, world_y), ...] where world coords are in meters
        
        Returns:
            3x3 homography matrix
        """
        source_points_list = []
        destination_points_list = []
        
        for pixel_x, pixel_y, world_x, world_y in calibration_points:
            source_points_list.append([pixel_x, pixel_y])
            # Convert world coordinates (meters) to BEV pixel coordinates
            bev_x = int(world_x / self.meters_per_pixel)
            bev_y = int(world_y / self.meters_per_pixel)
            destination_points_list.append([bev_x, bev_y])
        
        source_points = np.array(source_points_list, dtype=np.float32)
        destination_points = np.array(destination_points_list, dtype=np.float32)
        
        homography_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        return homography_matrix
    
    def pixel_to_bev(
        self,
        pixel_coords: Tuple[float, float]
    ) -> Optional[BEVPoint]:
        """
        Transform pixel coordinates to BEV world coordinates.
        
        Args:
            pixel_coords: (x, y) in pixel space
        
        Returns:
            BEVPoint in meters, or None if transformation fails
        """
        try:
            point = np.array([[pixel_coords]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self._homography_matrix)
            
            bev_pixel_x = transformed[0][0][0]
            bev_pixel_y = transformed[0][0][1]
            
            # Convert BEV pixels back to meters
            world_x = bev_pixel_x * self.meters_per_pixel
            world_y = bev_pixel_y * self.meters_per_pixel
            
            return BEVPoint(world_x, world_y)
        except (cv2.error, ValueError, IndexError) as e:
            logger.debug(f"BEV transform failed for pixel {pixel_coords}: {e}")
            return None
            
    def bev_to_gps(self, point: BEVPoint) -> Tuple[float, float]:
        """
        Convert BEV world coordinates (meters) to georeferenced GPS (latitude, longitude).
        
        Uses a local flat-Earth projection which is accurate for local sensor areas.
        
        Args:
            point: BEVPoint in meters
            
        Returns:
            (latitude, longitude) in degrees
        """
        reference_latitude = self.reference_latitude
        reference_longitude = self.reference_longitude
        
        # Distance in meters per degree of latitude is constant
        meters_per_degree_lat = 111319.9
        latitude_degrees = reference_latitude + (point.y / meters_per_degree_lat)
        
        # Convert longitude based on reference latitude cosine
        average_latitude_radians = np.radians(reference_latitude)
        meters_per_degree_lon = meters_per_degree_lat * np.cos(average_latitude_radians)
        longitude_degrees = reference_longitude + (point.x / meters_per_degree_lon)
        
        return float(latitude_degrees), float(longitude_degrees)
    
    def create_bev_canvas(self) -> np.ndarray:
        """
        Create fresh BEV canvas with grid.
        
        Returns:
            BEV canvas image (map_height x map_width x 3)
        """
        # Dark background
        canvas = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        
        # Draw grid (1 meter spacing)
        grid_spacing = int(1.0 / self.meters_per_pixel)
        
        for i in range(0, self.map_width, grid_spacing):
            cv2.line(canvas, (i, 0), (i, self.map_height), (40, 40, 40), 1)
        
        for i in range(0, self.map_height, grid_spacing):
            cv2.line(canvas, (0, i), (self.map_width, i), (40, 40, 40), 1)
        
        # Add labels
        cv2.putText(
            canvas, "BEV Map", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        return canvas
    
    def draw_target_on_bev(
        self,
        canvas: np.ndarray,
        pixel_coords: Tuple[float, float],
        target_id: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        is_anomalous: bool = False
    ) -> np.ndarray:
        """
        Draw target on BEV canvas.
        
        Args:
            canvas: BEV canvas to draw on
            pixel_coords: Target center in pixel coordinates
            target_id: Target ID for labeling
            color: Draw color (BGR)
            is_anomalous: Whether target has anomalous behavior
        
        Returns:
            Updated canvas
        """
        bev_point = self.pixel_to_bev(pixel_coords)
        if bev_point is None:
            return canvas
        
        bev_pixel_x, bev_pixel_y = bev_point.to_pixel(self.meters_per_pixel)
        
        # Bounds check
        if not (0 <= bev_pixel_x < self.map_width and 0 <= bev_pixel_y < self.map_height):
            return canvas
        
        # Draw circle
        draw_color = (0, 0, 255) if is_anomalous else color
        cv2.circle(canvas, (bev_pixel_x, bev_pixel_y), 5, draw_color, -1)
        cv2.circle(canvas, (bev_pixel_x, bev_pixel_y), 8, draw_color, 1)
        
        # Draw ID
        cv2.putText(
            canvas, f"T{target_id}", (bev_pixel_x + 10, bev_pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1
        )
        
        return canvas
    
    def draw_trajectory_on_bev(
        self,
        canvas: np.ndarray,
        trajectory_pixels: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (100, 200, 100)
    ) -> np.ndarray:
        """
        Draw trajectory trail on BEV canvas.
        
        Args:
            canvas: BEV canvas
            trajectory_pixels: List of pixel coordinates in time order
            color: Trail color (BGR)
        
        Returns:
            Updated canvas
        """
        if len(trajectory_pixels) < 2:
            return canvas
        
        bev_points = []
        for pixel_x, pixel_y in trajectory_pixels:
            bev_point = self.pixel_to_bev((pixel_x, pixel_y))
            if bev_point:
                bev_pixel_x, bev_pixel_y = bev_point.to_pixel(self.meters_per_pixel)
                if 0 <= bev_pixel_x < self.map_width and 0 <= bev_pixel_y < self.map_height:
                    bev_points.append((bev_pixel_x, bev_pixel_y))
        
        # Draw polyline
        if len(bev_points) >= 2:
            points = np.array(bev_points, dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=False, color=color, thickness=2)
        
        return canvas
    
    def get_homography_matrix(self) -> np.ndarray:
        """Get the computed homography matrix."""
        return self._homography_matrix.copy()
