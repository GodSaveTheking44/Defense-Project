"""
Defense COP v2.0 - BEV Mapper Unit Tests
Deterministic tests for Bird's Eye View transformation.
"""
import pytest
import numpy as np
from engine.spatial import BEVMapper, BEVPoint


class TestBEVPoint:
    """Test BEV point representation."""
    
    def test_to_pixel_conversion(self):
        """Test BEV coordinate to pixel conversion."""
        point = BEVPoint(x=5.0, y=3.0)
        meters_per_pixel = 0.025
        
        px, py = point.to_pixel(meters_per_pixel)
        
        assert px == 200  # 5.0 / 0.025
        assert py == 120  # 3.0 / 0.025
    
    def test_zero_coordinates(self):
        """Test zero origin point."""
        point = BEVPoint(x=0.0, y=0.0)
        px, py = point.to_pixel(0.025)
        
        assert px == 0
        assert py == 0


class TestBEVMapper:
    """Test BEV mapping system."""
    
    @pytest.fixture
    def standard_calibration(self):
        """Standard calibration points for testing."""
        return [
            (100, 400, 0, 0),
            (540, 400, 10, 0),
            (200, 150, 0, 10),
            (440, 150, 10, 10)
        ]
    
    @pytest.fixture
    def mapper(self, standard_calibration):
        """Create BEV mapper instance."""
        return BEVMapper(
            calibration_points=standard_calibration,
            map_width=400,
            map_height=400,
            meters_per_pixel=0.025
        )
    
    def test_initialization(self, mapper):
        """Test BEV mapper initialization."""
        assert mapper.map_width == 400
        assert mapper.map_height == 400
        assert mapper.meters_per_pixel == 0.025
        assert mapper._homography_matrix is not None
        assert mapper._homography_matrix.shape == (3, 3)
    
    def test_homography_matrix_deterministic(self, standard_calibration):
        """Test that homography matrix is deterministic."""
        mapper1 = BEVMapper(
            calibration_points=standard_calibration,
            map_width=400,
            map_height=400,
            meters_per_pixel=0.025
        )
        
        mapper2 = BEVMapper(
            calibration_points=standard_calibration,
            map_width=400,
            map_height=400,
            meters_per_pixel=0.025
        )
        
        np.testing.assert_array_almost_equal(
            mapper1.get_homography_matrix(),
            mapper2.get_homography_matrix()
        )
    
    def test_pixel_to_bev_transformation(self, mapper):
        """Test pixel to BEV coordinate transformation."""
        # Test calibration point transformation
        pixel_coord = (100, 400)
        bev_point = mapper.pixel_to_bev(pixel_coord)
        
        assert bev_point is not None
        # Should map close to origin (0, 0) in world coords
        assert abs(bev_point.x) < 1.0
        assert abs(bev_point.y) < 1.0
    
    def test_create_bev_canvas(self, mapper):
        """Test BEV canvas creation."""
        canvas = mapper.create_bev_canvas()
        
        assert canvas.shape == (400, 400, 3)
        assert canvas.dtype == np.uint8
        # Should not be all zeros (has grid and labels)
        assert np.any(canvas > 0)
    
    def test_draw_target_on_bev(self, mapper):
        """Test drawing target on BEV canvas."""
        canvas = mapper.create_bev_canvas()
        original_canvas = canvas.copy()
        
        pixel_coords = (300, 300)
        result = mapper.draw_target_on_bev(
            canvas,
            pixel_coords,
            target_id=1,
            color=(0, 255, 0),
            is_anomalous=False
        )
        
        # Canvas should be modified
        assert not np.array_equal(result, original_canvas)
    
    def test_draw_target_anomalous_color(self, mapper):
        """Test anomalous target uses red color."""
        canvas1 = mapper.create_bev_canvas()
        canvas2 = mapper.create_bev_canvas()
        
        pixel_coords = (300, 300)
        
        # Draw normal target
        mapper.draw_target_on_bev(
            canvas1,
            pixel_coords,
            target_id=1,
            is_anomalous=False
        )
        
        # Draw anomalous target
        mapper.draw_target_on_bev(
            canvas2,
            pixel_coords,
            target_id=2,
            is_anomalous=True
        )
        
        # Canvases should differ (different colors)
        assert not np.array_equal(canvas1, canvas2)
    
    def test_draw_trajectory(self, mapper):
        """Test trajectory drawing."""
        canvas = mapper.create_bev_canvas()
        
        trajectory = [
            (100, 400),
            (150, 350),
            (200, 300),
            (250, 250)
        ]
        
        result = mapper.draw_trajectory_on_bev(canvas, trajectory)
        
        # Canvas should have trajectory drawn
        assert result is not None
    
    def test_trajectory_single_point(self, mapper):
        """Test trajectory with single point (should do nothing)."""
        canvas = mapper.create_bev_canvas()
        original = canvas.copy()
        
        trajectory = [(300, 300)]
        result = mapper.draw_trajectory_on_bev(canvas, trajectory)
        
        # Should be unchanged
        np.testing.assert_array_equal(result, original)
    
    def test_out_of_bounds_handling(self, mapper):
        """Test handling of out-of-bounds coordinates."""
        canvas = mapper.create_bev_canvas()
        
        # Way out of bounds
        pixel_coords = (10000, 10000)
        result = mapper.draw_target_on_bev(
            canvas,
            pixel_coords,
            target_id=99
        )
        
        # Should not crash
        assert result is not None
    
    def test_different_map_sizes(self):
        """Test BEV mapper with different map sizes."""
        calibration = [(100, 400, 0, 0), (540, 400, 10, 0), 
                      (200, 150, 0, 10), (440, 150, 10, 10)]
        
        mapper_small = BEVMapper(
            calibration_points=calibration,
            map_width=200,
            map_height=200,
            meters_per_pixel=0.05
        )
        
        mapper_large = BEVMapper(
            calibration_points=calibration,
            map_width=800,
            map_height=800,
            meters_per_pixel=0.0125
        )
        
        assert mapper_small.map_width == 200
        assert mapper_large.map_width == 800

    def test_bev_to_gps_conversion(self):
        """Test conversion of BEV relative coordinates to georeferenced GPS."""
        calibration = [(100, 400, 0, 0), (540, 400, 10, 0), 
                      (200, 150, 0, 10), (440, 150, 10, 10)]
        mapper = BEVMapper(
            calibration_points=calibration,
            map_width=400,
            map_height=400,
            meters_per_pixel=0.025,
            reference_latitude=34.0522,
            reference_longitude=-118.2437
        )
        
        from engine.spatial import BEVPoint
        p1 = BEVPoint(0.0, 0.0)
        lat1, lon1 = mapper.bev_to_gps(p1)
        assert abs(lat1 - 34.0522) < 1e-6
        assert abs(lon1 - (-118.2437)) < 1e-6
        
        p2 = BEVPoint(1000.0, 1000.0)
        lat2, lon2 = mapper.bev_to_gps(p2)
        assert lat2 > 34.0522
        assert lon2 > -118.2437
