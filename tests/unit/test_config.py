"""
Defense COP v2.0 - Configuration Unit Tests
Tests for COPConfig typed configuration system.
"""
import yaml  # type: ignore

from core.config import (
    COPConfig
)


class TestCOPConfigDefaults:
    """Test default configuration values."""
    
    def test_default_config(self):
        config = COPConfig()
        assert config.bev.enabled is True
        assert config.bev.map_width == 400
        assert config.bev.map_height == 400
        assert config.anomaly.velocity_z_threshold == 2.5
        assert config.anomaly.direction_z_threshold == 2.0
        assert config.tracking.iou_threshold == 0.45
        assert config.video.fps == 30
        assert config.headless is False
    
    def test_bev_calibration_points(self):
        config = COPConfig()
        assert len(config.bev.calibration_points) == 4
    
    def test_default_validation_passes(self):
        config = COPConfig()
        errors = config.validate()
        assert len(errors) == 0


class TestCOPConfigYAML:
    """Test YAML load/save."""
    
    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        yaml_content = {
            "bev": {"enabled": False, "map_width": 800},
            "anomaly": {"velocity_z_threshold": 3.0},
            "video": {"fps": 60},
            "headless": True
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)
        
        config = COPConfig.from_yaml(yaml_path)
        
        assert config.bev.enabled is False
        assert config.bev.map_width == 800
        assert config.anomaly.velocity_z_threshold == 3.0
        assert config.video.fps == 60
        assert config.headless is True
    
    def test_from_yaml_empty(self, tmp_path):
        """Test loading empty YAML returns defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        
        config = COPConfig.from_yaml(yaml_path)
        
        # Should use defaults
        assert config.bev.enabled is True
        assert config.video.fps == 30
    
    def test_from_yaml_unknown_keys_ignored(self, tmp_path):
        """Test that unknown keys don't cause crashes."""
        yaml_content = {
            "bev": {"enabled": True, "unknown_field": 42},
            "completely_unknown": {"foo": "bar"}
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)
        
        config = COPConfig.from_yaml(yaml_path)  # Should not crash
        assert config.bev.enabled is True
    
    def test_round_trip(self, tmp_path):
        """Test save then load produces same config."""
        original = COPConfig()
        original.bev.map_width = 600
        original.anomaly.velocity_z_threshold = 3.5
        original.headless = True
        
        yaml_path = tmp_path / "roundtrip.yaml"
        original.to_yaml(yaml_path)
        loaded = COPConfig.from_yaml(yaml_path)
        
        assert loaded.bev.map_width == 600
        assert loaded.anomaly.velocity_z_threshold == 3.5
        assert loaded.headless is True


class TestCOPConfigValidation:
    """Test configuration validation."""
    
    def test_negative_fps(self):
        config = COPConfig()
        config.video.fps = -1
        errors = config.validate()
        assert any("fps" in e for e in errors)
    
    def test_zero_fps(self):
        config = COPConfig()
        config.video.fps = 0
        errors = config.validate()
        assert any("fps" in e for e in errors)
    
    def test_negative_z_threshold(self):
        config = COPConfig()
        config.anomaly.velocity_z_threshold = -1.0
        errors = config.validate()
        assert any("velocity_z_threshold" in e for e in errors)
    
    def test_invalid_bev_dimensions(self):
        config = COPConfig()
        config.bev.map_width = 0
        errors = config.validate()
        assert any("bev map dimensions" in e for e in errors)
    
    def test_invalid_calibration_points(self):
        config = COPConfig()
        config.bev.calibration_points = [(0, 0, 0, 0)]  # Only 1 point
        errors = config.validate()
        assert any("calibration_points" in e for e in errors)
    
    def test_iou_threshold_out_of_range(self):
        config = COPConfig()
        config.tracking.iou_threshold = 1.5
        errors = config.validate()
        assert any("iou_threshold" in e for e in errors)
    
    def test_valid_config_no_errors(self):
        config = COPConfig()
        errors = config.validate()
        assert errors == []
    
    def test_multiple_errors(self):
        config = COPConfig()
        config.video.fps = -1
        config.bev.map_width = 0
        config.tracking.iou_threshold = 2.0
        errors = config.validate()
        assert len(errors) >= 3

    def test_invalid_acceleration(self):
        config = COPConfig()
        config.detector.acceleration = "invalid_format"
        errors = config.validate()
        assert any("detector.acceleration" in err for err in errors)