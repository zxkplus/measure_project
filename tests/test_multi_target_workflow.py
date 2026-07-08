"""
Tests for MultiTargetWorkflow - multi-target template matching workflow.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from measure_template import TemplatePoint
from measure_gui.multi_target import (
    MultiTargetWorkflow,
    create_workflow_from_template_images
)


class TestMultiTargetWorkflow:
    """Test suite for MultiTargetWorkflow."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        # Reference image (100x100 with some features)
        ref_img = np.zeros((100, 100), dtype=np.uint8)
        ref_img[20:40, 20:40] = 255  # Top-left bright region
        ref_img[60:80, 60:80] = 200  # Bottom-right bright region
        
        # Inspection image with translated features
        insp_img = np.zeros((100, 100), dtype=np.uint8)
        insp_img[25:45, 25:45] = 255  # Top-left shifted by (5,5)
        insp_img[65:85, 65:85] = 200  # Bottom-right shifted by (5,5)
        
        return ref_img, insp_img
    
    @pytest.fixture
    def workflow(self, sample_images):
        """Create a workflow with two template points."""
        ref_img, _ = sample_images
        workflow = MultiTargetWorkflow(ref_img)
        
        # Add template point A at top-left feature
        workflow.add_template_point(
            name="target_a",
            row=30, col=30,
            template_size=30,
            search_region=(0, 0, 50, 50)
        )
        
        # Add template point B at bottom-right feature
        workflow.add_template_point(
            name="target_b",
            row=70, col=70,
            template_size=30,
            search_region=(50, 50, 100, 100)
        )
        
        return workflow
    
    def test_workflow_creation(self, workflow):
        """Test basic workflow creation."""
        assert workflow is not None
        assert len(workflow.template_points) == 2
        assert "target_a" in workflow.template_points
        assert "target_b" in workflow.template_points
    
    def test_add_template_point(self, workflow):
        """Test adding template points."""
        # Add another template point
        workflow.add_template_point(
            name="target_c",
            row=50, col=50,
            template_size=20
        )
        
        assert len(workflow.template_points) == 3
        assert "target_c" in workflow.template_points
    
    def test_add_duplicate_template(self, workflow):
        """Test that duplicate template names raise error."""
        with pytest.raises(ValueError, match="already exists"):
            workflow.add_template_point(
                name="target_a",  # Already exists
                row=50, col=50,
                template_size=20
            )
    
    def test_remove_template_point(self, workflow):
        """Test removing template points."""
        workflow.remove_template_point("target_a")
        
        assert len(workflow.template_points) == 1
        assert "target_a" not in workflow.template_points
        assert "target_b" in workflow.template_points
    
    def test_measure_all(self, workflow, sample_images):
        """Test measuring all template points."""
        _, insp_img = sample_images
        results = workflow.measure_all(insp_img)
        
        # Check we got results for both targets
        assert len(results) == 2
        assert "target_a" in results
        assert "target_b" in results
        
        # Check result structure
        for name, result in results.items():
            assert "matched_row" in result
            assert "matched_col" in result
            assert "match_score" in result
            assert "dx" in result
            assert "dy" in result
            assert "valid" in result
    
    def test_measure_single_target(self, workflow, sample_images):
        """Test measuring a single target."""
        _, insp_img = sample_images
        result = workflow.measure_single("target_a", insp_img)
        
        assert result is not None
        assert "matched_row" in result
        assert "matched_col" in result
    
    def test_measure_single_nonexistent(self, workflow, sample_images):
        """Test measuring a non-existent target."""
        _, insp_img = sample_images
        
        with pytest.raises(ValueError, match="not found"):
            workflow.measure_single("nonexistent", insp_img)
    
    def test_get_results(self, workflow, sample_images):
        """Test getting results after measurement."""
        _, insp_img = sample_images
        workflow.measure_all(insp_img)
        
        results = workflow.get_results()
        assert len(results) == 2
    
    def test_translation_detection(self, workflow, sample_images):
        """Test that translation is correctly detected."""
        _, insp_img = sample_images
        
        # Measure targets
        results = workflow.measure_all(insp_img)
        
        # Check that translation is detected (approximately 5 pixels)
        for name, result in results.items():
            if result["valid"]:
                assert abs(result["dx"]) < 10  # Should be around 5
                assert abs(result["dy"]) < 10  # Should be around 5
    
    def test_visualize(self, workflow, sample_images):
        """Test visualization."""
        _, insp_img = sample_images
        
        # Measure first
        workflow.measure_all(insp_img)
        
        # Visualize
        vis_img = workflow.visualize(insp_img)
        
        # Check that visualization is created
        assert vis_img is not None
        assert vis_img.shape == insp_img.shape
    
    def test_save_and_load(self, workflow, sample_images):
        """Test saving and loading workflow."""
        _, insp_img = sample_images
        
        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "workflow.json")
            
            # Save workflow
            workflow.save(save_path)
            assert os.path.exists(save_path)
            
            # Load workflow
            loaded_workflow = MultiTargetWorkflow.load(save_path)
            
            # Check loaded workflow has same template points
            assert len(loaded_workflow.template_points) == len(workflow.template_points)
            for name in workflow.template_points:
                assert name in loaded_workflow.template_points
    
    def test_save_with_results(self, workflow, sample_images):
        """Test saving workflow with measurement results."""
        _, insp_img = sample_images
        
        # Measure first
        workflow.measure_all(insp_img)
        
        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "workflow_with_results.json")
            
            # Save workflow with results
            workflow.save(save_path, save_results=True)
            
            # Load and check
            loaded_workflow = MultiTargetWorkflow.load(save_path)
            assert loaded_workflow.last_results is not None
    
    def test_get_summary(self, workflow, sample_images):
        """Test getting summary statistics."""
        _, insp_img = sample_images
        
        # Measure first
        workflow.measure_all(insp_img)
        
        # Get summary
        summary = workflow.get_summary()
        
        # Check summary structure
        assert "total_targets" in summary
        assert "valid_targets" in summary
        assert "average_score" in summary
        assert "max_dx" in summary
        assert "max_dy" in summary


class TestCreateWorkflowFromTemplateImages:
    """Test factory function for creating workflow from template images."""
    
    def test_create_workflow(self):
        """Test creating workflow from template images."""
        # Create a reference image
        ref_img = np.zeros((100, 100), dtype=np.uint8)
        ref_img[20:40, 20:40] = 255
        ref_img[60:80, 60:80] = 200
        
        # Define template regions
        template_regions = [
            {"name": "target_a", "row": 30, "col": 30, "size": 30},
            {"name": "target_b", "row": 70, "col": 70, "size": 30},
        ]
        
        # Create workflow
        workflow = create_workflow_from_template_images(
            ref_img, template_regions
        )
        
        # Check workflow
        assert workflow is not None
        assert len(workflow.template_points) == 2
        assert "target_a" in workflow.template_points
        assert "target_b" in workflow.template_points


class TestMultiTargetWorkflowEdgeCases:
    """Test edge cases for MultiTargetWorkflow."""
    
    def test_empty_workflow(self):
        """Test workflow with no template points."""
        ref_img = np.zeros((100, 100), dtype=np.uint8)
        workflow = MultiTargetWorkflow(ref_img)
        
        # Should have no template points
        assert len(workflow.template_points) == 0
        
        # Measuring all should return empty results
        results = workflow.measure_all(ref_img)
        assert len(results) == 0
    
    def test_large_template_size(self):
        """Test with template size larger than image."""
        ref_img = np.zeros((50, 50), dtype=np.uint8)
        workflow = MultiTargetWorkflow(ref_img)
        
        # Try to add template with size larger than image
        with pytest.raises(ValueError):
            workflow.add_template_point(
                name="large_template",
                row=25, col=25,
                template_size=100  # Larger than image
            )
    
    def test_template_at_boundary(self):
        """Test template at image boundary."""
        ref_img = np.zeros((100, 100), dtype=np.uint8)
        ref_img[0:20, 0:20] = 255
        
        workflow = MultiTargetWorkflow(ref_img)
        
        # Add template at boundary
        workflow.add_template_point(
            name="boundary_template",
            row=10, col=10,
            template_size=20
        )
        
        # Should work
        assert len(workflow.template_points) == 1
    
    def test_search_region_validation(self):
        """Test that search region validation works."""
        ref_img = np.zeros((100, 100), dtype=np.uint8)
        workflow = MultiTargetWorkflow(ref_img)
        
        # Invalid search region (top > bottom)
        with pytest.raises(ValueError):
            workflow.add_template_point(
                name="invalid_region",
                row=50, col=50,
                template_size=20,
                search_region=(60, 0, 40, 100)  # top > bottom
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
