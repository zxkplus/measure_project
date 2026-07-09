"""
MultiTargetWorkflow - Multi-target template matching workflow.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from measure.measure_template import TemplatePoint, Preprocessor


class MultiTargetWorkflow:
    """Multi-target template matching workflow manager."""
    
    def __init__(self, reference_image: np.ndarray, preprocessor: Preprocessor = None):
        """Initialize workflow with reference image and optional preprocessor."""
        self.reference_image = reference_image
        self.preprocessor = preprocessor
        self.targets: Dict[str, TemplatePoint] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def add_target(self, name: str, row: float, col: float, 
                   template_size: int = 80, **kwargs) -> None:
        """Add a target template point."""
        self.targets[name] = TemplatePoint(
            self.reference_image, row, col, 
            template_size=template_size,
            preprocessor=self.preprocessor,
            **kwargs
        )
    
    def remove_target(self, name: str) -> None:
        """Remove a target by name."""
        if name not in self.targets:
            raise ValueError(f"Target '{name}' not found")
        del self.targets[name]
        self.results.pop(name, None)
    
    def measure_single(self, name: str, inspection_image: np.ndarray) -> Dict[str, Any]:
        """Measure a single target on inspection image."""
        if name not in self.targets:
            raise ValueError(f"Target '{name}' not found")
        result = self.targets[name].measure(inspection_image)
        self.results[name] = result
        return result
    
    def measure_all(self, inspection_image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Measure all targets on inspection image."""
        self.results = {}
        for name, target in self.targets.items():
            self.results[name] = target.measure(inspection_image)
        return self.results
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all measurement results."""
        return self.results
    
    def visualize(self, image: np.ndarray, show: bool = True, 
                  wait_time: int = 1000) -> np.ndarray:
        """Visualize measurement results on image."""
        result_img = image.copy()
        
        for name, result in self.results.items():
            if not result.get('valid', False):
                continue
            
            row = int(result['matched_row'])
            col = int(result['matched_col'])
            score = result.get('match_score', 0)
            
            # Draw crosshair
            cv2 = __import__('cv2')
            cv2.drawMarker(result_img, (col, row), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Draw label
            label = f"{name}: {score:.2f}"
            cv2.putText(result_img, label, (col + 10, row - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if show:
            cv2.imshow('MultiTarget Results', result_img)
            cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        
        return result_img
    
    def save(self, path: str) -> None:
        """Save workflow to .npz file."""
        save_data = {
            'reference_image': self.reference_image,
            'target_names': list(self.targets.keys()),
            'target_configs': []
        }
        
        for name, target in self.targets.items():
            save_data['target_configs'].append({
                'name': name,
                'row': target.row,
                'col': target.col,
                'template_size': target.template_size
            })
        
        np.savez(path, **save_data)
    
    @classmethod
    def load(cls, path: str, preprocessor: Preprocessor = None) -> 'MultiTargetWorkflow':
        """Load workflow from .npz file."""
        data = np.load(path, allow_pickle=True)
        workflow = cls(data['reference_image'], preprocessor)
        
        for config in data['target_configs']:
            workflow.add_target(
                config['name'],
                config['row'],
                config['col'],
                config['template_size']
            )
        
        return workflow
