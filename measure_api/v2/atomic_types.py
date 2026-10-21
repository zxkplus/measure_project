"""Typed observations and geometric ports shared by atomic operations and recipes."""

from typing import List, Optional

from pydantic import Field, model_validator

from measure_api.v2.models import Point, StrictModel


class Observation(StrictModel):
    """Validity describes the observation, independently of execution and business limits."""

    valid: bool
    reason: Optional[str] = None
    frame: str = Field(default='source', min_length=1)
    unit: str = Field(default='px', min_length=1)


class PositionedPoint(Observation):
    point: Optional[Point] = None


class Scalar(Observation):
    value: Optional[float] = None


class Line(Observation):
    """Normalized infinite a*x+b*y+c=0 line with optional measured endpoints."""

    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    start: Optional[Point] = None
    end: Optional[Point] = None

    @model_validator(mode='after')
    def normalize(self) -> 'Line':
        """Reject missing or degenerate valid coefficients; normalize external lines.

        Returns:
            This line with unit-normalized coefficients and validated nondegenerate geometry.
        """
        if self.valid:
            if self.a is None or self.b is None or self.c is None:
                raise ValueError('Valid line requires a, b and c')
            norm = (self.a * self.a + self.b * self.b) ** 0.5
            if norm <= 1e-12:
                raise ValueError('Line normal must be nonzero')
            self.a, self.b, self.c = self.a / norm, self.b / norm, self.c / norm
        return self


class Circle(Observation):
    center: Optional[Point] = None
    radius: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode='after')
    def check_circle(self) -> 'Circle':
        """Usable circles require an observed center and positive radius.

        Returns:
            This circle after validating its observed center and positive radius.
        """
        if self.valid and (self.center is None or self.radius is None):
            raise ValueError('Valid circle requires center and positive radius')
        return self


class FitDiagnostics(StrictModel):
    sample_count: int
    requested_probe_count: int = 0
    successful_probe_count: int = 0
    residuals: List[float] = Field(default_factory=list)
    mean_absolute_error: Optional[float] = None
    max_absolute_error: Optional[float] = None
    rms_error: Optional[float] = None
