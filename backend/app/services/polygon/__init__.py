"""Polygon.io data service for stock market data and technical analysis."""

from .client import PolygonClient, get_polygon_client
from .indicators import calculate_rsi, calculate_sma
from .screening import TechnicalAnalysis, TechnicalData, compute_technical_score

__all__ = [
    "PolygonClient",
    "TechnicalAnalysis",
    "TechnicalData",
    "calculate_rsi",
    "calculate_sma",
    "compute_technical_score",
    "get_polygon_client",
]
