"""
Logging Configuration for CoT Rewriter

Uses standard Python logging library.
Compatible with Ray distributed environment.

This module is kept for backward compatibility.
New code should use training_logger.py instead.
"""

import logging
import os
import sys


def get_logger(name: str = "cot_rewriter") -> logging.Logger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    level = os.getenv("VERL_LOGGING_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Add handler if not already added
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logger.level)
        
        # Format: timestamp | level | name | message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


# Default logger for backward compatibility
logger = get_logger("cot_rewriter")

# Export for use in other modules
__all__ = ["logger", "get_logger"]
