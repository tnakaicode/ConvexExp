#!/usr/bin/env python3
"""
Simple face unfolding demonstration.
Clean implementation without complex geometry calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from convex import CovExp
import math


def simple_demonstration():
    """Demonstrate simple face unfolding"""
    print("=" * 60)
    print("SIMPLE FACE UNFOLDING DEMONSTRATION")
    print("=" * 60)
    
    obj = CovExp(touch=False)
    
    # Use the simple clean demo
    try:
        solid, base_face, result = obj.demo_split_and_unfold_clean(
            num_splits=3,
            seed=42,
            solid_index=0,
            face_index=0
        )
        
        if solid:
            print(f"✓ Simple demonstration completed!")
            print(f"  Polyhedron volume: {obj.cal_vol(solid):.2f}")
            print(f"  Base face area: {obj.cal_are(base_face):.2f}")
            print(f"  Processed faces: {result}")
        else:
            print("✗ Demonstration failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    simple_demonstration()
