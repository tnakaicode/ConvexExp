#!/usr/bin/env python3
"""
Simple test for polyhedron unfolding.
Non-interactive execution only.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from convex import CovExp


def test_basic_unfold():
    """Basic test: Split box and unfold polyhedron"""
    print("=" * 50)
    print("BASIC UNFOLD TEST")
    print("=" * 50)
    
    obj = CovExp(touch=False)
    
    solid, base_face, result = obj.demo_split_and_unfold_clean(
        num_splits=3,
        seed=42,
        solid_index=0,
        face_index=0
    )
    
    if solid:
        print(f"✓ Test passed!")
        print(f"Volume: {obj.cal_vol(solid):.2f}")
        print(f"Base area: {obj.cal_are(base_face):.2f}")
        print(f"Unfolded faces: {result}")
    else:
        print("✗ Test failed!")


def test_different_parameters():
    """Test with different parameters"""
    print("=" * 50)
    print("PARAMETER VARIATION TEST")
    print("=" * 50)
    
    test_cases = [
        (2, 100, 0, 0),  # Simple case
        (4, 200, 1, 1),  # More complex
        (3, 300, 0, 2),  # Different face
    ]
    
    for i, (splits, seed, solid_idx, face_idx) in enumerate(test_cases):
        print(f"\nTest case {i+1}: splits={splits}, solid={solid_idx}, face={face_idx}")
        
        obj = CovExp(touch=False)
        try:
            solid, base_face, result = obj.demo_split_and_unfold_clean(
                splits, seed, solid_idx, face_idx
            )
            
            if solid:
                vol = obj.cal_vol(solid)
                area = obj.cal_are(base_face)
                print(f"  ✓ Success: Volume={vol:.2f}, Area={area:.2f}")
            else:
                print(f"  ✗ Failed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    # Non-interactive execution
    test_basic_unfold()
    test_different_parameters()
    print("\nAll tests completed!")
