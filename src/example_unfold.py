#!/usr/bin/env python3
"""
Example script demonstrating the Box Split → Polyhedron Extract → Face Unfold workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from convex import CovExp


def example_basic_unfold():
    """Basic example: Split box and unfold first polyhedron"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Box Split and Unfold")
    print("=" * 60)
    
    # Create the convex expansion object
    obj = CovExp(touch=False)
    
    # Run the complete demonstration
    solid, base_face, result = obj.demo_split_and_unfold(
        num_splits=3,     # Split with 3 random planes
        seed=42,          # Fixed seed for reproducible results
        solid_index=0,    # Use the first split solid
        face_index=0      # Use the first face as base
    )
    
    if solid:
        print(f"✓ Success! Unfolded polyhedron with volume {obj.cal_vol(solid):.2f}")
    else:
        print("✗ Failed to unfold polyhedron")


def example_multiple_solids():
    """Example: Test unfolding with multiple split solids"""
    print("=" * 60)
    print("EXAMPLE 2: Multiple Solids Test")
    print("=" * 60)
    
    obj = CovExp(touch=False)
    
    # Split the box
    print("Splitting box with 4 planes...")
    obj.split_run(4, seed=123)
    
    # Try unfolding different solids
    for i in range(3):  # Test first 3 solids
        print(f"\n--- Testing Solid {i} ---")
        
        try:
            obj_temp = CovExp(touch=False)
            obj_temp.splitter = obj.splitter  # Copy the split result
            
            solid, base_face, result = obj_temp.extract_and_unfold_polyhedron(i, 0)
            
            if solid:
                vol = obj_temp.cal_vol(solid)
                area = obj_temp.cal_are(base_face)
                print(f"  ✓ Solid {i}: Volume={vol:.2f}, Base area={area:.2f}")
            else:
                print(f"  ✗ Solid {i}: Failed to extract")
                
        except Exception as e:
            print(f"  ✗ Solid {i}: Error - {e}")


def example_different_faces():
    """Example: Test unfolding from different base faces"""
    print("=" * 60)
    print("EXAMPLE 3: Different Base Faces Test")
    print("=" * 60)
    
    obj = CovExp(touch=False)
    
    # Split the box
    obj.split_run(3, seed=456)
    
    # Try different faces as base
    for face_idx in range(3):  # Test first 3 faces
        print(f"\n--- Using Face {face_idx} as Base ---")
        
        try:
            obj_temp = CovExp(touch=False)
            obj_temp.splitter = obj.splitter
            
            solid, base_face, result = obj_temp.extract_and_unfold_polyhedron(0, face_idx)
            
            if solid:
                area = obj_temp.cal_are(base_face)
                print(f"  ✓ Face {face_idx}: Base area={area:.2f}")
                # Display result
                obj_temp.ShowOCC()
            else:
                print(f"  ✗ Face {face_idx}: Failed")
                
        except Exception as e:
            print(f"  ✗ Face {face_idx}: Error - {e}")


def example_custom_parameters():
    """Example: Custom parameters demonstration"""
    print("=" * 60)
    print("EXAMPLE 4: Custom Parameters")
    print("=" * 60)
    
    # Test different numbers of splitting planes
    split_counts = [2, 3, 5]
    
    for splits in split_counts:
        print(f"\n--- Testing {splits} splits ---")
        
        obj = CovExp(touch=False)
        
        try:
            solid, base_face, result = obj.demo_split_and_unfold(
                num_splits=splits,
                seed=100 + splits,  # Different seed for each test
                solid_index=0,
                face_index=0
            )
            
            if solid:
                vol = obj.cal_vol(solid)
                print(f"  ✓ {splits} splits: Volume={vol:.2f}")
            else:
                print(f"  ✗ {splits} splits: Failed")
                
        except Exception as e:
            print(f"  ✗ {splits} splits: Error - {e}")


def interactive_example():
    """Interactive example with user input"""
    print("=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)
    
    print("This example lets you customize the parameters.")
    
    try:
        # Get user input
        splits = int(input("Number of splitting planes (2-10): ") or "3")
        solid_idx = int(input("Solid index to unfold (0-based): ") or "0")
        face_idx = int(input("Face index for base (0-based): ") or "0")
        
        print(f"\nRunning with: {splits} splits, solid {solid_idx}, face {face_idx}")
        
        obj = CovExp(touch=False)
        solid, base_face, result = obj.demo_split_and_unfold(splits, 999, solid_idx, face_idx)
        
        if solid:
            print(f"✓ Success! Check the 3D viewer for results.")
        else:
            print("✗ Unfolding failed.")
            
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or interrupted.")


def main():
    """Main function to run examples"""
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
    else:
        print("Available examples:")
        print("1 - basic")
        print("2 - multiple") 
        print("3 - faces")
        print("4 - custom")
        print("5 - interactive")
        print()
        example_name = input("Select example (1-5): ")
    
    examples = {
        "1": example_basic_unfold,
        "basic": example_basic_unfold,
        "2": example_multiple_solids,
        "multiple": example_multiple_solids,
        "3": example_different_faces,
        "faces": example_different_faces,
        "4": example_custom_parameters,
        "custom": example_custom_parameters,
        "5": interactive_example,
        "interactive": interactive_example
    }
    
    if example_name in examples:
        try:
            examples[example_name]()
            print("\nExample completed!")
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid example selection")
        print("Running basic example...")
        example_basic_unfold()


if __name__ == "__main__":
    main()
