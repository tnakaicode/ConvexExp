#!/usr/bin/env python3
"""
Test script for the face unfolding functionality.
This script demonstrates various ways to use the unfold_adjacent_faces method.
"""

import sys
import os
sys.path.append(os.path.join("../"))

from convex import CovExp
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_box


def test_simple_cube():
    """Test unfolding with a simple cube"""
    print("="*50)
    print("Testing Simple Cube Unfolding")
    print("="*50)
    
    obj = CovExp(touch=False)
    unfolded_faces = obj.test_unfold_cube()
    
    print(f"Successfully unfolded {len(unfolded_faces)} faces")
    print("Close the 3D viewer to continue...")
    obj.ShowOCC()


def test_rectangular_box():
    """Test unfolding with a rectangular box"""
    print("="*50)
    print("Testing Rectangular Box Unfolding")
    print("="*50)
    
    obj = CovExp(touch=False)
    
    # Create a rectangular box
    rect_box = make_box(80, 60, 40)
    
    # Get the first face as base
    face_exp = TopExp_Explorer(rect_box, TopAbs_FACE)
    base_face = face_exp.Current()
    
    print(f"Box dimensions: 80 x 60 x 40")
    print(f"Volume: {obj.cal_vol(rect_box)}")
    print(f"Number of faces: {TopologyExplorer(rect_box).number_of_faces()}")
    
    # Unfold adjacent faces
    unfolded_faces = obj.unfold_adjacent_faces(base_face, rect_box)
    
    # Display the original box
    obj.display.DisplayShape(rect_box, color="GRAY", transparency=0.7)
    
    print(f"Successfully unfolded {len(unfolded_faces)} faces")
    print("Close the 3D viewer to continue...")
    obj.ShowOCC()


def test_different_base_faces():
    """Test unfolding using different faces as base"""
    print("="*50)
    print("Testing Different Base Faces")
    print("="*50)
    
    obj = CovExp(touch=False)
    
    # Test with each face as base
    for face_num in range(1, 4):  # Test first 3 faces
        print(f"\n--- Using Face {face_num} as Base ---")
        
        obj_temp = CovExp(touch=False)
        face_exp = TopExp_Explorer(obj_temp.base, TopAbs_FACE)
        base_face = face_exp.Current()
        
        # Navigate to the specified face
        for i in range(face_num - 1):
            if face_exp.More():
                face_exp.Next()
                base_face = face_exp.Current()
        
        # Unfold adjacent faces
        unfolded_faces = obj_temp.unfold_adjacent_faces(base_face, obj_temp.base)
        
        # Display the original box
        obj_temp.display.DisplayShape(obj_temp.base, color="LIGHTGRAY", transparency=0.8)
        
        print(f"Unfolded {len(unfolded_faces)} faces using face {face_num} as base")
        print(f"Close the viewer to continue to face {face_num + 1}...")
        obj_temp.ShowOCC()


def test_split_solid():
    """Test unfolding with a split solid"""
    print("="*50)
    print("Testing Split Solid Unfolding")
    print("="*50)
    
    obj = CovExp(touch=False)
    
    # Split the base solid
    print("Splitting the base solid with random planes...")
    obj.split_run(2, seed=42)  # Use fixed seed for reproducible results
    
    # Show all split solids
    print("Displaying all split solids...")
    obj.show_split_solid()
    
    # Get the first split solid
    sol_exp = TopExp_Explorer(obj.splitter.Shape(), TopAbs_SOLID)
    if sol_exp.More():
        current_solid = sol_exp.Current()
        face_exp = TopExp_Explorer(current_solid, TopAbs_FACE)
        
        if face_exp.More():
            base_face = face_exp.Current()
            
            print(f"Selected solid volume: {obj.cal_vol(current_solid)}")
            print(f"Number of faces: {TopologyExplorer(current_solid).number_of_faces()}")
            
            # Unfold adjacent faces
            unfolded_faces = obj.unfold_adjacent_faces(base_face, current_solid)
            
            print(f"Successfully unfolded {len(unfolded_faces)} faces")
            print("Close the 3D viewer to continue...")


def main():
    """Main test function"""
    print("Face Unfolding Test Suite")
    print("This will run several tests to demonstrate the unfolding functionality.")
    print()
    
    tests = [
        ("Simple Cube", test_simple_cube),
        ("Rectangular Box", test_rectangular_box),
        ("Different Base Faces", test_different_base_faces),
        ("Split Solid", test_split_solid)
    ]
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{i}. {name}")
        choice = input("Run this test? (y/n/q): ").lower()
        
        if choice == 'q':
            break
        elif choice == 'y' or choice == '':
            try:
                test_func()
            except Exception as e:
                print(f"Error during test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipped.")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
