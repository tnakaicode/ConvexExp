import numpy as np
import sys
import os

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Circ, gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.gp import gp_Trsf
from OCC.Core.Geom import Geom_Circle, Geom_TrimmedCurve
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Iterator
from OCC.Core.TopoDS import (
    TopoDS_Edge,
    TopoDS_Solid,
    TopoDS_Face,
    topods,
)
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Prs3d import Prs3d_DimensionAspect
from OCC.Core.PrsDim import PrsDim_AngleDimension
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_RED1, Quantity_NOC_BLACK
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_box
from OCCUtils.Construct import vec_to_dir, dir_to_vec, vertex2pnt

from PyQt5.QtWidgets import QApplication, qApp
from PyQt5.QtWidgets import QDialog, QCheckBox

sys.path.append(os.path.join("../"))
from src.base_occ import dispocc


def axs1_to_axs3(axs=gp_Ax1()):
    return gp_Ax3(axs.Location(), axs.Direction())


def axs_pln(axs):
    pnt = axs.Location()
    vx = dir_to_vec(axs.XDirection()).Scaled(100)
    vy = dir_to_vec(axs.YDirection()).Scaled(200)
    vz = dir_to_vec(axs.Direction()).Scaled(300)
    lx = make_edge(pnt, gp_Pnt((gp_Vec(pnt.XYZ()) + vx).XYZ()))
    ly = make_edge(pnt, gp_Pnt((gp_Vec(pnt.XYZ()) + vy).XYZ()))
    lz = make_edge(pnt, gp_Pnt((gp_Vec(pnt.XYZ()) + vz).XYZ()))
    return lx, ly, lz


def get_axs_deg(ax0=gp_Ax3(), ax1=gp_Ax3(), ref=gp_Dir()):
    org_angle = ax0.Angle(ax1)
    ref_angle = ax0.Direction().AngleWithRef(ax1.Direction(), ax0.XDirection())

    if np.sign(org_angle) == np.sign(ref_angle):
        return org_angle
    else:
        return np.pi - ref_angle


class CovExp(dispocc):

    def __init__(self, temp=True, disp=True, touch=False):
        dispocc.__init__(self, temp, disp, touch)
        self.prop = GProp_GProps()
        self.base = make_box(100, 100, 100)
        self.base_vol = self.cal_vol(self.base)

        self.nsol = 1
        self.nfce = 1

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

        self.context = []

    def init_base(self, shape):
        self.nsol = 0
        self.nfce = 0
        self.base = shape
        self.base_vol = self.cal_vol(self.base)

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

    def fileout(self, dirname="./shp/"):
        num = 0
        stp_file = dirname + "shp_{:04d}.stp".format(num)
        write_step_file(self.base, stp_file)

        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            num += 1
            stp_file = dirname + "shp_{:04d}.stp".format(num)
            write_step_file(sol_exp.Current(), stp_file)
            sol_exp.Next()

    def split_run(self, num=5, seed=11):
        if seed != None:
            np.random.seed(seed)
        for i in range(num):
            pnt = gp_Pnt(*np.random.rand(3) * 100)
            vec = gp_Vec(*np.random.randn(3))
            pln = gp_Pln(pnt, vec_to_dir(vec))
            fce = make_face(pln, -10000, 10000, -10000, 10000)
            self.splitter.AddTool(fce)
        self.splitter.Perform()

    def cal_len(self, shp=TopoDS_Shape()):
        brepgprop.LinearProperties(shp, self.prop)
        return self.prop.Mass()

    def cal_are(self, shp=TopoDS_Shape()):
        brepgprop.SurfaceProperties(shp, self.prop)
        return self.prop.Mass()

    def cal_vol(self, shp=TopoDS_Shape()):
        brepgprop.VolumeProperties(shp, self.prop)
        return self.prop.Mass()

    def prop_edge(self, edge=TopoDS_Edge()):
        edge_adaptor = BRepAdaptor_Curve(edge)
        edge_line = edge_adaptor.Line()
        return edge_line

    def face_midpoint(self, face=TopoDS_Face()):
        pts = [vertex2pnt(v) for v in TopologyExplorer(face).vertices()]
        x, y, z = 0, 0, 0
        for p in pts:
            x += p.X()
            y += p.Y()
            z += p.Z()
        return gp_Pnt(x / len(pts), y / len(pts), z / len(pts))

    def face_tranfer(self, face=TopoDS_Face(), axs=gp_Ax1()):
        axs_3 = gp_Ax3(axs.Location(), axs.Direction())
        trf = gp_Trsf()
        trf.SetTransformation(axs_3, self.tmp_axs3)
        loc_face = TopLoc_Location(trf)
        face.Location(loc_face)
        return face

    def pln_on_face(self, face=TopoDS_Face()):
        face_adaptor = BRepAdaptor_Surface(face)
        face_trf = face_adaptor.Trsf()
        face_pln = face_adaptor.Plane()
        # face_dir = face_adaptor.Direction()

        face_umin = face_adaptor.FirstUParameter()
        face_vmin = face_adaptor.FirstVParameter()
        face_umax = face_adaptor.LastUParameter()
        face_vmax = face_adaptor.LastVParameter()
        face_u = (face_umax + face_umin) / 2
        face_v = (face_vmax + face_vmin) / 2
        face_pnt = face_adaptor.Value(face_u, face_v)
        face_pnt = self.face_midpoint(face)
        face_pln.SetLocation(face_pnt)
        return face_pln

    def face_fillet(self, face=TopoDS_Face()):
        plan = self.pln_on_face(face)
        find_edge = LocOpe_FindEdges(self.fix_face, face)
        find_edge.InitIterator()
        edge_n = 0

        edge = find_edge.EdgeTo()
        self.display.DisplayShape(edge, color="BLUE1")
        self.fill.Add(10, edge)
        self.fill.Build()
        self.display.DisplayShape(self.fill.Shape(), transparency=0.8)
        self.export_stp(self.fill.Shape(), self.tempname + "_fillet.stp")

    def unfold_adjacent_faces(self, base_face=TopoDS_Face(), solid=TopoDS_Solid()):
        """Unfold all faces adjacent to the given base face onto the same plane as the base face.
           All adjacent faces will be rotated around their common edge with the base face
           to lie in the same plane as the base face.

        Args:
            base_face (TopoDS_Face): The reference face that defines the target plane
            solid (TopoDS_Solid): The solid containing all faces to be unfolded
        """
        # Get the plane of the base face (target plane for unfolding)
        base_plane = self.pln_on_face(base_face)
        base_normal = base_plane.Position().Direction()

        print(f"Base face plane: {base_plane}")
        print(
            f"Base face normal: {base_normal.X():.3f}, {base_normal.Y():.3f}, {base_normal.Z():.3f}"
        )

        # Display the base face
        self.display.DisplayShape(base_face, color="RED", transparency=0.3)

        # Find all faces adjacent to the base face
        adjacent_faces = self._find_adjacent_faces(base_face, solid)

        print(f"Found {len(adjacent_faces)} adjacent faces")

        # Unfold each adjacent face
        unfolded_faces = []
        for i, (adjacent_face, common_edge) in enumerate(adjacent_faces):
            try:
                unfolded_face = self._unfold_face_to_plane(
                    face=adjacent_face,
                    target_plane=base_plane,
                    rotation_edge=common_edge,
                    face_index=i,
                )
                unfolded_faces.append(unfolded_face)

                # Display the unfolded face
                color = (
                    self.colors[i % len(self.colors)]
                    if hasattr(self, "colors")
                    else "BLUE"
                )
                self.display.DisplayShape(unfolded_face, color=color, transparency=0.5)

            except Exception as e:
                print(f"Failed to unfold face {i}: {e}")

        return unfolded_faces

    def _find_adjacent_faces(self, base_face, solid):
        """Find all faces that share an edge with the base face.

        Returns:
            list: List of tuples (adjacent_face, common_edge)
        """
        adjacent_faces = []

        # Get all edges of the base face
        base_edges = list(TopologyExplorer(base_face).edges())

        # Iterate through all faces in the solid
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while face_explorer.More():
            current_face = face_explorer.Current()

            # Skip if it's the base face itself
            if not base_face.IsEqual(current_face):
                # Check if this face shares an edge with the base face
                common_edge = self._find_common_edge(base_face, current_face)
                if common_edge:
                    adjacent_faces.append((current_face, common_edge))

            face_explorer.Next()

        return adjacent_faces

    def _find_common_edge(self, face1, face2):
        """Find the common edge between two faces.

        Returns:
            TopoDS_Edge or None: The common edge if found, None otherwise
        """
        edges1 = list(TopologyExplorer(face1).edges())
        edges2 = list(TopologyExplorer(face2).edges())

        for edge1 in edges1:
            for edge2 in edges2:
                if edge1.IsEqual(edge2):
                    return edge1
        return None

    def _unfold_face_to_plane(self, face, target_plane, rotation_edge, face_index=0):
        """Unfold a face to lie in the target plane by rotating around the rotation edge.

        Args:
            face (TopoDS_Face): Face to unfold
            target_plane (gp_Pln): Target plane to unfold onto
            rotation_edge (TopoDS_Edge): Edge to rotate around
            face_index (int): Index for debugging/labeling

        Returns:
            TopoDS_Face: The unfolded face
        """
        # Get the current plane of the face
        current_plane = self.pln_on_face(face)
        current_normal = current_plane.Position().Direction()
        target_normal = target_plane.Position().Direction()

        # Get edge geometry for rotation axis
        edge_curve, u0, u1 = BRep_Tool.Curve(rotation_edge)
        edge_start = edge_curve.Value(u0)
        edge_end = edge_curve.Value(u1)
        edge_direction = gp_Vec(edge_start, edge_end).Normalized()
        edge_midpoint = gp_Pnt(
            (edge_start.X() + edge_end.X()) / 2,
            (edge_start.Y() + edge_end.Y()) / 2,
            (edge_start.Z() + edge_end.Z()) / 2,
        )

        # Create rotation axis
        rotation_axis = gp_Ax1(edge_midpoint, vec_to_dir(edge_direction))

        # Calculate the rotation angle to align normals
        # The angle should make the face normal align with the target plane normal
        angle = current_normal.AngleWithRef(target_normal, vec_to_dir(edge_direction))

        # Ensure we rotate in the correct direction for unfolding
        # We want the face to "open" away from the solid, not fold into it
        if abs(angle) > np.pi / 2:
            angle = np.pi - abs(angle) if angle > 0 else -(np.pi - abs(angle))

        print(f"Face {face_index}: Rotation angle = {np.rad2deg(angle):.1f}°")

        # Create transformation
        transform = gp_Trsf()
        transform.SetRotation(rotation_axis, angle)

        # Apply transformation
        location = TopLoc_Location(transform)
        unfolded_face = face.Moved(location)

        # Display rotation information
        self.display.DisplayShape(rotation_edge, color="GREEN")
        self.display.DisplayShape(edge_midpoint)
        self.context.append(
            self.display.DisplayMessage(
                edge_midpoint, f"Face{face_index}: {np.rad2deg(angle):.1f}°"
            )
        )

        return unfolded_face

    def face_rotate(self, face=TopoDS_Face(), axs=gp_Ax3(), flg=1):
        """face rotate

        Args:
            face (_type_, optional): TopDS_Face with TopDS_Edge in common with the TopDS_Face (self.tmp_face) to be used as a reference
            axs (_type_, optional): gp_Ax1 defining the common TopDS_Edge of the "face" and the TopDS_Face (self.tmp_face) to be referenced

        Returns:
            _type_: _description_
        """
        # Axis of rotated face
        plan = self.pln_on_face(face)
        plan_axs = plan.Position()

        rim_axs = axs.Ax2()
        rim_circl = Geom_Circle(rim_axs, 10)
        rim_u0, rim_u1 = rim_circl.FirstParameter(), rim_circl.LastParameter()
        rim_p0 = rim_circl.Value(rim_u0)

        pln_angle = self.fix_axis.Direction().Angle(plan_axs.Direction())
        pln_angle = self.fix_axis.Direction().AngleWithRef(
            plan_axs.Direction(), axs.Direction()
        )
        # print("Angle", np.rad2deg(pln_angle))

        ang = -pln_angle % np.pi

        if flg >= 0:
            ang = np.pi - pln_angle
        else:
            ang = -pln_angle % np.pi
        print("Angle", np.rad2deg(pln_angle), np.rad2deg(ang))

        rim_u2 = ang
        rim_p2 = rim_circl.Value(rim_u2)
        rim_angle = Geom_TrimmedCurve(rim_circl, rim_u2, rim_u0)

        trf = gp_Trsf()
        trf.SetRotation(axs.Axis(), ang)
        loc_face = TopLoc_Location(trf)
        new_face = face.Moved(loc_face)
        self.display.DisplayShape(new_face, transparency=0.5)
        self.display.DisplayShape(rim_angle)
        self.display.DisplayShape(plan_axs.Location())
        self.context.append(
            self.display.DisplayVector(
                dir_to_vec(plan_axs.Direction()).Scaled(5), plan_axs.Location()
            )
        )
        self.context.append(
            self.display.DisplayVector(
                dir_to_vec(axs.Direction()).Scaled(5), axs.Location()
            )
        )
        self.context.append(
            self.display.DisplayVector(
                dir_to_vec(axs.XDirection()).Scaled(5), axs.Location()
            )
        )
        self.display.DisplayShape(rim_p0)
        # self.display.DisplayShape(rim_p2)
        # self.show_axs_pln(axs, scale=5)
        # self.display.DisplayMessage(rim_p0,
        #                            f"rim_p0: {np.rad2deg(rim_u0):.1f}")
        # self.display.DisplayMessage(rim_p2,
        #                            f"rim_p2: {np.rad2deg(rim_u2):.1f}")

        ag_aspect = Prs3d_DimensionAspect()
        ag_aspect.SetCommonColor(Quantity_Color(Quantity_NOC_BLACK))
        ag = PrsDim_AngleDimension(rim_p2, axs.Location(), rim_p0)
        ag.SetDimensionAspect(ag_aspect)
        # self.context.append(self.display.Context.Display(ag, True))

        return new_face

    def face_init(self, face=TopoDS_Face()):
        self.fix_face = face
        self.fix_plan = self.pln_on_face(self.fix_face)
        self.fix_axis = self.fix_plan.Position()
        self.fix_face_n = 0
        # self.show_axs_pln(self.fix_axis, scale=20, name="Fix-Face")
        self.context.append(
            self.display.DisplayVector(
                dir_to_vec(self.fix_axis.Direction()).Scaled(5),
                self.fix_axis.Location(),
            )
        )
        self.display.DisplayShape(self.fix_face, color="RED")
        self.display.DisplayShape(self.fix_axis.Location())

    def face_expand(self, face=TopoDS_Face()):
        """Expand face by rotating around common edge with base face.

        Args:
            face (TopoDS_Face): Face to be expanded
        """
        try:
            # Get plane of the face
            plan = self.pln_on_face(face)
            plan_axs = plan.Position()

            # Find common edges between this face and the base face
            find_edge = LocOpe_FindEdges(self.fix_face, face)
            find_edge.InitIterator()
            edge_n = 0

            while find_edge.More():
                i = (edge_n + self.fix_face_n) % len(self.colors)

                # Get common edge
                edge = find_edge.EdgeTo()
                line = self.prop_edge(edge)

                # Get edge curve information
                e_curve, u0, u1 = BRep_Tool.Curve(edge)
                vz = gp_Vec(0, 0, 1)  # tangent vector
                p0 = gp_Pnt(0, 0, 1)  # midpoint
                e_curve.D1((u0 + u1) / 2, p0, vz)
                e_vec = gp_Vec(e_curve.Value(u0), e_curve.Value(u1)).Normalized()

                # Display edge and label
                txt = f"Face{self.fix_face_n}-Edge{edge_n}"
                self.display.DisplayShape(edge, color=self.colors[i])
                self.context.append(self.display.DisplayMessage(p0, txt))

                # Create rotation axis
                line_axs = line.Position()
                line_axs.SetLocation(p0)

                # Calculate rotation direction
                px = gp_Vec(p0, self.fix_axis.Location()).Normalized()
                py = gp_Vec(p0, plan_axs.Location()).Normalized()
                vy = vz.Crossed(px)
                vx = vz.Crossed(vy)
                if px.Dot(vx) > 0:
                    vx.Reverse()

                line_axs = gp_Ax3(p0, vec_to_dir(vz), vec_to_dir(vx))
                line_flg = py.Dot(dir_to_vec(line_axs.YDirection()))

                print(f"Face: {self.fix_face_n}, Edge: {edge_n}")
                print(f"Edge length: {self.cal_len(edge):.2f}")
                print(f"Face area: {self.cal_are(face):.2f}")
                print(f"Direction flag: {line_flg:.3f}")

                # Rotate the face
                self.face_rotate(face, line_axs, flg=line_flg)

                find_edge.Next()
                edge_n += 1

        except Exception as e:
            print(f"Error in face_expand: {e}")
            raise

    def prop_fillet(self, sol=TopoDS_Solid()):
        self.fill = BRepFilletAPI_MakeFillet(sol)
        fce_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        print()
        print(sol, self.cal_vol(sol))
        print(sol_top.number_of_faces())

        self.face_init(fce_exp.Current())
        fce_exp.Next()
        fce_exp.Next()
        fce_exp.Next()

        face = fce_exp.Current()
        self.face_fillet(face)
        fce_exp.Next()
        self.fix_face_n += 1

    def prop_soild(self, sol=TopoDS_Solid()):
        """Property of TopoDS_Solid
           Select one TopoDS_Face as the basis for unfolding and unfold all adjacent faces.

        Args:
            sol (TopoDS_Solid): The solid to unfold
        """
        fce_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        print()
        print(sol, self.cal_vol(sol))
        print(f"Number of faces: {sol_top.number_of_faces()}")

        if sol_top.number_of_faces() < self.nfce:
            self.nfce = 1

        # Select the base face for unfolding
        base_face = fce_exp.Current()
        face_count = 1
        while fce_exp.More() and self.nfce > face_count:
            base_face = fce_exp.Current()
            face_count += 1
            fce_exp.Next()

        print(f"Using face {self.nfce} as base for unfolding")

        # Unfold all adjacent faces to the same plane as the base face
        unfolded_faces = self.unfold_adjacent_faces(base_face, sol)

        print(f"Successfully unfolded {len(unfolded_faces)} faces")
        return unfolded_faces

    def prop_solids(self):
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            self.prop_soild(sol_exp.Current())
            sol_exp.Next()

    def test_unfold_cube(self):
        """Test the unfolding functionality with a simple cube"""
        # Create a simple cube for testing
        test_cube = make_box(50, 50, 50)

        # Get the first face as the base
        face_exp = TopExp_Explorer(test_cube, TopAbs_FACE)
        base_face = face_exp.Current()

        print("Testing cube unfolding...")
        unfolded_faces = self.unfold_adjacent_faces(base_face, test_cube)

        # Display the original cube
        self.display.DisplayShape(test_cube, color="BLACK", transparency=0.8)

        return unfolded_faces

    def extract_and_unfold_clean(self, solid_index=0, face_index=0):
        """Extract a polyhedron and unfold it with clean visualization (only show selected solid and unfolded faces).

        Args:
            solid_index (int): Index of the solid to extract (0-based)
            face_index (int): Index of the face to use as base for unfolding (0-based)

        Returns:
            tuple: (extracted_solid, base_face, unfolded_faces)
        """
        print(f"=== Clean Extraction and Unfolding ===")

        # Clear any existing display
        self.display.EraseAll()

        # Get all split solids
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        solids = []

        while sol_exp.More():
            solids.append(sol_exp.Current())
            sol_exp.Next()

        if not solids:
            print("No split solids found!")
            return None, None, []

        print(f"Found {len(solids)} split solids")

        # Select the specified solid
        if solid_index >= len(solids):
            solid_index = 0
            print(f"Solid index out of range, using solid 0")

        selected_solid = solids[solid_index]
        print(f"Selected solid {solid_index}")
        print(f"Volume: {self.cal_vol(selected_solid):.2f}")

        # Get all faces of the selected solid
        face_exp = TopExp_Explorer(selected_solid, TopAbs_FACE)
        faces = []

        while face_exp.More():
            faces.append(face_exp.Current())
            face_exp.Next()

        print(f"Solid has {len(faces)} faces")

        # Select the specified face as base
        if face_index >= len(faces):
            face_index = 0
            print(f"Face index out of range, using face 0")

        base_face = faces[face_index]
        print(f"Using face {face_index} as base for unfolding")

        # Display ONLY the selected solid (semi-transparent)
        self.display.DisplayShape(selected_solid, color="LIGHTGRAY", transparency=0.6)

        # Unfold adjacent faces (this will display the unfolded faces)
        unfolded_faces = self.unfold_from_base_face(base_face, selected_solid)

        return selected_solid, base_face, unfolded_faces

    def demo_split_and_unfold_clean(
        self, num_splits=3, seed=42, solid_index=0, face_index=0
    ):
        """Clean demonstration: Split box, extract polyhedron, and unfold with minimal visualization.

        Args:
            num_splits (int): Number of random splitting planes
            seed (int): Random seed for reproducible results
            solid_index (int): Index of solid to unfold
            face_index (int): Index of face to use as base
        """
        print("=" * 60)
        print("CLEAN DEMO: Box Split → Polyhedron Extract → Face Unfold")
        print("=" * 60)

        # Step 1: Split the base box (don't display all solids)
        print(f"Step 1: Splitting base box with {num_splits} random planes")
        self.split_run(num_splits, seed)

        # Count solids but don't display them all
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        solid_count = 0
        while sol_exp.More():
            solid_count += 1
            sol_exp.Next()

        print(f"Created {solid_count} split solids")

        # Step 2: Extract and unfold specified polyhedron with clean display
        print(
            f"Step 2: Extracting solid {solid_index} and unfolding from face {face_index}"
        )

        try:
            solid, base_face, result = self.extract_and_unfold_clean(
                solid_index, face_index
            )

            if solid:
                print(f"✓ Successfully unfolded polyhedron")
                print(f"  Volume: {self.cal_vol(solid):.2f}")
                print(f"  Base face area: {self.cal_are(base_face):.2f}")

                # Show the clean result
                print("\nDisplaying clean unfolded result...")
                self.ShowOCC()

                return solid, base_face, result
            else:
                print("✗ Failed to extract polyhedron")
                return None, None, []

        except Exception as e:
            print(f"✗ Error during unfolding: {e}")
            import traceback

            traceback.print_exc()
            return None, None, []

    def unfold_from_base_face(self, base_face, solid):
        """Unfold all adjacent faces from a base face using the existing face_expand logic.

        Args:
            base_face (TopoDS_Face): The base face for unfolding
            solid (TopoDS_Solid): The solid containing the faces

        Returns:
            list: List of unfolded faces
        """
        print(f"--- Starting Face Unfolding ---")

        # Initialize the base face (using existing logic)
        self.face_init(base_face)

        # Get all faces of the solid
        face_exp = TopExp_Explorer(solid, TopAbs_FACE)
        faces = []

        while face_exp.More():
            faces.append(face_exp.Current())
            face_exp.Next()

        print(f"Processing {len(faces)} faces for unfolding")

        # Process each face (except the base face)
        unfolded_count = 0
        for face in faces:
            if not self.fix_face.IsEqual(face):
                print(f"Unfolding face {unfolded_count + 1}")
                try:
                    self.face_expand(face)
                    unfolded_count += 1
                except Exception as e:
                    print(f"Failed to unfold face: {e}")

            self.fix_face_n += 1

        print(f"Successfully processed {unfolded_count} faces")
        return unfolded_count

    def demo_split_and_unfold(self, num_splits=3, seed=42, solid_index=0, face_index=0):
        """Demonstration function: Split box, extract polyhedron, and unfold.

        Args:
            num_splits (int): Number of random splitting planes
            seed (int): Random seed for reproducible results
            solid_index (int): Index of solid to unfold
            face_index (int): Index of face to use as base
        """
        print("=" * 60)
        print("DEMONSTRATION: Box Split → Polyhedron Extract → Face Unfold")
        print("=" * 60)

        # Step 1: Split the base box
        print(f"Step 1: Splitting base box with {num_splits} random planes")
        self.split_run(num_splits, seed)

        # Show split result
        print("Displaying split result...")
        colors = ["BLUE", "RED", "GREEN", "YELLOW", "CYAN", "MAGENTA"]
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        solid_count = 0

        while sol_exp.More():
            color = colors[solid_count % len(colors)]
            self.display.DisplayShape(sol_exp.Current(), color=color, transparency=0.8)
            solid_count += 1
            sol_exp.Next()

        print(f"Created {solid_count} split solids")

        # Step 2: Extract and unfold specified polyhedron
        print(
            f"\nStep 2: Extracting solid {solid_index} and unfolding from face {face_index}"
        )

        try:
            solid, base_face, result = self.extract_and_unfold_polyhedron(
                solid_index, face_index
            )

            if solid:
                print(f"✓ Successfully unfolded polyhedron")
                print(f"  Volume: {self.cal_vol(solid):.2f}")
                print(f"  Base face area: {self.cal_are(base_face):.2f}")

                # Show the result
                print("\nDisplaying unfolded result...")
                self.ShowOCC()

                return solid, base_face, result
            else:
                print("✗ Failed to extract polyhedron")
                return None, None, []

        except Exception as e:
            print(f"✗ Error during unfolding: {e}")
            import traceback

            traceback.print_exc()
            return None, None, []

    def show_split_solid(self):
        colors = ["BLUE", "RED", "GREEN", "YELLOW", "BLACK", "WHITE"]
        num = 0
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            num += 1
            self.display.DisplayShape(
                sol_exp.Current(), color=colors[num % len(colors)], transparency=0.5
            )
            sol_exp.Next()
        self.ShowOCC()


if __name__ == "__main__":
    # シンプルなデモ実行（インタラクティブ要素なし）
    print("=" * 60)
    print("POLYHEDRON UNFOLDING DEMONSTRATION")
    print("=" * 60)

    obj = CovExp(touch=False)

    # デフォルトパラメータでクリーンな展開実行
    print("Running clean unfold demo with default parameters...")
    solid, base_face, result = obj.demo_split_and_unfold_clean(
        num_splits=3,  # 3つの平面で分割
        seed=42,  # 固定シード値
        solid_index=0,  # 最初の立体を抽出
        face_index=0,  # 最初の面を基準に展開
    )

    if solid:
        print(f"\n✓ Demonstration completed successfully!")
        print(f"Final results:")
        print(f"  - Polyhedron volume: {obj.cal_vol(solid):.2f}")
        print(f"  - Base face area: {obj.cal_are(base_face):.2f}")
        print(f"  - Processed faces: {result}")
    else:
        print(f"\n✗ Demonstration failed")

    print("\nDemo completed!")
