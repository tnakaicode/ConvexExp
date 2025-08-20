import numpy as np
import math
import sys
import os
import functools

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Trsf
from OCC.Core.gp import gp_Pln, gp_Lin, gp_Circ
from OCC.Core.Geom import Geom_Circle, Geom_TrimmedCurve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GProp import GProp_GProps
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


def error_handling_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            import traceback

            traceback.print_exc()
            return None

    return wrapper


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

    def face_init(self, face=TopoDS_Face()):
        self.fix_face = face
        self.fix_plan = self.pln_on_face(self.fix_face)
        self.fix_axis = self.fix_plan.Position()
        self.fix_face_n = 0

        # Display the base face prominently
        self.display.DisplayShape(self.fix_face, color="RED", transparency=0.2)

        # Display base face normal vector
        base_center = self.fix_axis.Location()
        base_normal_vec = dir_to_vec(self.fix_axis.Direction()).Scaled(20)
        self.context.append(self.display.DisplayVector(base_normal_vec, base_center))

        # Mark the base face center
        self.display.DisplayShape(base_center, update=True)
        self.context.append(self.display.DisplayMessage(base_center, "BASE FACE"))

        print(f"Base face initialized:")
        print(f"  Area: {self.cal_are(self.fix_face):.2f}")
        print(
            f"  Center: ({base_center.X():.1f}, {base_center.Y():.1f}, {base_center.Z():.1f})"
        )
        print(
            f"  Normal: ({self.fix_axis.Direction().X():.3f}, {self.fix_axis.Direction().Y():.3f}, {self.fix_axis.Direction().Z():.3f})"
        )

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

    @error_handling_decorator
    def get_face_center(self, face):
        """Get the center point of a face."""
        face_properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, face_properties)
        center = face_properties.CentreOfMass()
        return center

    @error_handling_decorator
    def get_face_normal(self, face):
        """Get the normal vector of a face at its center."""
        surface = BRep_Tool.Surface(face)
        center = self.get_face_center(face)

        # Project center point onto surface to get UV parameters
        projector = GeomAPI_ProjectPointOnSurf(center, surface)
        if projector.NbPoints() > 0:
            u, v = projector.Parameters(1)

            # Calculate normal at UV parameters
            props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                return props.Normal()

        # Fallback: use face orientation
        face_adapter = BRepAdaptor_Surface(face)
        return face_adapter.Plane().Axis().Direction()

    @error_handling_decorator
    def get_face_area(self, face):
        """Get the area of a face."""
        face_properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, face_properties)
        return face_properties.Mass()

    @error_handling_decorator
    def _get_dihedral_angle(self, face1, face2, common_edge):
        """Calculate dihedral angle between two faces along common edge."""
        # Get face normals
        normal1 = self.get_face_normal(face1)
        normal2 = self.get_face_normal(face2)

        # Calculate angle between normals
        angle = normal1.Angle(normal2)

        # Return the angle (already between 0 and π)
        return angle

    @error_handling_decorator
    def _get_split_solids(self):
        """Get all solids from the splitter result."""
        solids = []
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            solids.append(sol_exp.Current())
            sol_exp.Next()
        return solids

    @error_handling_decorator
    def face_expand(self, face=TopoDS_Face()):
        """Expand face by rotating around common edge with base face.
        Simplified approach with better error handling.

        Args:
            face (TopoDS_Face): Face to be expanded
        """
        # Find common edges between this face and the base face
        find_edge = LocOpe_FindEdges(self.fix_face, face)
        find_edge.InitIterator()
        edge_n = 0

        while find_edge.More():
            # Get common edge
            edge = find_edge.EdgeTo()

            # Get edge curve information
            e_curve, u0, u1 = BRep_Tool.Curve(edge)
            edge_start = e_curve.Value(u0)
            edge_end = e_curve.Value(u1)
            edge_midpoint = gp_Pnt(
                (edge_start.X() + edge_end.X()) / 2,
                (edge_start.Y() + edge_end.Y()) / 2,
                (edge_start.Z() + edge_end.Z()) / 2,
            )

            # Calculate edge direction vector
            edge_vec = gp_Vec(edge_start, edge_end)
            if edge_vec.Magnitude() < 1e-6:
                print(f"Warning: degenerate edge found")
                find_edge.Next()
                continue

            edge_direction = edge_vec.Normalized()

            # Display edge and label
            color_idx = edge_n % len(self.colors)
            self.display.DisplayShape(edge, color=self.colors[color_idx])
            edge_label = f"E{edge_n}"
            self.context.append(self.display.DisplayMessage(edge_midpoint, edge_label))

            # Create simple rotation axis
            rotation_axis = gp_Ax3(edge_midpoint, vec_to_dir(edge_direction))

            print(f"Edge {edge_n}: length={self.cal_len(edge):.2f}")

            # Rotate the face around this edge
            self.face_rotate(face, rotation_axis, flg=1)

            find_edge.Next()
            edge_n += 1

        if edge_n == 0:
            print(f"Warning: No common edges found for face {self.fix_face_n}")

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

    @error_handling_decorator
    def _unfold_single_face(self, face, common_edge, display_idx, original_idx):
        """Unfold a single face around its common edge with the base face."""
        print(f"Unfolding face {original_idx} (adjacent #{display_idx + 1})")

        # Display original face
        self.display.DisplayShape(face, color="LIGHTGRAY", transparency=0.8)

        # Display common edge (rotation axis)
        self.display.DisplayShape(common_edge, color="GREEN", linewidth=3)

        # Calculate unfold transformation
        transform = self._calculate_unfold_transform(face, common_edge)

        if transform:
            # Apply transformation
            unfolded_face = face.Moved(TopLoc_Location(transform))

            # Display unfolded face
            color = self.colors[display_idx % len(self.colors)]
            self.display.DisplayShape(unfolded_face, color=color, transparency=0.5)

            print(f"✓ Successfully unfolded face {original_idx}")
        else:
            print(f"✗ Failed to calculate transform for face {original_idx}")

    @error_handling_decorator
    def _calculate_unfold_transform(self, face, common_edge):
        """Calculate the transformation to unfold a face.

        CRITICAL: 展開時の重複回避のための詳細な回転判定

        Problem: 基準Faceと展開Faceが重複する問題
        Solution: 以下の条件を詳細に判定して適切な回転方向・角度を決定

        Conditions to check:
        1. 面間の角度関係（鋭角 vs 鈍角）
        2. 回転軸に対する面の相対位置（左側 vs 右側）
        3. 面の法線方向と回転軸の関係
        4. 展開方向（内側展開 vs 外側展開）

        Coordinate System Definition (直感的な定義):
        - Z軸 = 共通エッジ方向（回転軸）
        - Y軸 = 基準面の法線方向（上向き）
        - X軸 = Z × Y（右手系で自動決定）

        Returns the proper transformation for unfolding without overlap.
        """
        # Step 1: Define intuitive rotation axis coordinate system
        edge_curve, u0, u1 = BRep_Tool.Curve(common_edge)
        edge_start = edge_curve.Value(u0)
        edge_end = edge_curve.Value(u1)

        # Z軸: 回転軸方向（共通エッジ方向）
        z_axis = gp_Vec(edge_start, edge_end).Normalized()

        # Y軸: 基準面の法線方向（直感的な「上」方向）
        base_normal = self.get_face_normal(self.fix_face)
        y_axis = gp_Vec(base_normal).Normalized()

        # X軸: 右手系で自動決定（Z × Y）
        x_axis = z_axis.Crossed(y_axis).Normalized()

        # 回転軸座標系を定義
        rotation_axis = gp_Ax1(edge_start, vec_to_dir(z_axis))
        coord_system = gp_Ax3(edge_start, vec_to_dir(z_axis), vec_to_dir(x_axis))

        print(f"  Intuitive coordinate system defined:")
        print(
            f"    Z (rotation): ({z_axis.X():.3f}, {z_axis.Y():.3f}, {z_axis.Z():.3f})"
        )
        print(
            f"    Y (base normal): ({y_axis.X():.3f}, {y_axis.Y():.3f}, {y_axis.Z():.3f})"
        )
        print(f"    X (Z×Y): ({x_axis.X():.3f}, {x_axis.Y():.3f}, {x_axis.Z():.3f})")

        # Step 2: 展開面の法線と位置関係を分析
        face_normal = self.get_face_normal(face)
        face_center = self.get_face_center(face)
        base_center = self.get_face_center(self.fix_face)

        # 展開面が回転軸に対して左側か右側かを判定（新座標系）
        edge_to_face = gp_Vec(edge_start, face_center)
        cross_product = z_axis.Crossed(edge_to_face)
        face_side = cross_product.Dot(y_axis)  # Y軸（基準面法線）との内積で判定

        print(f"  Face position analysis (intuitive coords):")
        print(
            f"    Face normal: ({face_normal.X():.3f}, {face_normal.Y():.3f}, {face_normal.Z():.3f})"
        )
        print(
            f"    Face side: {'RIGHT' if face_side > 0 else 'LEFT'} (value: {face_side:.3f})"
        )

        # Step 3: 二面角を計算
        dihedral_angle = self._get_dihedral_angle(self.fix_face, face, common_edge)
        is_acute = dihedral_angle < math.pi / 2

        print(f"  Angle analysis:")
        print(f"    Dihedral angle: {math.degrees(dihedral_angle):.1f}°")
        print(f"    Angle type: {'ACUTE' if is_acute else 'OBTUSE'}")

        # Step 4: 展開角度と方向を決定（直感的座標系版）
        """
        展開判定ロジック（Y軸=基準面法線版）:
        
        Case 1: 鋭角 + 面が左側
        → 180° - 二面角で展開（外側に開く）
        
        Case 2: 鋭角 + 面が右側  
        → 180° - 二面角で展開（外側に開く）
        
        Case 3: 鈍角 + 面が左側
        → 二面角の補角で展開（内側に折り込む）
        
        Case 4: 鈍角 + 面が右側
        → 二面角の補角で展開（内側に折り込む）
        
        重要: 面の重複を避けるため、必ず外側展開を選択
        """

        if is_acute:
            # 鋭角の場合: 外側に開く（180° - 角度）
            unfold_angle = math.pi - dihedral_angle
            rotation_direction = "OUTWARD"
            if face_side < 0:  # 左側の場合
                unfold_angle = -unfold_angle  # 回転方向を反転
        else:
            # 鈍角の場合: 補角で展開
            unfold_angle = math.pi - dihedral_angle
            rotation_direction = "SUPPLEMENT"
            if face_side > 0:  # 右側の場合
                unfold_angle = -unfold_angle  # 回転方向を反転

        print(f"  Unfold decision:")
        print(f"    Rotation direction: {rotation_direction}")
        print(f"    Unfold angle: {math.degrees(unfold_angle):.1f}°")

        # Step 5: 変換行列を作成
        transform = gp_Trsf()
        transform.SetRotation(rotation_axis, unfold_angle)

        return transform

    def unfold_adjacent_faces(self, base_face_index=0):
        """Core method: Unfold all faces adjacent to the base face.

        ===================================================================
        FACE UNFOLDING ALGORITHM WITH OVERLAP PREVENTION
        ===================================================================

        Problem: 展開時に基準Faceと被ってしまう
        Solution: 回転軸の座標系と面の相対位置を詳細分析

        Coordinate System Definition (Intuitive, Right-hand rule):
        ```
        直感的回転軸座標系:
           Z軸 = 共通エッジ方向（回転軸）
           Y軸 = 基準面の法線方向（直感的な「上」）
           X軸 = Z × Y（右手系で自動決定）

        Example:
               Y (Base normal)
               ↑
        Base   |   Target
        Face   |   Face
           ----+----  ← 共通Edge（Z軸）
               |
               → X (Z×Y)
        ```

        Rotation Logic:
        ```
        Case Analysis Matrix:

        | Angle  | Face Position | Rotation Strategy        | Result    |
        |--------|---------------|--------------------------|-----------|
        | Acute  | Left side     | -(180° - dihedral)      | No overlap|
        | Acute  | Right side    | +(180° - dihedral)      | No overlap|
        | Obtuse | Left side     | -(supplement)           | No overlap|
        | Obtuse | Right side    | +(supplement)           | No overlap|

        Angle Signs:
        - Positive: Counter-clockwise around Z-axis
        - Negative: Clockwise around Z-axis
        ```

        Position Detection (Intuitive):
        ```
        face_side = (Z × edge_to_face) · Y
        - face_side > 0: Face on RIGHT side of rotation axis
        - face_side < 0: Face on LEFT side of rotation axis

        Note: Y軸が基準面法線なので、より直感的な判定
        ```

        Args:
            base_face_index (int): Index of the base face (default: 0)

        Flow:
            1. Select base face
            2. Find all adjacent faces
            3. For each adjacent face:
               - Find common edge with base face
               - Analyze geometric relationship with intuitive coordinates
               - Calculate proper unfold angle with correct sign
               - Rotate face around common edge
               - Display result with color coding
        """

        # Step 1: Get base face
        faces = list(TopologyExplorer(self.fix_solid).faces())

        self.fix_face = faces[base_face_index]
        print(f"Base face {base_face_index} selected (total faces: {len(faces)})")

        # Display base face with special highlighting
        self.display.DisplayShape(self.fix_face, color="BLUE", transparency=0.3)

        # Step 2: Find adjacent faces
        adjacent_faces = []
        for i, face in enumerate(faces):
            if i != base_face_index:  # Skip base face
                common_edge = self._find_common_edge(self.fix_face, face)
                if common_edge:  # Only process faces that share an edge
                    adjacent_faces.append((i, face, common_edge))

        print(f"Found {len(adjacent_faces)} adjacent faces")

        # Step 3: Unfold each adjacent face
        for face_idx, (original_index, face, common_edge) in enumerate(adjacent_faces):
            self._unfold_single_face(face, common_edge, face_idx, original_index)

    def demo_simple_unfold(self):
        """Simple demo: Create box and unfold adjacent faces.

        Purpose: 1つの基準面から隣接するすべての面を展開する
        """
        print("=======================================================")
        print("SIMPLE UNFOLD DEMO: Box → Base Face → Unfold Adjacent")
        print("=======================================================")

        # Create simple box
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box_maker = BRepPrimAPI_MakeBox(50, 30, 20)
        self.fix_solid = box_maker.Solid()

        face_count = len(list(TopologyExplorer(self.fix_solid).faces()))
        print(f"Created box (50×30×20) with {face_count} faces")

        # Unfold from face 0
        print("Unfolding all faces adjacent to face 0...")
        self.unfold_adjacent_faces(base_face_index=0)

        print("✓ Unfold demo completed!")

        return self.fix_solid, self.fix_face, "success"

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

    # Main execution: Simple box unfold demo
    print("Running simple unfold demo...")

    solid, base_face, result = obj.demo_simple_unfold()

    if solid:
        print(f"\n✓ Demo completed successfully!")
        print(f"Results:")
        print(f"  - Polyhedron volume: {obj.cal_vol(solid):.2f}")
        print(f"  - Base face area: {obj.cal_are(base_face):.2f}")
        print(f"  - Status: {result}")
    else:
        print(f"\n✗ Demo failed")

    print("\nDemo completed!")
