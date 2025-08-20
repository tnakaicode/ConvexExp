# Convex polyhedron Expansion

* ref
  * <https://github.com/tpaviot/pythonocc-core/issues/554>

* Calculate Mass
  * brepgprop_LinearProperties(TopoDS_Shape, GProp_GProps)
  * GProp_GProps.Mass()
* Calculate Volume
  * brepgprop_VolumeProperties(TopoDS_Shape, GProp_GProps)
  * GProp_GProps.Mass()

## Calculate Mass

Computes the linear global properties of the shape S
i.e.
the global properties induced by each edge of the shape S, and brings them together with the global properties still retained by the framework LProps.

If the current system of LProps was empty,
its global properties become equal to the linear global properties of S.

For this computation no linear density is attached to the edges.
for example,
the added mass corresponds to the sum of the lengths of the edges of S.

The density of the composed systems,
i.e.
that of each component of the current system of LProps, and that of S which is considered to be equal to 1, must be coherent.

Note:
that this coherence cannot be checked.
You are advised to use a separate framework for each density, and then to bring these frameworks together into a global one.
The point relative to which the inertia of the system is computed is the  reference point of the framework LProps.

Note:
if your programming ensures that the framework LProps retains only linear global properties
(brought together for example, by the function LinearProperties)
for objects the density of which is equal to 1(or is not defined),
the function Mass will return the total length of edges of the system analysed by LProps.

Warning
No check is performed to verify that the shape S retains truly linear properties.
If S is simply a vertex, it is not considered to present any additional global properties.

## Calculate Volume

Updates GProp_GProps with the shape, that contains its pricipal properties.
The volume properties of all the FORWARD and REVERSED faces in the shape are computed.
If OnlyClosed is True then computed faces must belong to closed Shells.
Adaptive 2D Gauss integration is used.

Parameter Eps sets maximal relative error of computed mass (volume) for each face.
Error is calculated as Abs((M(i+1)-M(i))/M(i+1)),
M(i+1) and M(i) are values for two successive steps of adaptive integration.

Method returns estimation of relative error reached for whole shape.

WARNING:
if Eps > 0.001 algorithm performs non-adaptive integration.

GK
Parameter IsUseSpan says
if it is necessary to define spans on a face.

This option has an effect only for BSpline faces.

Parameter Eps sets maximal relative error of computed property for each face.
Error is delivered by the adaptive Gauss-Kronrod method of integral computation
that is used for properties computation.
Method returns estimation of relative error reached for whole shape.
Returns negative value if the computation is failed.

### 現状の出力

| Face   | dihedral_angle | sign1 | sign2 | Chosen (現状) | 実際の回転 |   望ましい回転角度 | 結果・問題点                                |
| ------ | -------------: | ----: | ----: | ------------: | ---------: | -----------------: | ------------------------------------------- |
| BLUE1  |        -109.2° |  -1.0 |   1.0 |         70.8° |     +70.8° |            +109.2° | fix_baseと同一平面にならない（+109.2°必要） |
| RED    |          22.1° |   1.0 |   1.0 |         22.1° |     +22.1° |             +22.1° | OK                                          |
| GREEN  |          72.4° |   1.0 |  -1.0 |        -72.4° |     -72.4° |             -72.4° | OK                                          |
| YELLOW |          67.5° |   1.0 |  -1.0 |        -67.5° |     -67.5° | -247.5° or +112.5° | もう180度回転したい（-247.5° or +112.5°）   |
| BLACK  |        -109.2° |  -1.0 |   1.0 |         70.8° |     +70.8° |            +109.2° | fix_baseと同一平面にならない（+109.2°必要） |

| Face   | dihedral_angle | sign1 | sign2 | Chosen (現状) | 実際の回転 | 望ましい回転角度 | 結果・問題点                         |
| ------ | -------------: | ----: | ----: | ------------: | ---------: | ---------------: | ------------------------------------ |
| BLUE1  |        -109.2° |  -1.0 |   1.0 |       -250.8° |    -250.8° |          -250.8° | OK                                   |
| RED    |          22.1° |   1.0 |   1.0 |        202.1° |    +202.1° |           +22.1° | もう180度回転したい（+22.1°が正解）  |
| GREEN  |          72.4° |   1.0 |  -1.0 |       -252.4° |    -252.4° |           -72.4° | もう180度回転したい（-72.4°が正解）  |
| YELLOW |          67.5° |   1.0 |  -1.0 |       -247.5° |    -247.5° |          -247.5° | OK                                   |
| BLACK  |        -109.2° |  -1.0 |   1.0 |       -250.8° |    -250.8° |          +109.2° | もう180度回転したい（+109.2°が正解） |

| Face   | dihedral_angle | sign1 | sign2 | 望ましい回転角度 |
| ------ | -------------: | ----: | ----: | ---------------: |
| BLUE1  |        -109.2° |  -1.0 |   1.0 |          +109.2° |
| RED    |          22.1° |   1.0 |   1.0 |           +22.1° |
| GREEN  |          72.4° |   1.0 |  -1.0 |           -72.4° |
| YELLOW |          67.5° |   1.0 |  -1.0 |           -67.5° |
| BLACK  |        -109.2° |  -1.0 |   1.0 |          +109.2° |
