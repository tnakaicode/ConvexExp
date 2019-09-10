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
