import math


def calc_unfold_angle(dihedral_angle, sign1, sign2):
    # 1. sign1 < 0 の場合は補角（符号はそのまま）
    if sign1 < 0:
        angle = math.copysign(math.pi - abs(dihedral_angle), dihedral_angle)
    else:
        angle = dihedral_angle
    # 2. 回転方向
    angle = angle * sign2
    # 3. ±180°以内に正規化
    if angle > math.pi:
        angle -= 2 * math.pi
    if angle < -math.pi:
        angle += 2 * math.pi
    return angle


# | Face   | dihedral_angle | sign1 | sign2 | 望ましい回転角度 |
# | ------ | -------------: | ----: | ----: | ---------------: |
# | BLUE1  |        -109.2° |  -1.0 |   1.0 |          +109.2° |
# | RED    |          22.1° |   1.0 |   1.0 |           +22.1° |
# | GREEN  |          72.4° |   1.0 |  -1.0 |           -72.4° |
# | YELLOW |          67.5° |   1.0 |  -1.0 |           -67.5° |
# | BLACK  |        -109.2° |  -1.0 |   1.0 |          +109.2° |

for dihedral_angle, sign1, sign2, ans in [
    [-109.2, -1.0, 1.0, +109.2],
    [22.1, 1.0, 1.0, +22.1],
    [72.4, 1.0, -1.0, -72.4],
    [67.5, 1.0, -1.0, -67.5],
    [-109.2, -1.0, 1.0, +109.2],
]:
    angle = calc_unfold_angle(math.radians(dihedral_angle), sign1, sign2)
    print(
        f"dihedral_angle={dihedral_angle:>9.1f}°, sign1={sign1:>4.1f}, sign2={sign2:>4.1f} => "
        f"angle={math.degrees(angle):>9.1f}° (expected={ans:>9.1f}°)"
    )
