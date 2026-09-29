#Dihedral angle calculation
import numpy 
from numpy import array, cross, pi, dot, cos, arccos as acos
from numpy.linalg import norm
import numpy as np

class DihedralGeometryError(Exception): pass
class AngleGeometryError(Exception): pass


def calc_angle(vec1,vec2,vec3):
    if len(vec1) == 3:
        v1, v2, v3 = map(create_vector,[vec1,vec2,vec3])
    else:
        v1, v2, v3 = map(create_vector2d,[vec1,vec2,vec3])
    v12 = v2 - v1
    v23 = v2 - v3
    return acos(dot(v12, v23))



def dihedral(vec1,vec2,vec3,vec4):
    """
    Returns a float value for the dihedral angle between 
    the four vectors. They define the bond for which the 
    torsion is calculated (~) as:
    V1 - V2 ~ V3 - V4 

    The vectors vec1 .. vec4 can be array objects, lists or tuples of length 
    three containing floats. 
    For Scientific.geometry.Vector objects the behavior is different 
    on Windows and Linux. Therefore, the latter is not a featured input type 
    even though it may work.
    
    If the dihedral angle cant be calculated (because vectors are collinear),
    the function raises a DihedralGeometryError
    """    
    # create array instances.
    try:
        v1,v2,v3,v4 = vec1.to_numpy(),vec2.to_numpy(),vec3.to_numpy(),vec4.to_numpy()
    except:
        v1,v2,v3,v4 = vec1,vec2,vec3,vec4
    all_vecs = [v1,v2,v3,v4]


    # calculate vectors representing bonds
    v12 = v2-v1
    v23 = v3-v2
    v34 = v4-v3

    # calculate vectors perpendicular to the bonds
    normal1 = cross(v12,v23)
    normal2 = cross(v23,v34)

    # check for linearity
    if norm(normal1) == 0 or norm(normal2)== 0:
        raise DihedralGeometryError("Vectors are in one line; cannot calculate normals!")

    # normalize them to length 1.0
    normal1 = normal1/norm(normal1)
    normal2 = normal2/norm(normal2)

    # calculate torsion and convert to degrees
    torsion = acos(dot(normal1,normal2))*180.0/pi
#    if torsion > 180:
#        torsion = torsion - 360

#    return torsion

    # take into account the determinant
    # (the determinant is a scalar value distinguishing
    # between clockwise and counter-clockwise torsion.
    if dot(normal1,v34) >= 0:
        torsion = torsion
    else:
        torsion = 360-torsion
        if torsion == 360: torsion = 0.0
    
    if torsion > 180:
        torsion = torsion - 360
    try:
       torsion = torsion.to_numpy()
    except:
       torsion = torsion
    return torsion

