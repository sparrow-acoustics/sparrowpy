"""Analytical solution for form factors for testing."""
import numpy as np

def perpendicular_patch_coincidentline(w, h, length):
    """
    Returns the exact form factor solution of two rectangular patches,
    The patches are orthogonal and share a side of length l
    w and h correspond to the width of each of the patches.
    """

    x = w/length
    y = h/length

    z = np.sqrt(x**2 + y**2)

    k = 1/(np.pi * y)

    a = x*np.arctan(1/x)
    b = y*np.arctan(1/y)
    c = -z*np.arctan(1/z)

    dk = 1/4
    da = (1+x**2)*(1+y**2)/(1+z**2)
    db = ( x**2 * (1+z**2) / (z**2 * (1 + x**2)) )**(x**2)
    dc = ( y**2 * (1+z**2) / (z**2 * (1 + y**2)) )**(y**2)

    d = dk*np.log(da*db*dc)

    return k * ( a + b + c + d )


def perpendicular_patch_coincidentpoint(aa,bb,cc,dd):
    """
    Returns the exact form factor solution of two rectangular patches,
    The patches are orthogonal and share a vertex.
    """

    a = aa/(bb+dd)
    b = cc/(bb+dd)
    c = np.sqrt(a**2+b**2)
    d = aa/bb
    e = cc/bb
    f = np.sqrt(d**2+e**2)
    x = aa/dd
    y = cc/dd
    z = np.sqrt(x**2+y**2)

    k = 1/(8*np.pi*bb*cc)

    out  = (bb+dd)**2 * np.log(((c**2 + a**2*b**2 + 1)/(c**2 +1)) * ((
        b**2 + b**2*c**2)/(c**2 +c**2*b**2))**(b**2) * ((
            a**2 + a**2*c**2)/(c**2 +c**2*a**2))**(a**2))

    out += bb**2 * np.log(((f**2 + 1)/(f**2 + d**2*e**2 + 1)) * ((
        f**2 + f**2*e**2)/(e**2 + e**2*f**2))**(e**2) * ((
            f**2 + f**2*d**2)/(d**2 + d**2*f**2))**(d**2))

    out += dd**2 * np.log(((z**2 + 1)/(z**2 + x**2*y**2 + 1)) * ((
        z**2 + z**2*y**2)/(y**2 + y**2*z**2))**(y**2) * ((
            z**2 + z**2*x**2)/(x**2 + x**2*z**2))**(x**2))

    out += 4 * (bb+dd)**2 * (
        a*np.arctan(1/a) + b*np.arctan(1/b) - c*np.arctan(1/c))

    out -= 4 * bb**2 * (
        d*np.arctan(1/d) + e*np.arctan(1/e) - f*np.arctan(1/f)  )

    out -= 4 * dd**2 * (
        x*np.arctan(1/x) + y*np.arctan(1/y) - z*np.arctan(1/z))

    return out * k


def perpendicular_patch_floating(aa,bb,cc,dd,ee):
    """
    Returns the exact form factor solution of two rectangular patches.
    The patches are orthogonal.
    """

    a = (dd+ee)/bb
    b = (aa+cc)/bb
    c = np.sqrt(aa**2 + bb**2)
    d = dd/bb
    y = cc/bb
    f = np.sqrt(d**2+b**2)
    j = np.sqrt(d**2+y**2)
    z = np.sqrt(a**2 + y**2)

    out = np.log(  ( (1+f**2)*(1+z**2) / ( (1+c**2)*(1+j**2) ) ) * ((
        z**2 + z**2 * c**2 )/(  c**2 + c**2 * z**2  ))**(a**2)  * ((
            f**2 + f**2 * c**2 )/(  c**2 + c**2 * f**2  ))**(b**2) * ((
                f**2 + f**2 * j**2 )/(  j**2 + j**2 * f**2  ))**(d**2) * ((
                    z**2 + z**2 * j**2 )/(  j**2 + j**2 * z**2  ))**(y**2)  )
    out += 4 * (
        f*np.arctan(1/f) + z*np.arctan(1/z) - c*np.arctan(
            1/c) - j*np.arctan(1/j))

    return out*bb/(4*aa*np.pi)

def parallel_patches(a,b,c):
    """
    Returns the exact form factor solution of parallel patches
    facing each other both patches have a length a and width b.
    They have a distance c from each other.
    """

    x = a/c

    y = b/c

    kk = 2 / ( np.pi * x * y )

    g = np.log( np.sqrt(( 1 + x**2 ) * ( 1 + y**2 ) / ( 1 + x**2 + y**2 )) )
    h = x * np.sqrt( 1 + y**2 ) * np.arctan( x / np.sqrt( 1 + y**2 ) )
    i = y * np.sqrt( 1 + x**2 ) * np.arctan( y / np.sqrt( 1 + x**2 ) )
    j = - x * np.arctan( x )
    k = - y * np.arctan( y )

    return kk * ( g + h + i + j + k)
