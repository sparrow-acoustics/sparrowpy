"""Module for the geometry of the room and the environment."""
import matplotlib.axes
import numpy as np


class Polygon():
    """Polygon constructed from greater than two points.

    Only convex polygons are allowed!
    Order of points is of course important!
    """

    up_vector: np.ndarray
    pts: np.ndarray

    def __init__(
            self, points: np.ndarray, up_vector: np.ndarray,
            normal: np.ndarray) -> None:
        """Create a Polygon from points, up_vector and normal.

        Parameters
        ----------
        points : np.ndarray
            Cartesian edge points of the polygon
        up_vector : np.ndarray
            Cartesian up vector of the polygon
        normal : np.ndarray
            Cartesian up normal of the polygon

        """
        self.pts = np.array(points, dtype=float)
        normal = np.array(normal, dtype=float)
        self.form_factors = []
        # check if points are in one plane
        assert self.pts.shape[1] >= 3, \
            'You need at least 3 points to build a Polygon'
        if self.n_points > 3:
            x_0 = np.array(self.pts[0])
            for i in range(1, self.n_points-2):
                # the determinant of the vectors (volume) must always be 0
                x_i = np.array(self.pts[i])
                x_i1 = np.array(self.pts[i+1])
                x_i2 = np.array(self.pts[i+2])
                det = np.linalg.det([x_0-x_i, x_0-x_i1, x_0-x_i2])
                assert _cmp_floats(det, 0.0), \
                    'Points must be in a plane to create a Polygon'
        self.up_vector = _norm(np.array(up_vector, dtype=float))
        vec1 = np.array(self.pts[0])-np.array(self.pts[1])
        vec2 = np.array(self.pts[0])-np.array(self.pts[2])
        calc_normal = _norm(np.cross(vec1, vec2))
        assert all(np.cross(normal, calc_normal) == 0), \
            'The normal vector is not perpendicular to the polygon'
        self._normal = normal

    def to_dict(self):
        """Convert this object to dictionary. Used for read write."""
        return {
            'up_vector': self.up_vector.tolist(),
            'pts': self.pts.tolist(),
            'normal': self._normal.tolist(),
        }

    @classmethod
    def from_dict(cls, input_dict):
        """Create an object from a dictionary. Used for read write."""
        return cls(
            input_dict['pts'], input_dict['up_vector'], input_dict['normal'])

    @property
    def normal(self) -> np.ndarray:
        """Return the normal vector of the polygon."""
        return self._normal

    @property
    def size(self) -> np.ndarray:
        """Return the size in (lxmxn) of the polygon."""
        vec1 = np.array(self.pts[0])-np.array(self.pts[1])
        vec2 = np.array(self.pts[1])-np.array(self.pts[2])
        size = np.abs(vec1-vec2)
        return size

    @property
    def area(self) -> np.ndarray:
        """Return the area in m^2 of the polygon.
        supports all convex polygons and some concave polygons.
        """

        area = 0

        if len(self.pts) == 3:
            area_pts = np.array([self.pts])

        elif len(self.pts) == 4:
            area_pts = np.array([
                self.pts[0:3],[self.pts[2],self.pts[3],self.pts[0]] ])

        else:
            # slow, can be optimized
            area_pts = np.empty((self.pts.shape[0],3,3))

            for i in range(area_pts.shape[0]):
                area_pts[i] = np.array([
                    self.pts[i%self.pts.shape[0]],
                    self.pts[(i+1)%self.pts.shape[0]],
                    self.center])

        for tri in area_pts:
            area  +=  .5*np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))

        return area

    @property
    def center(self) -> np.ndarray:
        """Return the center coordinates of the polygon."""
        return np.sum(self.pts, axis=0) / self.n_points

    @property
    def n_points(self) -> int:
        """Return the number of points of the polygon."""
        return self.pts.shape[0]

    def on_surface(self, point: np.ndarray) -> bool:
        """Return if a point is on the surface of the polygon.

        Returns True if the point is on the polygon's
        surface and false otherwise.
        """
        n = self.pts.shape[0]
        sum_angle = 0
        p = point

        for i in range(n):
            v1 = np.array(self.pts[i]) - p
            v2 = np.array(self.pts[(i+1) % n]) - p

            m1 = _magnitude(v1)
            m2 = _magnitude(v2)

            if _cmp_floats(m1*m2, 0.):
                return True  # point is one of the nodes
            else:
                cos_theta = np.dot(v1, v2)/(m1*m2)
            sum_angle = sum_angle + np.arccos(cos_theta)
        return _cmp_floats(sum_angle, 2*np.pi)

    def intersection(
            self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Return a intersection point with a ray and the polygon.

        Parameters
        ----------
        origin : np.ndarray
            origin of the incoming wave
        direction : np.ndarray
            direction of the incoming wave

        Returns
        -------
        np.ndarray
            intersection point, if it hit, otherwise None

        """
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        n = self.normal

        # Ray is parallel to the polygon
        if _cmp_floats(np.dot(direction, n), 0.):
            return None

        t = 1/(np.dot(direction, n)) * \
            (np.dot(n, self.pts[0]) -
             np.dot(n, origin))

        # Intersection point is behind the ray
        if t <= 0.0:
            return None

        # Calculate intersection point
        point = np.array(origin) + t*np.array(direction)

        # Check if intersection point is really in the polygon or only on
        # the (infinite) plane
        if self.on_surface(point):
            return point

        return None
# I changed this part
    def plot(self, ax: matplotlib.axes.Axes = None, color=None):
        """Plot the polygon.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            _description_, by default None
        color : _type_, optional
            _description_, by default None

        """
        points = self.pts.T
        points = np.concatenate((points, points), axis=1)
        # plot wall
        self.plot_point(ax, color)

    def plot_point(self, ax: matplotlib.axes.Axes = None, color=None):
        """Plot the polygon points."""
        points = self.pts.T
        points = np.concatenate((points, points), axis=1)
        # plot wall
        ax.plot(points[0], points[1], points[2], color=color)

    def plot_view_up(self, ax: matplotlib.axes.Axes = None):
        """Plot the view and up vector of the polygon."""
        ax.quiver(
            self.center[0], self.center[1], self.center[2],
            self.normal[0]*2, self.normal[1]*2, self.normal[2]*2,
            color='red', label='View vector')
        ax.quiver(
            self.center[0], self.center[1], self.center[2],
            self.up_vector[0]*2, self.up_vector[1]*2, self.up_vector[2]*2,
            color='blue', label='up vector')


def _cmp_floats(a, b, atol=1e-12):
    return abs(a-b) < atol


def _magnitude(vector):
    return np.sqrt(np.dot(np.array(vector), np.array(vector)))


def _norm(vector):
    return np.array(vector)/_magnitude(np.array(vector))
