import sparapy as sp
import numpy as np
import trimesh
from sparapy import geometry as geo
from scipy.spatial import ConvexHull
import matplotlib.tri as mtri

def read_geometry(path, shape):
    """Reads stl file and converts it into a list of polygons. Faces can be read as either rectangles or triangles.   

    Parameters
    ----------
    path : str
        path of stl file
    shape : str
        shape in which the faces are read. Rectangle or triangle

    Returns
    -------
    Polygon list
        a list of Polygon objects 

    """
    if shape == 'triangle':
        faces = load_triangular_faces(path)
    elif shape == 'rectangle':
        faces = load_rectangular_prism_faces_as_rectangles(path)
    else:
        print("Shape not recognized!")

    face_normals = calculate_face_normals(faces)

    list_polygon = []

    for i, object in enumerate(faces):

        if np.all(face_normals[i] == np.abs([1, 0, 0])):
            up_vector = [0, 0, 1]
        else:
            up_vector = [1, 0, 0]

        polygon = geo.Polygon(faces[i], up_vector, face_normals[i])
        list_polygon.append(polygon)

    return list_polygon


def load_rectangular_prism_faces_as_rectangles(stl_file):
    """Reads the faces of the structure in the stl file as rectangles. Default faces read using trimesh.faces are triangles. This functions combines the triangular faces into rectangles.    

    Parameters
    ----------
    stl_file : str
        path of stl file

    Returns
    -------
    faces list
        a list of faces using trimesh.faces 

    """
    # Load the STL file
    mesh = trimesh.load(stl_file)

    # Get the triangular faces and vertices
    faces = mesh.faces
    vertices = mesh.vertices

    # To store the rectangular faces
    rectangle_faces = []
    used_faces = set()

    # Function to check if two triangles share two vertices
    def share_two_vertices(tri1, tri2):
        return len(set(tri1) & set(tri2)) == 2

    # Loop through each pair of triangles to find pairs that form rectangles
    for i, tri1 in enumerate(faces):
        if i in used_faces:
            continue
        for j, tri2 in enumerate(faces):
            if i != j and j not in used_faces and share_two_vertices(tri1, tri2):
                # Combine the two triangles
                combined_vertices = list(set(tri1) | set(tri2))

                # We should have exactly 4 unique vertices now
                if len(combined_vertices) == 4:
                    # Get the 3D coordinates of the vertices
                    rectangle_face = np.array([vertices[v] for v in combined_vertices])

                    # Step 1: Find the centroid of the face
                    centroid = np.mean(rectangle_face, axis=0)

                    # Step 2: Determine the dominant plane by checking which axis has the smallest range
                    axis_variation = np.ptp(rectangle_face, axis=0)
                    dominant_axis = np.argmin(axis_variation)  # The axis with the least variation

                    # Step 3: Project the vertices onto the 2D plane that ignores the dominant axis
                    if dominant_axis == 0:  # X is constant, use YZ plane
                        projected_vertices = rectangle_face[:, 1:]  # Use YZ plane
                        projected_centroid = centroid[1:]
                    elif dominant_axis == 1:  # Y is constant, use XZ plane
                        projected_vertices = rectangle_face[:, [0, 2]]  # Use XZ plane
                        projected_centroid = centroid[[0, 2]]
                    else:  # Z is constant, use XY plane
                        projected_vertices = rectangle_face[:, :2]  # Use XY plane
                        projected_centroid = centroid[:2]

                    # Step 4: Compute the angle of each vertex relative to the centroid
                    angles = np.arctan2(projected_vertices[:, 1] - projected_centroid[1],
                                        projected_vertices[:, 0] - projected_centroid[0])

                    # Step 5: Sort vertices based on the angle, ensuring proper rectangle ordering
                    ordered_indices = np.argsort(angles)
                    ordered_rectangle_face = rectangle_face[ordered_indices]

                    # Append the ordered rectangle face to the result list
                    rectangle_faces.append(ordered_rectangle_face)

                    # Mark both triangles as used
                    used_faces.add(i)
                    used_faces.add(j)
                    break

    if not rectangle_faces:
        print("No rectangular faces were detected! Please check the STL file.")
    
    return rectangle_faces


def load_triangular_faces(stl_file):
    """Reads the faces of the structure in the stl file as triangles.

    Parameters
    ----------
    stl_file : str
        path of stl file

    Returns
    -------
    faces list
        a list of faces using trimesh.faces 

    """    
    # Load the STL file
    mesh = trimesh.load(stl_file)
    
    # Get the triangular faces and vertices
    faces = mesh.faces
    vertices = mesh.vertices

    # To store the triangular faces
    triangular_faces = []

    # Loop through each triangular face
    for i, tri in enumerate(faces):
        # Get the 3D coordinates of the vertices that make up the triangle
        triangle_face = np.array([vertices[v] for v in tri])
        
        # Optionally, you can compute any additional data such as centroid or normals
        # Example: Centroid of the triangular face
        centroid = np.mean(triangle_face, axis=0)
        
        # Append the triangle face to the list
        triangular_faces.append(triangle_face)

    if not triangular_faces:
        print("No triangular faces were detected! Please check the STL file.")
    
    return triangular_faces


def calculate_face_normals(square_faces):
    
    normals = []

    for face in square_faces:
        # Choose two edges from the face
        edge1 = face[1] - face[0]  # Vector from vertex 0 to vertex 1
        edge2 = face[2] - face[0]  # Vector from vertex 0 to vertex 2

        # Compute the normal as the cross product of edge1 and edge2
        normal = np.cross(edge1, edge2)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Append the normal to the list
        normals.append(normal)

    return np.array(normals)

