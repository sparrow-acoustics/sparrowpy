import sparapy as sp
import numpy as np
import trimesh
from sparapy import geometry as geo

def read_geometry(path):
    """Read geometry from a file.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    geometry : sparapy.geometry.Polygons
        The polygon.

    """
    faces = load_rectangular_prism_faces_as_rectangles(path)
    face_normals = calculate_face_normals(faces)

    list_polygon = []

    for i, object in enumerate(faces):
        polygon = geo.Polygon(faces[i], [0, 0, 1], face_normals[i])
        list_polygon.append(polygon)

    return list_polygon

def load_rectangular_prism_faces_as_rectangles(stl_file):
    # Load the STL file
    mesh = trimesh.load(stl_file)

    # Print mesh info for debugging
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of triangular faces: {len(mesh.faces)}")
    
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

def load_cube_faces_as_squares(stl_file):
    # Load the STL file
    mesh = trimesh.load(stl_file)

    # Print mesh info for debugging
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of triangular faces: {len(mesh.faces)}")
    
    # Get the triangular faces and vertices
    faces = mesh.faces
    vertices = mesh.vertices

    # To store the square faces
    square_faces = []
    used_faces = set()

    # Function to check if two triangles share two vertices
    def share_two_vertices(tri1, tri2):
        return len(set(tri1) & set(tri2)) == 2

    # Loop through each pair of triangles to find pairs that form squares
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
                    square_face = np.array([vertices[v] for v in combined_vertices])

                    # Sort the vertices in a way that forms a square
                    # Step 1: Find the centroid of the face
                    centroid = np.mean(square_face, axis=0)

                    # Step 2: Project vertices onto the dominant axis (ignore the axis with the least variation)
                    axis_variation = np.ptp(square_face, axis=0)  # range of values along X, Y, Z
                    dominant_axis = np.argmax(axis_variation)  # Find the axis with the largest range

                    if dominant_axis == 0:  # X is dominant, project onto YZ plane
                        square_face_2d = square_face[:, 1:]  # YZ
                        centroid_2d = centroid[1:]
                    elif dominant_axis == 1:  # Y is dominant, project onto XZ plane
                        square_face_2d = square_face[:, [0, 2]]  # XZ
                        centroid_2d = centroid[[0, 2]]
                    else:  # Z is dominant, project onto XY plane
                        square_face_2d = square_face[:, :2]  # XY
                        centroid_2d = centroid[:2]

                    # Step 3: Compute angles relative to the centroid
                    angles = np.arctan2(square_face_2d[:, 1] - centroid_2d[1],
                                        square_face_2d[:, 0] - centroid_2d[0])

                    # Step 4: Sort vertices based on their angle
                    ordered_indices = np.argsort(angles)

                    # Reorder the vertices based on the calculated angles
                    ordered_square_face = square_face[ordered_indices]

                    # Append the ordered square face to the result list
                    square_faces.append(ordered_square_face)

                    # Mark both triangles as used
                    used_faces.add(i)
                    used_faces.add(j)
                    break

    if not square_faces:
        print("No square faces were detected! Please check the STL file.")
    
    return square_faces

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