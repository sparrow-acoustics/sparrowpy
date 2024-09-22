# %%
# imports

import numpy as np
import trimesh
from sparapy import geometry as geo
import matplotlib.pyplot as plt

# %%
# load stl file

mesh = trimesh.load('models/shoebox.stl')

# %%
# basic info and visualize mesh

print(mesh)
mesh.show()
# %%
# new vertices of rectangular faces

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

# Usage example
stl_file = 'models/shoebox.stl'  # Replace with the correct path to your STL file
faces = load_cube_faces_as_squares(stl_file)

for i, face in enumerate(faces):
    print(f"Face {i}: {face}")

# %%
# create polygon

face0 = geo.Polygon(rectangles[0],[0, 1, 0], [0, 0, 1])
face1 = geo.Polygon(rectangles[1],[0, 1, 0], [0, 0, 1])
face2 = geo.Polygon(rectangles[2],[0, 1, 0], [0, 0, 1])
face3 = geo.Polygon(rectangles[3],[0, 1, 0], [0, 0, 1])
face4 = geo.Polygon(rectangles[4],[0, 1, 0], [0, 0, 1])
face5 = geo.Polygon(rectangles[5],[0, 1, 0], [0, 0, 1])

list_polygon = [face0, face1, face2, face3, face4, face5]

# %%
# plot polygon
source = [0.5, 0.5, 0.5]
receiver = [0.25, 0.25, 0.25]
speed_of_sound = 343
env = geo.Environment(
    list_polygon,
    source, receiver, speed_of_sound)
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
plt.tick_params(axis='both',which='both', labelsize=8)
env.plot(ax)
ax.set_aspect('equal', adjustable='box')
ax.locator_params(axis='x', nbins=8)
ax.locator_params(axis='y', nbins=4)
ax.locator_params(axis='z', nbins=4)
ax.legend()


# %%
