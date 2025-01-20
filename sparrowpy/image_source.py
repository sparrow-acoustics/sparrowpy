import numpy as np
from sparrowpy import is_obj as ob


def get_walls_hesse(RoomX,RoomY,RoomZ):
    WallsHesse = np.array([
    [-1, 0,  0, 0],
    [1,  0,  0, RoomX],
    [0, -1,  0, 0],
    [0,  1,  0, RoomY],
    [0,  0, -1, 0],
    [0,  0,  1, RoomZ],
    ])
    return WallsHesse


def calculate_image_sources(WallsHesse, SourcePos, MaxOrder):
    n_walls = WallsHesse.shape[0]
    n_IS = int(1 + n_walls / (n_walls - 2) * ((n_walls - 1) ** MaxOrder - 1))

    ISList = [ob.ISObj() for object in range(1,n_IS+1)]


    ISList[0].Position = SourcePos
    ISList[0].Order = 0


    list_end = 0


    for order in range(1, MaxOrder + 1):
        for iISID in range(list_end + 1):
            if ISList[iISID].Order == order - 1:
                for iWalls in range(1,n_walls + 1):


                    n0 = WallsHesse[iWalls-1, 0:3]
                    p = ISList[iISID].Position
                    d = WallsHesse[iWalls-1, 3]
                    t0 = d - np.dot(n0, p)

                    if order == 1:
                        wallCriterium = True
                    else:
                            wallCriterium = (iWalls != ISList[iISID].Walls[-1])

                    if wallCriterium and (t0 > 0):


                        ISList[list_end + 1].Position = p + 2 * t0 * n0
                        ISList[list_end + 1].Walls = ISList[iISID].Walls+[iWalls]
                        ISList[list_end + 1].Order = order


                        list_end += 1

    # this was changed in order to cut the list


    ISList = ISList[:list_end + 1]
    return ISList






def get_corners_of_walls_hesse(Walls_Hesse):
    """
    Compute the corners of walls based on their Hesse normal form representation.
   
    Args:
    - walls_hesse (np.ndarray): An (n_walls, 4) array where each row represents a wall
                                in Hesse normal form [n0x, n0y, n0z, d].
   
    Returns
    -------
    - np.ndarray: An array of corner points.
    """
    Corners = []


    for m in range(Walls_Hesse.shape[0] - 2):
        wall_m = Walls_Hesse[m, 0:4]
        for n in range(m+1,Walls_Hesse.shape[0]-1):
            wall_n = Walls_Hesse[n, 0:4]
            for o in range(n+1,Walls_Hesse.shape[0]):
                wall_o = Walls_Hesse[o, 0:4]


                n1 = wall_m[0:3]
                n2 = wall_n[0:3]
                n3 = wall_o[0:3]


                angles_cos = np.array([
                    np.dot(n1, n2),
                    np.dot(n1, n3),
                    np.dot(n2, n3),
                ])


                if np.max(np.abs(angles_cos)) > 0.9999:
                    continue


                A = np.vstack([n1, n2, n3])
                b = np.array([wall_m[3], wall_n[3], wall_o[3]])
                corner = np.linalg.solve(A, b)


                Corners.append(corner)

    array_8x3 = np.array(Corners)
    array_8x3 = np.abs(array_8x3)
    return array_8x3




def filter_image_sources(ISList, WallsHesse, ReceiverPos):
    """
    Filter image sources based on their positions relative to the walls.
   
    Args:
    - ISList (list of dict): List of image sources, where each source is a dictionary with keys:
                            'Position': (np.ndarray) position of the image source
                            'Walls': (list) list of wall indices that the source is reflected off
    - WallsHesse (np.ndarray): An (n_walls, 4) array where each row represents a wall
                            in Hesse normal form [n0x, n0y, n0z, d].
    - ReceiverPos (np.ndarray): The position of the receiver.
    - MaxOrder (int): The maximum order of reflections to consider.


    Returns
    -------
    - list of dict: Filtered list of image sources.
    """
    corners = get_corners_of_walls_hesse(WallsHesse)
    bound_min = np.min(corners, axis=0) - 0.001 #axis=0
    bound_max = np.max(corners, axis=0) + 0.001

    n_walls = WallsHesse.shape[0]
    filter_flags = np.ones(len(ISList), dtype=bool)

    for ind, mis in enumerate(ISList):
        walls = mis.Walls

        if len(walls) < 1:
            continue

        a = ReceiverPos
        b = mis.Position - ReceiverPos

        for k in range(len(walls)-1, -1, -1):
            wall_hesse = WallsHesse[walls[k]-1,:]  # PROBLEM HERE original was walls[k]. maybe walls[k-1] is correct?
            n0 = wall_hesse[0:3]
            d = wall_hesse[3]  # Index 3 to get the scalar value

            t = (d - np.dot(n0, a)) / np.dot(n0, b)

            if t < 0:
                filter_flags[ind] = False
                break

            a = a + t * b

            # Ensure that a, bound_min, and bound_max are scalars; otherwise, use np.any() or np.all()


            if any(a[i] < bound_min[i] or a[i] > bound_max[i] for i in range(3)):
                filter_flags[ind] = False
                break

            test_in_room = True
            except_wall = walls[k]  # Changed from walls[k] to element
            for ind2 in range(1,n_walls+1): #original was for ind2 in range(n_walls)
                if ind2 == except_wall:
                    continue

                wall = WallsHesse[ind2-1,:] #original was WallsHesse[ind2,:]
                if np.dot(a, wall[0:3]) > wall[3]:
                    test_in_room = False
                    break

            if not test_in_room:
                filter_flags[ind] = False
                break

            b = b - 2 * np.dot(b, n0) * n0

        # if filter_flags[ind]:
        #     errorvalue = np.linalg.norm(np.cross(b, mis.Position - a)) / np.linalg.norm(b)
        #     if errorvalue > 0.01:
        #         raise ValueError(f'error: errorvalue ({errorvalue:.6f}) too big')

    return [ISList[i] for i in range(len(ISList)) if filter_flags[i]]




def calculate_impulse_response(ISList, WallsR, ReceiverPos):
    c0 = 346.18
    sampling_rate = 10000 #normally 44100


    longest_distance = 0.0


    for ind, mis in enumerate(ISList):

        im_source = np.array(mis.Position)
        receiver = np.array(ReceiverPos)
        dis = np.linalg.norm(im_source - receiver)
        if dis > longest_distance:
            longest_distance = dis


    T = longest_distance / c0
    T_rounded = np.ceil(20 * T) / 20
    IR = np.zeros(round(T_rounded * sampling_rate) + 1)


    for ind, mis in enumerate(ISList):
        imagesource = mis
        order = float(imagesource.Order)
        im_source = np.array(mis.Position)
        receiver = np.array(ReceiverPos)
        distance = np.linalg.norm(im_source - receiver)
        t = distance / c0
        t_index = round(t * sampling_rate) #aslinya tadi +1
        pressure = (1 / distance) * (WallsR ** order)
        IR[t_index] = pressure

    return IR



