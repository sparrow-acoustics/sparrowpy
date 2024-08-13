import numpy as np
from sparapy import is_obj as ob

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
    return ISList
    

    

def get_corners_of_walls_hesse(walls_hesse):
    """
    Compute the corners of walls based on their Hesse normal form representation.
    
    Args:
    - walls_hesse (np.ndarray): An (n_walls, 4) array where each row represents a wall
                                in Hesse normal form [n0x, n0y, n0z, d].
    
    Returns:
    - np.ndarray: An array of corner points.
    """
    corners = []
    for i in range(walls_hesse.shape[0]):
        n0 = walls_hesse[i, :3]
        d = walls_hesse[i, 3]
        # Example calculation (a placeholder)
        # Calculate a point on the plane, in practice, you would need actual corner points calculation
        corner = d * n0 / np.linalg.norm(n0)
        corners.append(corner)
    return np.array(corners)

def filter_image_sources(ISList, WallsHesse, ReceiverPos, MaxOrder):
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

    Returns:
    - list of dict: Filtered list of image sources.
    """
    corners = get_corners_of_walls_hesse(WallsHesse)
    bound_min = np.min(corners, axis=0) - 0.001
    bound_max = np.max(corners, axis=0) + 0.001
    
    n_walls = WallsHesse.shape[0]
    filter_flags = np.ones(len(ISList), dtype=bool)
    
    for ind, mis in enumerate(ISList):
        walls = mis['Walls']
        
        if len(walls) < 1:
            continue
        
        a = ReceiverPos
        b = mis['Position'] - ReceiverPos
        
        for k in range(len(walls) - 1, -1, -1):
            wall_hesse = WallsHesse[walls[k]]
            n0 = wall_hesse[:3]
            d = wall_hesse[3]

            t = (d - np.dot(n0, a)) / np.dot(n0, b)
            
            if t < 0:
                filter_flags[ind] = False
                break
            
            a = a + t * b
            
            if (a[0] < bound_min[0] or a[1] < bound_min[1] or a[2] < bound_min[2] or
                a[0] > bound_max[0] or a[1] > bound_max[1] or a[2] > bound_max[2]):
                filter_flags[ind] = False
                break
                
            test_in_room = True
            except_wall = walls[k]
            for ind2 in range(n_walls):
                if ind2 == except_wall:
                    continue
                
                wall = WallsHesse[ind2]
                if np.dot(a, wall[:3]) > wall[3]:
                    test_in_room = False
                    break
            
            if not test_in_room:
                filter_flags[ind] = False
                break
            
            b = b - 2 * np.dot(b, n0) * n0
        
        if filter_flags[ind]:
            errorvalue = np.linalg.norm(np.cross(b, mis['Position'] - a)) / np.linalg.norm(b)
            if errorvalue > 0.01:
                raise ValueError(f'error: errorvalue ({errorvalue:.6f}) too big')
    
    return [ISList[i] for i in range(len(ISList)) if filter_flags[i]]


def calculate_impulse_response(ISList, WallsR, ReceiverPos):
    c0 = 340.0
    sampling_rate = 1000

    longest_distance = 0.0

    for mis in ISList:
        dis = np.linalg.norm(mis['Position'] - ReceiverPos)
        if dis > longest_distance:
            longest_distance = dis

    T = longest_distance / c0
    T_rounded = np.ceil(20 * T) / 20
    IR = np.zeros(round(T_rounded * sampling_rate) + 1)

    for mis in ISList:
        imagesource = mis
        order = float(imagesource['Order'])
        distance = np.linalg.norm(imagesource['Position'] - ReceiverPos)
        t = distance / c0
        t_index = round(t * sampling_rate) + 1
        pressure = (1 / distance) * (WallsR ** order)
        IR[t_index] = pressure