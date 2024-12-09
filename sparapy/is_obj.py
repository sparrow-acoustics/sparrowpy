"""creates an image source object """


import numpy as np


class ISObj:


    def __init__(self, position=None, walls=None, order=None):
        if position is None:
            self.Position = np.array([0.00, 0.00, 0.00], dtype=float)
        else:
            self.Position = np.array(position, dtype=float)


        if walls is None:
            self.Walls = []
        else:
            self.Walls = walls


       
        if order is None:
            self.Order = -1
        else:
            self.Order = order


       


    def find_image_sources_with_same_position(self, ISList):
        IndexList = []
        for ind, is_obj in enumerate(ISList):
            if np.array_equal(self.Position, is_obj.Position):
                IndexList.append(ind)
        return IndexList


    def get_small_info_strings(self):
        lines = []
        walls_str = 'Wall IDs: '
        if not self.Walls:
            walls_str += '[]'
        else:
            walls_str += ', '.join(map(str, self.Walls))
        lines.append(f'Order: {self.Order}')
        lines.append(walls_str)
        return lines


    def export_is_as_array(self):
        return np.concatenate((self.Position, self.Walls)).astype(float)


    @staticmethod
    def import_array(arr):
        obj = ISObj()
        obj.Position = np.array(arr[:3], dtype=float)
        walls = np.array(arr[3:], dtype=int)
        walls = walls[walls != 0]
        if walls.size > 0:
            obj.Walls = walls.tolist()
            obj.Order = len(walls)
        else:
            obj.Order = 0
        return obj


    @staticmethod
    def export_is_list_as_matrix(islist):
        mat = np.zeros((len(islist), 3 + max(len(is_obj.Walls) for is_obj in islist)))
        for ind, IS in enumerate(islist):
            array = IS.export_is_as_array()
            mat[ind, :len(array)] = array
        return mat


    @staticmethod
    def import_matrix_as_is_list(mat):
        islist = []
        for row in mat:
            islist.append(ISObj.import_array(row))
        return islist


    @staticmethod
    def scatter_is_callback(event_obj):
        pos = event_obj["Position"]
        userdata = event_obj.get("Target", {}).get("UserData", None)
        output_txt = []


        if userdata and isinstance(userdata, list) and all(isinstance(obj, ISObj) for obj in userdata):
            output_txt.append(f'Position: {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}')
            for list_ind, mis in enumerate(userdata):
                if np.allclose(mis.Position, pos[:3], atol=0.001):
                    output_txt.append(f'--Index: {list_ind} ---')
                    newlines = mis.get_small_info_strings()
                    output_txt.extend(newlines)
        else:
            output_txt = [f'X: {pos[0]:.4f}', f'Y: {pos[1]:.4f}']
            if len(pos) > 2:
                output_txt.append(f'Z: {pos[2]:.4f}')


        return output_txt






