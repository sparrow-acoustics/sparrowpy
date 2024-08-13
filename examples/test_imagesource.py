# %%
import numpy as np
import sparapy as sp
from sparapy import image_source as ims

# %%
RoomSizes = (1, 1, 1) 
WallsHesse = ims.get_walls_hesse(*RoomSizes)

WallsR = 0.9
SourcePos = [0.50, 0.50, 0.50]
MaxOrder = 3

ISList_valid = ims.calculate_image_sources(WallsHesse, SourcePos, MaxOrder)

a = 230 + 250

# %%
