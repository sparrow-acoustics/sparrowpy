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

ReceiverPos = [0.25, 0.25, 0.25]
ISList_audible = ims.filter_image_sources(ISList_valid, WallsHesse, ReceiverPos)

a = 230 + 250
# %%
print(ISList_valid[89].Position)


# %%
print(ISList_valid[89].Walls)

# %%
print(ISList_valid[89].Order)


# %%
if np.array_equal(ISList_valid[89].Position, np.array([0.5, -2.5, 0.5])):
    print(True)
# %%
print()