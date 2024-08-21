# %%
import numpy as np
import sparapy as sp
from sparapy import image_source as ims

def test_calculate_image_sources():

    RoomSizes = (1, 1, 1) 
    WallsHesse = ims.get_walls_hesse(*RoomSizes)

    SourcePos = [0.50, 0.50, 0.50]
    MaxOrder = 3

    ISList_valid = ims.calculate_image_sources(WallsHesse, SourcePos, MaxOrder)

    # choose random values and check if values are the same
    # check total number of image sources in list
    # check 3 random values and see if theyre the same or not
    
    assert len(ISList_valid) == 163
    
    assert np.array_equal(ISList_valid[89].Position, np.array([0.5, -2.5, 0.5]))
    assert np.array_equal(ISList_valid[89].Walls, np.array([3,4,3]))
    assert ISList_valid[89].Order == 3



def test_filter_image_sources():

    RoomSizes = (1, 1, 1) 
    WallsHesse = ims.get_walls_hesse(*RoomSizes)

    SourcePos = [0.50, 0.50, 0.50]
    MaxOrder = 3

    ISList_valid = ims.calculate_image_sources(WallsHesse, SourcePos, MaxOrder)

    ReceiverPos = [0.25, 0.25, 0.25]
    ISList_audible = ims.filter_image_sources(ISList_valid, WallsHesse, ReceiverPos)

    assert len(ISList_audible) == 91