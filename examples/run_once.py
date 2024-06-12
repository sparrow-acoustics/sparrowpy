from sparapy.radiosity_fast import DRadiosityFast
import sparapy.geometry as geo
import sparapy as sp
import cProfile
import pyfar as pf
import sofar as sf
import os
import tqdm
import time
import numpy as np
import tracemalloc

sample_walls = sp.testing.shoebox_room_stub(1, 1, 1)


radiosity = DRadiosityFast.from_polygon(sample_walls, 1)

radiosity.check_visibility()

t0 = time.time()
radiosity.calculate_form_factors(method='kang')
tkang = time.time()-t0

kang = radiosity.form_factors

t0 = time.time()
radiosity.calculate_form_factors(method='universal')
tuniv = time.time()-t0

univ = radiosity.form_factors

msr_error = np.mean(np.sqrt(np.square(kang-univ)))

print("\\n\n\nRUNTIME:")
print(f"kang {tkang: .10f}s")
print(f"univ: {tuniv: .10f}s")

print("\n\nMSR: ")
print(f"abs: {msr_error: .10f}")
print(f"rel: {100*msr_error/np.mean(kang): .10f}%")
print("end")


