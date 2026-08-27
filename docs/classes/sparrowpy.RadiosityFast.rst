Radiosity fast
--------------

This class represents an optimized radiosity implementation. It includes
the following features:

- It is optimized for speed
- It supports various forms and relationships of patches
- It supports :py:mod:`BRDFs<sparrowpy.brdf>`

.. warning::
   This class contains a known issue in the energy exchange calculation
   for non-perfectly Lambertian surfaces. The problem will be fixed
   in a future release.


.. autoclass:: sparrowpy.DirectionalRadiosityFast
   :members:
   :undoc-members:
   :inherited-members:
