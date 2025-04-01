Radiosity after Kang
--------------------

This module contains the implementation of the radiosity algorithm after
Kang (2002).
It was used to simulate sound propagation in street canyons, where the walls are
assumed to be diffusely reflecting. The limitations of the implementation are
- it is not optimized for speed 
- it does not support other patches then parallel or perpendicular relationships
- it only supports rectangular patches

This was extended in the :py:class:`sparrowpy.DirectionalRadiosityKang` class,
which additionally supports directional reflections for surfaces in terms of
:py:mod:`BRDFs<sparrowpy.brdf>`.

References:

-  Jian Kang, “Sound propagation in street canyons: Comparison between 
   diffusely and geometrically reflecting boundaries,” The Journal of the
   Acoustical Society of America, vol. 107, no. 3, pp. 1394-1404, Mar. 2000,
   doi: 10.1121/1.428580.


.. autoclass:: sparrowpy.RadiosityKang
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: sparrowpy.PatchesKang
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: sparrowpy.DirectionalRadiosityKang
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: sparrowpy.PatchesDirectionalKang
   :members:
   :undoc-members:
   :inherited-members:
