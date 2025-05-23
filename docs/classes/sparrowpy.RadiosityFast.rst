Radiosity fast
--------------

This class represents an optimized radiosity implementation. It includes
the following features:

- It is optimized for speed
- It supports various forms and relationships of patches
- It supports :py:mod:`BRDFs<sparrowpy.brdf>`

It is now possible to load geometries in this class from `Blender`_ or STL files.
We recommend you see how this is done in our `example`_ on geometry loading, which includes some important details about import options.

.. autoclass:: sparrowpy.DirectionalRadiosityFast
   :members:
   :undoc-members:
   :inherited-members:

.. _Blender: https://www.blender.org/
.. _example: ../examples/geometry_loading_from_file.ipynb