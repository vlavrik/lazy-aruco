Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: laruco.laruco.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autofunction:: laruco.laruco.get_person

.. autoclass:: laruco.generate.Generate
   :members:

.. autoexception:: laruco.laruco.InvalidKindError

For example:

>>> import laruco
>>> laruco.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

