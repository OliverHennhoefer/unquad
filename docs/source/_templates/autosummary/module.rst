{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:

   {% for item in items %}
   {{ item }}
   {%- endfor %} 