{% if not obj.display %}
:orphan:

{% endif %}

:py:mod:`{{ obj.name }}`
=========={{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|prepare_docstring|indent(3) }}

{% endif %}

{% block submodules_subpackages %}

{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% set visible_modules_packages = visible_subpackages + visible_submodules %}

{% if visible_modules_packages %}

.. raw:: html

   <div style="display: None">

.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for child in visible_modules_packages %}
   {{ child.short_name }}/index.rst
{% endfor %}


.. raw:: html

   </div><h2 id="packages-and-modules-table">Packages and Modules</h2>

.. list-table::

{% for child in visible_modules_packages %}
   *
      - :py:mod:`{{child.id}}`
      - {{ child.summary }}
{% endfor %}
{% endif %}

{% endblock %}

{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}

{% set obj_short_name = obj.name.split(".") | last %}
{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
{% if "show-module-summary" in autoapi_options and (visible_classes or visible_functions or visible_attributes) %}

{% block classes scoped %}
{% if visible_classes %}

Classes
-------

.. list-table::

{% for child in visible_classes %}
   *
      - :py:class:`{{ child.id }}`
      - {{ child.summary }}
{% endfor %}

{% for klass in visible_classes %}
{{ klass.render()|indent(0) }}
{% endfor %}


{% endif %}
{% endblock %}

{% block functions scoped %}
{% if visible_functions %}

Functions
---------

.. list-table::

{% for child in visible_functions %}
   *
      - :py:func:`{{ child.id }}`
      - {{ child.summary }}
{% endfor %}

{% for function in visible_functions %}
{{ function.render()|indent(0) }}
{% endfor %}


{% endif %}
{% endblock %}

{% block attributes scoped %}
{% if visible_attributes %}

Attributes
----------

.. list-table::

{% for child in visible_attributes %}
   *
      - :py:data:`{{ child.id }}`
      - {{ child.summary }}
{% endfor %}

{% for attribute in visible_attributes %}
{{ attribute.render()|indent(0) }}
{% endfor %}


{% endif %}
{% endblock %}



{% endif %}
{% endif %}
{% endblock %}
