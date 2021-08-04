{{ fullname | escape | underline(line="=")}}

{% set documented_members = [] %}

.. automodule:: {{ fullname }}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
   {% set _ = documented_members.append(item) %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
{{ _('Classes') | escape | underline(line="-") }}

   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :special-members: __init__
      :undoc-members:
      :show-inheritance:
      :inherited-members:
      {% set _ = documented_members.append(item) %}
   {%- endfor %}

{% endif %}
{% endblock %}


{% block functions %}
{% if functions %}
{{ _('Functions') | escape | underline(line="-") }}

   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}

   {% for item in functions %}
   .. autofunction:: {{ item }}
   {% set _ = documented_members.append(item) %}
   {%- endfor %}

{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
{{ _('Exceptions') | escape | underline(line="-") }}

   .. autosummary::
      :nosignatures:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {% set _ = documented_members.append(item) %}
   {%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
{{ _('Module Attributes') | escape | underline(line="-") }}

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      {{ item }}
      {%- set _ = documented_members.append(item) -%}
   {%- endfor %}

{% endif %}
{% endblock %}

{#
{% block others %}

{% set others = [] %}
{% for item in members %}
   {% if item | first != "_" and not (item in documented_members) %}
      {% set _ = others.append(item) %}
   {% endif %}
{%- endfor %}

{% if others %}
{{ _('Other Module Members') | escape | underline(line="-") }}

   .. autosummary::
      :nosignatures:
      :toctree: other_members
   {% for item in others %}
      {{ item }}
   {%- endfor %}

{% endif %}
{% endblock %}
#}