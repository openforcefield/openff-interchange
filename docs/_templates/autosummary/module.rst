{% if fullname == "openff." ~ objname -%}
{%- set title = fullname | escape -%}
{%- else -%}
{%- set title = objname | escape -%}
{%- endif %}
{{ ("``" ~ title ~ "``") | underline('=')}}

.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: {{ fullname }}

{% block classes -%}

{%- set types = [] -%}
{%- for item in members -%}
   {%- if not item.startswith('_') and not (
      item in functions
      or item in attributes
      or item in exceptions
      or item in modules
      or item == "TYPE_CHECKING"
) -%}
      {%- set _ = types.append(item) -%}
   {%- endif -%}
{%- endfor %}

{% if types %}
{{ _('Classes') | escape | underline(line="-") }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in types %}
      {% if item.startswith(fullname ~ ".") -%}
      {{- item[((fullname ~ ".") | length):] -}}
      {%- else -%}
      {{- item -}}
      {%- endif %}
   {%- endfor %}

{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
{{ _('Functions') | escape | underline(line="-") }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {% if item.startswith(fullname ~ ".") -%}
      {{- item[((fullname ~ ".") | length):] -}}
      {%- else -%}
      {{- item -}}
      {%- endif %}
   {%- endfor %}

{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
{{ _('Exceptions') | escape | underline(line="-") }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in exceptions %}
      {% if item.startswith(fullname ~ ".") -%}
      {{- item[((fullname ~ ".") | length):] -}}
      {%- else -%}
      {{- item -}}
      {%- endif %}
   {%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
{{ _('Module Attributes') | escape | underline(line="-") }}

   {% for item in attributes %}
   .. autoattribute:: {% if item.startswith(fullname ~ ".") -%}
                      {{- item[((fullname ~ ".") | length):] -}}
                      {%- else -%}
                      {{- item -}}
                      {%- endif %}
   {%- endfor %}

{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
{{ _('Modules') | escape | underline(line="-") }}

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules if item not in exclude_modules %}
   {% if item.startswith(fullname ~ ".") -%}
   {{- item[((fullname ~ ".") | length):] -}}
   {%- else -%}
   {{- item -}}
   {%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}
