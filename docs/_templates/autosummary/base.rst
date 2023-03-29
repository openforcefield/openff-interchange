{% block title -%}

{{ ("``" ~ objname ~ "``") | underline}}

{%- endblock %}
{% block base %}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
    {% if objtype in ["attribute", "data"] -%}
    :no-value:
    {%- endif %}

{%- endblock %}