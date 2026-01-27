<%
    from docs.generate_api import ToMarkdown, Module, Function

    def to_markdown(text):
        return ToMarkdown.convert(text, module=module)

    def get_description(module):
        first_paragraph = module.docstring.split('\n')[0]
        if len(first_paragraph) > 0:
            if first_paragraph[-1] == '.':
                return first_paragraph[:-1]
        return first_paragraph

    variables = module.variables
    classes = module.classes
    functions = module.functions
    subpackages = list(filter(lambda x: x.is_package, module.submodules))
    submodules = list(filter(lambda x: not x.is_package, module.submodules))

    icon = get_icon(module)
    tags = get_tags(module)
%>
<%def name="format_heading(dobj, level, is_callable)" buffered="True">
<%
    if isinstance(dobj, Module) and dobj.fname != 'api' and icon is not None:
        module_icon = ":" + icon.replace("/", "-") + ": "
    else:
        module_icon = ""
    name = dobj.fname if isinstance(dobj, Module) and dobj.fname != 'api' else dobj.name
    github_link = format_github_link(dobj)
    para = "()" if is_callable else ""
%>
${"#" * level} ${module_icon}${name} <span class="dobjtype">${dobj.type_name}</span><a class="githublink" href="${github_link}" target="_blank" title="Jump to source">:material-github:</a> { #${dobj.refname} data-toc-label='${dobj.name + para}' }
</%def>
<%def name="callable_to_markdown(dobj)" buffered="True">
<%
    params = Function.get_params(dobj, module=dobj.module)
    qualname = dobj.qualname
%>
% if len(params) == 0:
```python
${qualname}()
```
% else:
```python
${qualname}(
    % for i, param in enumerate(params):
    % if i < len(params) - 1:
    ${param},
    % else:
    ${param}
    % endif
    % endfor
)
```
% endif
</%def>
<%def name="dobj_to_markdown(dobj, level, is_callable)" buffered="True">
${format_heading(dobj, level, is_callable)}

% if is_callable:
${callable_to_markdown(dobj)}
% endif

${dobj.docstring | to_markdown}
</%def>

## Start the output logic for an entire module.

---
title: ${"API" if module.fname == "api" else module.fname}
% if len(get_description(module)) > 0:
description: ${get_description(module)}
% endif
% if icon is not None:
icon: ${icon}
% endif
% if len(tags) > 0:
tags:
% for tag in tags:
    - ${tag}
% endfor
% endif
---

${format_heading(module, 1, False)}

${module.docstring | to_markdown}

% if variables:
% for variable in variables:
${'---'}
${dobj_to_markdown(variable, 2, False)}

% endfor
% endif
% if functions:
% for function in functions:
${'---'}
${dobj_to_markdown(function, 2, True)}

% endfor
% endif
% if classes:
% for cls in classes:
${'---'}
${dobj_to_markdown(cls, 2, True)}
<%
    dobjs = []
    for dobj in cls.class_variables:
        dobjs.append((dobj, False))
    for dobj in cls.functions:
        dobjs.append((dobj, True))
    for dobj in cls.instance_variables:
        dobjs.append((dobj, False))
    for dobj in cls.methods:
        dobjs.append((dobj, True))
    dobjs = sorted(dobjs, key=lambda x: x[0].name)
    superclasses = cls.superclasses
    subclasses = cls.subclasses
%>
% if superclasses:
${'__Superclasses__'}

% for supercls in superclasses:
* ${supercls.link}
% endfor
%endif
<%
    members = cls.inherited_members
%>
% if members:
${'__Inherited members__'}

% for member in members:
* ${member.link}
% endfor

% endif
% if subclasses:
${'__Subclasses__'}

% for subcls in subclasses:
* ${subcls.link}
% endfor

%endif
% if len(dobjs) > 0:
% for dobj, is_callable in dobjs:
${'---'}
${dobj_to_markdown(dobj, 3, is_callable)}

% endfor
% endif
% endfor
% endif
% if subpackages:
${'---'}
${'##'} ${'Sub-packages'}

% for subpackage in subpackages:
* ${subpackage.link}
% endfor

% endif
% if submodules:
${'---'}
${'##'} ${'Sub-modules'}

% for submodule in submodules:
* ${submodule.link}
% endfor

% endif