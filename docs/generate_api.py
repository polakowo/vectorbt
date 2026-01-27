# This file is a modified derivative of pdoc3.
# Copyright (c) the pdoc3 authors.
# Modifications Copyright (c) 2021 Oleg Polakow. All rights reserved.
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# See docs/LICENSE.md for the full license text.

"""Auto-generate API documentation in Markdown format."""

import ast
import enum
import importlib
import inspect
import os
import os.path as path
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
import typing
from contextlib import contextmanager
from copy import copy
from functools import lru_cache, partial, reduce, wraps
from itertools import tee
from warnings import warn

from mako.exceptions import TopLevelLookupException
from mako.lookup import TemplateLookup
from numba.core.registry import CPUDispatcher

_get_type_hints = lru_cache()(typing.get_type_hints)
_UNKNOWN_MODULE = "?"
T = typing.TypeVar("T", "Module", "Class", "Function", "Variable")

__pdoc__ = {}

tpl_lookup = TemplateLookup(
    cache_args=dict(cached=True, cache_type="memory"),
    input_encoding="utf-8",
    directories=[path.join(path.dirname(__file__), "templates")],
)


class Context(dict):
    __pdoc__["Context.__init__"] = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blacklisted = getattr(args[0], "blacklisted", set()) if args else set()


global_context = Context()


def import_module(module, *, reload=False):
    @contextmanager
    def _module_path(module):
        from os.path import abspath, dirname, isdir, isfile, split

        pth = "_dummy_nonexistent"
        module_name = inspect.getmodulename(module)
        if isdir(module):
            pth, module = split(abspath(module))
        elif isfile(module) and module_name:
            pth, module = dirname(abspath(module)), module_name
        try:
            sys.path.insert(0, pth)
            yield module
        finally:
            sys.path.remove(pth)

    if isinstance(module, Module):
        module = module.obj
    if isinstance(module, str):
        with _module_path(module) as module_path:
            try:
                module = importlib.import_module(module_path)
            except Exception as e:
                raise ImportError(f"Error importing {module!r}: {e.__class__.__name__}: {e}")

    assert inspect.ismodule(module)
    if reload and not module.__name__.startswith(__name__):
        module = importlib.reload(module)
        for mod_key, mod in list(sys.modules.items()):
            if mod_key.startswith(module.__name__):
                importlib.reload(mod)
    return module


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def pep224_docstrings(doc_obj, *, _init_tree=None):
    if isinstance(doc_obj, Module) and doc_obj.is_namespace:
        return {}, {}

    vars = {}
    instance_vars = {}

    if _init_tree:
        tree = _init_tree
    else:
        try:
            _ = inspect.findsource(doc_obj.obj)
            tree = ast.parse(doc_obj.source)
        except (OSError, TypeError, SyntaxError) as exc:
            is_builtin = getattr(doc_obj.obj, "__module__", None) == "builtins"
            if not is_builtin:
                warn(
                    f"Couldn't read PEP-224 variable docstrings from {doc_obj!r}: {exc}",
                    stacklevel=3 + int(isinstance(doc_obj, Class)),
                )
            return {}, {}

        if isinstance(doc_obj, Class):
            tree = tree.body[0]
            for node in reversed(tree.body):
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    instance_vars, _ = pep224_docstrings(doc_obj, _init_tree=node)
                    break

    def get_name(assign_node):
        if isinstance(assign_node, ast.Assign) and len(assign_node.targets) == 1:
            target = assign_node.targets[0]
        elif isinstance(assign_node, ast.AnnAssign):
            target = assign_node.target
        else:
            return None

        if not _init_tree and isinstance(target, ast.Name):
            name = target.id
        elif (
            _init_tree
            and isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            name = target.attr
        else:
            return None

        if not is_public(name) and not is_whitelisted(name, doc_obj):
            return None
        return name

    for assign_node, str_node in pairwise(ast.iter_child_nodes(tree)):
        if not (
            isinstance(assign_node, (ast.Assign, ast.AnnAssign))
            and isinstance(str_node, ast.Expr)
            and isinstance(str_node.value, ast.Str)
        ):
            continue
        name = get_name(assign_node)
        if not name:
            continue
        docstring = inspect.cleandoc(str_node.value.s).strip()
        if docstring:
            vars[name] = docstring

    for assign_node in ast.iter_child_nodes(tree):
        if not isinstance(assign_node, (ast.Assign, ast.AnnAssign)):
            continue
        name = get_name(assign_node)
        if not name or name in vars:
            continue

        def _get_indent(line):
            return len(line) - len(line.lstrip())

        source_lines = doc_obj.source.splitlines()
        assign_line = source_lines[assign_node.lineno - 1]
        assign_indent = _get_indent(assign_line)
        comment_lines = []
        MARKER = "#: "
        for line in reversed(source_lines[: assign_node.lineno - 1]):
            if _get_indent(line) == assign_indent and line.lstrip().startswith(MARKER):
                comment_lines.append(line.split(MARKER, maxsplit=1)[1])
            else:
                break
        comment_lines = comment_lines[::-1]
        if MARKER in assign_line:
            comment_lines.append(assign_line.rsplit(MARKER, maxsplit=1)[1])
        if comment_lines:
            vars[name] = "\n".join(comment_lines)

    return vars, instance_vars


@lru_cache()
def is_whitelisted(name, doc_obj):
    refname = f"{doc_obj.refname}.{name}"
    module = doc_obj.module
    while module:
        qualname = refname[len(module.refname) + 1 :]
        if module.__pdoc__.get(qualname) or module.__pdoc__.get(refname):
            return True
        module = module.supermodule
    return False


@lru_cache()
def is_blacklisted(name, doc_obj):
    refname = f"{doc_obj.refname}.{name}"
    module = doc_obj.module
    while module:
        qualname = refname[len(module.refname) + 1 :]
        if module.__pdoc__.get(qualname) is False or module.__pdoc__.get(refname) is False:
            return True
        module = module.supermodule
    return False


def is_public(ident_name):
    return not ident_name.startswith("_")


def is_function(obj):
    return inspect.isroutine(obj) and callable(obj)


def is_descriptor(obj):
    return (
        inspect.isdatadescriptor(obj)
        or inspect.ismethoddescriptor(obj)
        or inspect.isgetsetdescriptor(obj)
        or inspect.ismemberdescriptor(obj)
    )


def filter_type(type_, values):
    if isinstance(values, dict):
        values = values.values()
    return [i for i in values if isinstance(i, type_)]


def toposort(graph):
    items_without_deps = reduce(set.union, graph.values(), set()) - set(graph.keys())
    yield from items_without_deps
    ordered = items_without_deps
    while True:
        graph = {item: (deps - ordered) for item, deps in graph.items() if item not in ordered}
        ordered = {item for item, deps in graph.items() if not deps}
        yield from ordered
        if not ordered:
            break
    assert not graph, f"A cyclic dependency exists amongst {graph!r}"


def link_inheritance(context=None):
    if context is None:
        context = global_context
    graph = {cls: set(cls.mro(only_documented=True)) for cls in filter_type(Class, context)}
    for cls in toposort(graph):
        cls.fill_inheritance()
    for module in filter_type(Module, context):
        module.link_inheritance()


class Doc:
    __slots__ = ("module", "name", "obj", "docstring", "inherits")

    def __init__(self, name, module, obj, docstring=None):
        self.module = module
        self.name = name
        self.obj = obj
        self.docstring = (docstring or inspect.getdoc(obj) or "").strip()
        self.inherits = None

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.refname!r}>"

    @property
    @lru_cache()
    def source(self):
        try:
            lines, _ = inspect.getsourcelines(self.obj)
        except (ValueError, TypeError, OSError):
            return ""
        return inspect.cleandoc("".join(["\n"] + lines))

    @property
    def refname(self):
        return self.name

    @property
    def qualname(self):
        return getattr(self.obj, "__qualname__", self.name)

    @property
    def url_base(self):
        return f"{self.module.url_base}#{self.refname}"

    @property
    @lru_cache()
    def inherits_top(self):
        top = self
        while top.inherits:
            top = top.inherits
        return top

    @property
    def link(self):
        return f'[{self.qualname}]({self.inherits_top.url_base} "{self.refname}")'

    @property
    def type_name(self):
        return "?"

    def __lt__(self, other):
        return self.qualname < other.qualname


class Module(Doc):
    __slots__ = ("supermodule", "doc", "_context", "_is_inheritance_linked", "_skipped_submodules", "_curr_dir")

    def __init__(
        self,
        module,
        *,
        docfilter=None,
        supermodule=None,
        context=None,
        skip_errors=False,
        curr_dir="api",
    ):
        if isinstance(module, str):
            module = import_module(module)
        super().__init__(module.__name__, self, module)
        if self.name.endswith(".__init__") and not self.is_package:
            self.name = self.name[: -len(".__init__")]

        self._context = global_context if context is None else context
        assert isinstance(self._context, Context)

        self.supermodule = supermodule
        self.doc = {}
        self._is_inheritance_linked = False
        self._skipped_submodules = set()
        self._curr_dir = curr_dir

        var_docstrings, _ = pep224_docstrings(self)
        public_objs = []

        if hasattr(self.obj, "__pdoc__all__"):
            for name in self.obj.__pdoc__all__:
                try:
                    obj = getattr(self.obj, name)
                except AttributeError:
                    warn(f"Module {self.module!r} doesn't contain identifier `{name}` exported in `__pdoc__all__`")
                    continue
                if not is_blacklisted(name, self):
                    obj = inspect.unwrap(obj)
                public_objs.append((name, obj))
        else:

            def is_from_this_module(obj):
                mod = inspect.getmodule(inspect.unwrap(obj))
                return mod is None or mod.__name__ == self.obj.__name__

            for name, obj in inspect.getmembers(self.obj):
                if (is_public(name) or is_whitelisted(name, self)) and (
                    is_blacklisted(name, self) or is_from_this_module(obj) or name in var_docstrings
                ):
                    if is_blacklisted(name, self):
                        self._context.blacklisted.add(f"{self.refname}.{name}")
                        continue
                    obj = inspect.unwrap(obj)
                    public_objs.append((name, obj))

            index = list(self.obj.__dict__).index
            public_objs.sort(key=lambda i: index(i[0]))

        for name, obj in public_objs:
            if is_function(obj):
                self.doc[name] = Function(name, self, obj)
            elif inspect.isclass(obj):
                self.doc[name] = Class(name, self, obj)
            elif name in var_docstrings:
                self.doc[name] = Variable(name, self, var_docstrings[name], obj=obj)

        if self.is_package:

            def iter_modules(paths_):
                from os.path import isdir, join

                for pth in paths_:
                    for file in os.listdir(pth):
                        if file.startswith((".", "__pycache__", "__init__.py")):
                            continue
                        module_name = inspect.getmodulename(file)
                        if module_name:
                            yield module_name
                        if isdir(join(pth, file)) and "." not in file:
                            yield file

            for root in iter_modules(self.obj.__path__):
                if root in self.doc:
                    continue
                if not is_public(root) and not is_whitelisted(root, self):
                    continue
                if is_blacklisted(root, self):
                    self._skipped_submodules.add(root)
                    continue
                assert self.refname == self.name
                fullname = f"{self.name}.{root}"
                try:
                    m = Module(
                        import_module(fullname),
                        docfilter=docfilter,
                        supermodule=self,
                        context=self._context,
                        skip_errors=skip_errors,
                        curr_dir=curr_dir,
                    )
                except Exception as ex:
                    if skip_errors:
                        warn(str(ex), Module.ImportWarning)
                        continue
                    raise
                self.doc[root] = m
                if m.is_namespace and not m.doc:
                    del self.doc[root]
                    self._context.pop(m.refname, None)

        if docfilter:
            for name, dobj in list(self.doc.items()):
                if not docfilter(dobj):
                    self.doc.pop(name, None)
                    self._context.pop(dobj.refname, None)

        self._context[self.refname] = self
        for docobj in self.doc.values():
            self._context[docobj.refname] = docobj
            if isinstance(docobj, Class):
                self._context.update((obj.refname, obj) for obj in docobj.doc.values())

    class ImportWarning(UserWarning):
        pass

    __pdoc__["Module.ImportWarning"] = False

    @property
    def __pdoc__(self):
        return getattr(self.obj, "__pdoc__", {})

    def link_inheritance(self):
        if self._is_inheritance_linked:
            return

        for name, docstring in self.__pdoc__.items():
            if docstring is True:
                continue

            refname = f"{self.refname}.{name}"
            if docstring in (False, None):
                if docstring is None:
                    warn(
                        "Setting `__pdoc__[key] = None` is deprecated; "
                        "use `__pdoc__[key] = False` "
                        f"(key: {name!r}, module: {self.name!r})."
                    )
                if name in self._skipped_submodules:
                    continue
                if (
                    not name.endswith(".__init__")
                    and name not in self.doc
                    and refname not in self._context
                    and refname not in self._context.blacklisted
                ):
                    warn(f"__pdoc__-overriden key {name!r} does not exist in module {self.name!r}")

                obj = self.find_ident(name)
                cls = getattr(obj, "cls", None)
                if cls:
                    del cls.doc[obj.name]
                self.doc.pop(name, None)
                self._context.pop(refname, None)
                for key in list(self._context.keys()):
                    if key.startswith(refname + "."):
                        del self._context[key]
                continue

            dobj = self.find_ident(refname)
            if isinstance(dobj, External):
                continue
            if not isinstance(docstring, str):
                raise ValueError(
                    f"__pdoc__ dict values must be strings; __pdoc__[{name!r}] is of type {type(docstring)}"
                )
            dobj.docstring = inspect.cleandoc(docstring)

        for c in filter_type(Class, self.doc):
            c.link_inheritance()

        self._is_inheritance_linked = True

    def to_markdown(self, **kwargs):
        return render_template("/markdown.mako", module=self, **kwargs)

    @property
    def is_package(self):
        return hasattr(self.obj, "__path__")

    @property
    def is_namespace(self):
        try:
            return self.obj.__spec__.origin in (None, "namespace")
        except AttributeError:
            return False

    def find_class(self, cls):
        return self.find_ident(f"{cls.__module__ or _UNKNOWN_MODULE}.{cls.__qualname__}")

    def find_ident(self, name):
        _name = name.rstrip("()")
        if _name.endswith(".__init__"):
            _name = _name[: -len(".__init__")]
        return (
            self.doc.get(_name)
            or self._context.get(_name)
            or self._context.get(f"{self.name}.{_name}")
            or External(name)
        )

    def filter_doc_objs(self, type_, sort=True):
        result = filter_type(type_, self.doc)
        return sorted(result) if sort else result

    @property
    def variables(self):
        return self.filter_doc_objs(Variable)

    @property
    def classes(self):
        return self.filter_doc_objs(Class)

    @property
    def functions(self):
        return self.filter_doc_objs(Function)

    @property
    def submodules(self):
        return self.filter_doc_objs(Module)

    @property
    def url_base(self):
        return "/" + self._curr_dir + "/" + "/".join(self.name.split(".")[1:]) + "/"

    @property
    def type_name(self):
        if self.is_namespace:
            return "namespace"
        if self.is_package:
            return "package"
        return "module"

    @property
    def fname(self):
        if len(self.module.name.split(".")) == 1:
            return self._curr_dir
        return self.name.split(".")[-1]


def _getmembers_all(obj):
    mro = obj.__mro__[:-1]
    names = set(dir(obj))
    for base in mro:
        names.update(base.__dict__.keys())
        names.update(getattr(obj, "__annotations__", {}).keys())

    results = []
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            for base in mro:
                if name in base.__dict__:
                    value = base.__dict__[name]
                    break
            else:
                value = None
        results.append((name, value))
    return results


class Class(Doc):
    __slots__ = ("doc", "_super_members")

    def __init__(self, name, module, obj, *, docstring=None):
        assert inspect.isclass(obj)
        if docstring is None:
            init_doc = inspect.getdoc(obj.__init__) or ""
            if init_doc == object.__init__.__doc__:
                init_doc = ""
            docstring = f'{inspect.getdoc(obj) or ""}\n\n{init_doc}'.strip()

        super().__init__(name, module, obj, docstring=docstring)
        self.doc = {}

        annotations = getattr(self.obj, "__annotations__", {})
        public_objs = []
        for _name, obj_ in _getmembers_all(self.obj):
            if (_name in self.obj.__dict__ or _name in annotations) and (
                is_public(_name) or is_whitelisted(_name, self)
            ):
                if is_blacklisted(_name, self):
                    self.module._context.blacklisted.add(f"{self.refname}.{_name}")
                    continue
                obj_ = inspect.unwrap(obj_)
                public_objs.append((_name, obj_))

        def definition_order_index(
            name, _annot_index=list(annotations).index, _dict_index=list(self.obj.__dict__).index
        ):
            try:
                return _dict_index(name)
            except ValueError:
                pass
            try:
                return _annot_index(name) - len(annotations)
            except ValueError:
                return 9e9

        public_objs.sort(key=lambda i: definition_order_index(i[0]))
        var_docstrings, instance_var_docstrings = pep224_docstrings(self)

        for name, obj_ in public_objs:
            if is_function(obj_):
                self.doc[name] = Function(name, self.module, obj_, cls=self)
            else:
                self.doc[name] = Variable(
                    name,
                    self.module,
                    docstring=(
                        var_docstrings.get(name)
                        or ((inspect.isclass(obj_) or is_descriptor(obj_)) and inspect.getdoc(obj_))
                    ),
                    cls=self,
                    obj=getattr(obj_, "fget", getattr(obj_, "__get__", None)),
                    instance_var=(is_descriptor(obj_) or name in getattr(self.obj, "__slots__", ())),
                )

        for name, docstring_ in instance_var_docstrings.items():
            self.doc[name] = Variable(
                name, self.module, docstring_, cls=self, obj=getattr(self.obj, name, None), instance_var=True
            )

    @property
    def refname(self):
        return f"{self.module.name}.{self.qualname}"

    def mro(self, only_documented=False):
        classes = [self.module.find_class(c) for c in inspect.getmro(self.obj) if c not in (self.obj, object)]
        if self in classes:
            classes.remove(self)
        if only_documented:
            classes = filter_type(Class, classes)
        return classes

    @property
    def superclasses(self):
        return sorted(self.mro())

    @property
    def subclasses(self):
        return sorted([self.module.find_class(c) for c in type.__subclasses__(self.obj)])

    @lru_cache()
    def params(self):
        name = self.name + ".__init__"
        qualname = self.qualname + ".__init__"
        refname = self.refname + ".__init__"
        exclusions = self.module.__pdoc__
        if name in exclusions or qualname in exclusions or refname in exclusions:
            return []
        return Function.get_params(self, module=self.module)

    def filter_doc_objs(self, type_, filter_func=lambda x: True, sort=True):
        result = [obj for obj in filter_type(type_, self.doc) if not obj.inherits and filter_func(obj)]
        return sorted(result) if sort else result

    @property
    def class_variables(self):
        return self.filter_doc_objs(Variable, filter_func=lambda dobj: not dobj.instance_var)

    @property
    def instance_variables(self):
        return self.filter_doc_objs(Variable, filter_func=lambda dobj: dobj.instance_var)

    @property
    def functions(self):
        return self.filter_doc_objs(Function, filter_func=lambda dobj: not dobj.is_method)

    @property
    def methods(self):
        return self.filter_doc_objs(Function, filter_func=lambda dobj: dobj.is_method)

    @property
    def inherited_members(self):
        return sorted([i.inherits for i in self.doc.values() if i.inherits])

    def fill_inheritance(self):
        super_members = self._super_members = {}
        for cls in self.mro(only_documented=True):
            for name, dobj in cls.doc.items():
                if name not in super_members and dobj.docstring:
                    super_members[name] = dobj
                    if name not in self.doc:
                        dobj2 = copy(dobj)
                        dobj2.cls = self
                        self.doc[name] = dobj2
                        self.module._context[dobj2.refname] = dobj2

    def link_inheritance(self):
        if not hasattr(self, "_super_members"):
            return
        for name, parent_dobj in self._super_members.items():
            try:
                dobj = self.doc[name]
            except KeyError:
                continue
            if dobj.obj is parent_dobj.obj or (dobj.docstring or parent_dobj.docstring) == parent_dobj.docstring:
                dobj.inherits = parent_dobj
                dobj.docstring = parent_dobj.docstring
        del self._super_members

    @property
    def type_name(self):
        return "class"


def maybe_lru_cache(func):
    cached_func = lru_cache()(func)

    @wraps(func)
    def wrapper(*args):
        try:
            return cached_func(*args)
        except TypeError:
            return func(*args)

    return wrapper


class Function(Doc):
    __slots__ = ("cls",)

    def __init__(self, name, module, obj, *, cls=None):
        assert callable(obj), (name, module, obj)
        super().__init__(name, module, obj)
        self.cls = cls

    @staticmethod
    def method_type(cls, name):
        func = getattr(cls, name, None)
        if inspect.ismethod(func):
            return classmethod
        for c in inspect.getmro(cls):
            if name in c.__dict__:
                if isinstance(c.__dict__[name], staticmethod):
                    return staticmethod
                return None
        raise RuntimeError(f"{cls}.{name} not found")

    @property
    def is_method(self):
        assert self.cls
        return self.method_type(self.cls.obj, self.name) is None

    @property
    def method(self):
        warn("`Function.method` is deprecated. Use: `Function.is_method`", DeprecationWarning, stacklevel=2)
        return self.is_method

    __pdoc__["Function.method"] = False

    def funcdef(self):
        return "async def" if self.is_async else "def"

    @property
    def is_async(self):
        try:
            obj = inspect.unwrap(self.obj)
            return inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj)
        except AttributeError:
            return False

    @lru_cache()
    def params(self):
        return self.get_params(self, module=self.module)

    @staticmethod
    def get_params(doc_obj, module=None):
        try:
            if inspect.isclass(doc_obj.obj) and doc_obj.obj.__init__ is not object.__init__:
                init_sig = inspect.signature(doc_obj.obj.__init__)
                init_params = list(init_sig.parameters.values())
                signature = init_sig.replace(parameters=init_params[1:])
            else:
                signature = inspect.signature(doc_obj.obj)
        except ValueError:
            signature = Function.signature_from_string(doc_obj)
            if not signature:
                return ["..."]

        def safe_default_value(p):
            value = p.default
            if value is inspect.Parameter.empty:
                return p

            replacement = next(
                (i for i in ("os.environ", "sys.stdin", "sys.stdout", "sys.stderr") if value is eval(i)),
                None,
            )
            if not replacement:
                if isinstance(value, CPUDispatcher):
                    replacement = value.py_func.__name__
                elif isinstance(value, enum.Enum):
                    replacement = str(value)
                elif inspect.isclass(value):
                    replacement = f"{value.__module__ or _UNKNOWN_MODULE}.{value.__qualname__}"
                elif " at 0x" in repr(value):
                    replacement = re.sub(r" at 0x\w+", "", repr(value))

            if replacement:

                class mock:
                    def __repr__(self):
                        return replacement

                return p.replace(default=mock())
            return p

        params = []
        kw_only = False
        pos_only = False
        EMPTY = inspect.Parameter.empty

        for p in signature.parameters.values():
            if not is_public(p.name) and p.default is not EMPTY:
                continue
            if p.name in {"self", "cls_self"}:
                continue

            if p.kind == p.POSITIONAL_ONLY:
                pos_only = True
            elif pos_only:
                params.append("/")
                pos_only = False

            if p.kind == p.VAR_POSITIONAL:
                kw_only = True
            if p.kind == p.KEYWORD_ONLY and not kw_only:
                kw_only = True
                params.append("*")

            p = safe_default_value(p)

            formatted = p.name
            if p.default is not EMPTY:
                formatted += f"={repr(p.default)}"
            if p.kind == p.VAR_POSITIONAL:
                formatted = f"*{formatted}"
            elif p.kind == p.VAR_KEYWORD:
                formatted = f"**{formatted}"

            params.append(formatted)

        if pos_only:
            params.append("/")

        return params

    @staticmethod
    @lru_cache()
    def signature_from_string(self):
        signature = None
        for expr, cleanup_docstring, filt in (
            (r"^{}\(.*\)(?: -> .*)?$", True, lambda s: s),
            (r"^{}\(.*\)(?= -|$)", False, lambda s: s.replace("[", "").replace("]", "")),
        ):
            strings = sorted(re.findall(expr.format(self.name), self.docstring, re.MULTILINE), key=len, reverse=True)
            if strings:
                string = filt(strings[0])
                _locals, _globals = {}, {}
                _globals.update({"capsule": None})
                _globals.update(typing.__dict__)
                _globals.update(self.module.obj.__dict__)
                module_basename = self.module.name.rsplit(".", maxsplit=1)[-1]
                if module_basename in string and module_basename not in _globals:
                    string = re.sub(rf"(?<!\.)\b{module_basename}\.\b", "", string)
                try:
                    exec(f"def {string}: pass", _globals, _locals)
                except SyntaxError:
                    continue
                signature = inspect.signature(_locals[self.name])
                if cleanup_docstring and len(strings) == 1:
                    self.docstring = self.docstring.replace(strings[0], "")
                break
        return signature

    @property
    def refname(self):
        return f"{self.cls.refname if self.cls else self.module.refname}.{self.name}"

    @property
    def qualname(self):
        qualname = getattr(self.obj, "__qualname__", self.name)
        if self.cls and len(qualname.split(".")) == 1:
            return f"{self.cls.qualname}.{self.name}"
        return qualname

    @property
    def link(self):
        return f'[{self.qualname}()]({self.inherits_top.url_base} "{self.refname}")'

    @property
    def type_name(self):
        if self.cls:
            if self.method_type(self.cls.obj, self.name) is classmethod:
                return "class method"
            if self.method_type(self.cls.obj, self.name) is staticmethod:
                return "static method"
            return "method"
        return "function"


class Variable(Doc):
    __slots__ = ("cls", "instance_var")

    def __init__(self, name, module, docstring, *, obj=None, cls=None, instance_var=False):
        super().__init__(name, module, obj, docstring)
        self.cls = cls
        self.instance_var = instance_var

    @property
    def refname(self):
        return f"{self.cls.refname if self.cls else self.module.refname}.{self.name}"

    @property
    def qualname(self):
        if self.cls:
            return f"{self.cls.qualname}.{self.name}"
        return self.name

    @property
    def type_name(self):
        if self.cls:
            if hasattr(self.cls.obj, self.name) and isinstance(getattr(self.cls.obj, self.name), property):
                return "property"
            if self.obj is not None:
                return type(self.obj).__name__
            if not self.instance_var:
                return "class variable"
        if self.obj is not None:
            return type(self.obj).__name__
        return "variable"


class External(Doc):
    def __init__(self, name):
        super().__init__(name, None, None)

    @property
    def link(self):
        return f"`{self.name}`"


@contextmanager
def fenced_code_blocks_hidden(text):
    def _hide(text_):
        def _replace(match):
            orig = match.group()
            new = f"@{hash(orig)}@"
            hidden[new] = orig
            return new

        return re.compile(r"^(?P<fence>```+|~~~+).*\n" r"(?:.*\n)*?" r"^(?P=fence)[ ]*(?!.)", re.MULTILINE).sub(
            _replace, text_
        )

    def _unhide(text_):
        for k, v in hidden.items():
            text_ = text_.replace(k, v)
        return text_

    hidden = {}
    result = [_hide(text)]
    yield result
    result[0] = _unhide(result[0])


class ToMarkdown:
    @staticmethod
    def deflist(name, type_, desc):
        type_parts = re.split(r"( *(?:, | of | or |, *default(?:=|\b)|, *optional\b) *)", type_ or "")
        type_parts[::2] = [f"`{s}`" if s else s for s in type_parts[::2]]
        type_ = "".join(type_parts)

        desc = desc or "&nbsp;"
        assert ToMarkdown.is_indented_4_spaces(desc)
        assert name or type_
        ret = ""
        if name:
            ret += f"**```{name.replace(', ', '```**, **```')}```**"
        if type_:
            ret += f" :&ensp;{type_}" if ret else type_
        ret += f"\n:   {desc}\n"
        return ret

    @staticmethod
    def is_indented_4_spaces(txt, _3_spaces_or_less=re.compile(r"\n\s{0,3}\S").search):
        return "\n" not in txt or not _3_spaces_or_less(txt)

    @staticmethod
    def fix_indent(name, type_, desc):
        if not ToMarkdown.is_indented_4_spaces(desc):
            desc = desc.replace("\n", "\n  ")
        return name, type_, desc

    @staticmethod
    def indent(indent, text, *, clean_first=False):
        if clean_first:
            text = inspect.cleandoc(text)
        return re.sub(r"\n", f"\n{indent}", indent + text.rstrip())

    @staticmethod
    def google(text):
        def _googledoc_sections(match):
            section, body = match.groups("")
            if not body:
                return match.group()
            body = textwrap.dedent(body)
            if section in ("Args", "Attributes"):
                body = re.compile(
                    r"^([\w*]+)(?: $begin:math:text$\(\[\\w\.\,\=\\\[\\\] \-\]\+\)$end:math:text$)?: "
                    r"((?:.*)(?:\n(?: {2,}.*|$))*)",
                    re.MULTILINE,
                ).sub(lambda m: ToMarkdown.deflist(*ToMarkdown.fix_indent(*m.groups())), inspect.cleandoc(f"\n{body}"))
            elif section in ("Returns", "Yields", "Raises", "Warns"):
                body = re.compile(
                    r"^()([\w.,$begin:math:display$$end:math:display$ ]+): " r"((?:.*)(?:\n(?: {2,}.*|$))*)",
                    re.MULTILINE,
                ).sub(lambda m: ToMarkdown.deflist(*ToMarkdown.fix_indent(*m.groups())), inspect.cleandoc(f"\n{body}"))
            return f"__{section}__\n\n{body}"

        return re.compile(
            r"^([a-zA-Z0-9_ \-]+):$\n{1}" r"( {2,}.*(?:\n?(?: {2,}.*|$))+)",
            re.MULTILINE,
        ).sub(_googledoc_sections, text)

    @staticmethod
    def raw_urls(text):
        pattern = re.compile(
            r"""
            (?P<code_span>(?<!`)(?P<fence>`+)(?!`).*?(?<!`)(?P=fence)(?!`))
            |
            (?P<markdown_link>\[.*?\]\(.*\))
            |
            (?<![<\"\'])(?P<url>(?:http|ftp)s?://[^>\s()]+(?:\([^>\s)]*\))*[^>\s)]*)
            """,
            re.VERBOSE,
        )
        return pattern.sub(lambda m: f'<{m.group("url")}>' if m.group("url") else m.group(), text)

    @staticmethod
    def convert(text, *, module=None):
        with fenced_code_blocks_hidden(text) as result:
            text = result[0]
            text = ToMarkdown.google(text)
            text = ToMarkdown.raw_urls(text)
            if module:
                _linkify = partial(linkify, module=module)
                text = re.sub(
                    r"(?P<inside_link>\[[^\]]*?)?"
                    r"(?:(?<!\\)(?:\\{2})+(?=`)|(?<!\\)(?P<fence>`+)"
                    r"(?P<code>.+?)(?<!`)"
                    r"(?P=fence)(?!`))",
                    lambda m: (m.group() if m.group("inside_link") or len(m.group("fence")) > 2 else _linkify(m)),
                    text,
                )
            result[0] = text
        return result[0]


class ReferenceWarning(UserWarning):
    pass


def linkify(match, *, module):
    try:
        refname = match.group("code")
    except IndexError:
        refname = match.group()

    if not re.match(r"^[\w.]+$", refname):
        return match.group()

    dobj = module.find_ident(refname)
    if isinstance(dobj, External):
        if "." not in refname:
            return match.group()
        module_part = module.find_ident(refname.split(".")[0])
        if not isinstance(module_part, External):
            warn(
                f'Code reference `{refname}` in module "{module.refname}" does not match any documented object.',
                ReferenceWarning,
                stacklevel=3,
            )
    return dobj.link


def format_github_link(dobj, user, repo, select_lines=True):
    try:
        commit = git_head_commit()
        abs_path = inspect.getfile(inspect.unwrap(dobj.obj))
        relpath = project_relative_path(abs_path)
        if os.name == "nt":
            relpath = relpath.replace("\\", "/")
        lines, start_line = inspect.getsourcelines(dobj.obj)
        if start_line and select_lines:
            start_line = start_line or 1
            end_line = start_line + len(lines) - 1
            template = "https://github.com/{user}/{repo}/blob/{commit}/{relpath}#L{start_line}-L{end_line}"
        else:
            template = "https://github.com/{user}/{repo}/blob/{commit}/{relpath}"
        return template.format(**locals())
    except Exception:
        if isinstance(dobj, (Variable, Function)) and getattr(dobj, "cls", None):
            return format_github_link(dobj.cls, user, repo, select_lines=False)
        if not isinstance(dobj, Module):
            return format_github_link(dobj.module, user, repo, select_lines=False)
        warn(f"format_github_link for {dobj.refname} failed:\n{traceback.format_exc()}")
        return None


@lru_cache()
def git_head_commit():
    process_args = ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(process_args, universal_newlines=True).strip()
    except OSError as error:
        warn(f"git executable not found on system:\n{error}")
    except subprocess.CalledProcessError as error:
        warn(
            "Ensure that generator is run within a git repository.\n"
            f"`{' '.join(process_args)}` failed with output:\n{error.output}"
        )
    return None


@lru_cache()
def git_project_root():
    for cmd in (["git", "rev-parse", "--show-superproject-working-tree"], ["git", "rev-parse", "--show-toplevel"]):
        try:
            p = subprocess.check_output(cmd, universal_newlines=True).rstrip("\r\n")
            if p:
                return os.path.normpath(p)
        except (subprocess.CalledProcessError, OSError):
            pass
    return None


@lru_cache()
def project_relative_path(absolute_path):
    from distutils.sysconfig import get_python_lib

    for prefix_path in (git_project_root() or os.getcwd(), get_python_lib()):
        common_path = os.path.commonpath([prefix_path, absolute_path])
        if os.path.samefile(common_path, prefix_path):
            return os.path.relpath(absolute_path, prefix_path)
    raise RuntimeError(
        f"absolute path {absolute_path!r} is not a descendant of the current working directory "
        "or of the system's python library."
    )


@lru_cache()
def str_template_fields(template):
    from string import Formatter

    return [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None]


def render_template(template_name, **kwargs):
    try:
        t = tpl_lookup.get_template(template_name)
    except TopLevelLookupException:
        paths = [path.join(p, template_name.lstrip("/")) for p in tpl_lookup.directories]
        raise OSError(f"No template found at any of: {', '.join(paths)}")
    return re.sub("\n\n\n+", "\n\n", t.render(**kwargs).strip())


@contextmanager
def open_write_file(filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            yield f
    except Exception:
        try:
            os.unlink(filename)
        except Exception:
            pass
        raise


def recursive_write_files(m, root_dir=".", clear=True, **kwargs):
    root_dir = path.join(root_dir, m.fname)
    if clear and os.path.exists(root_dir) and os.path.isdir(root_dir):
        shutil.rmtree(root_dir)

    if m.is_package:
        filepath = path.join(root_dir, "index.md")
    else:
        filepath = root_dir + ".md"

    dirpath = path.dirname(filepath)
    if not os.access(dirpath, os.R_OK):
        os.makedirs(dirpath)

    with open_write_file(filepath) as f:
        f.write(m.to_markdown(**kwargs))

    for submodule in m.submodules:
        recursive_write_files(submodule, root_dir=root_dir, clear=False, **kwargs)


def generate_api(
    module_name,
    root_dir=".",
    clear=True,
    docfilter=None,
    reload=False,
    skip_errors=False,
    curr_dir="api",
    **kwargs,
):
    m = Module(
        import_module(module_name, reload=reload),
        docfilter=docfilter,
        skip_errors=skip_errors,
        curr_dir=curr_dir,
    )
    link_inheritance()
    recursive_write_files(m, root_dir=root_dir, clear=clear, **kwargs)


if __name__ == "__main__":
    generate_api(
        "../vectorbt",
        root_dir="docs",
        get_icon=lambda module: None,
        get_tags=lambda module: set(),
        format_github_link=partial(format_github_link, user="polakowo", repo="vectorbt"),
    )
