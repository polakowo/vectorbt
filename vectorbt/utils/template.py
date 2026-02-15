# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for working with templates."""

from copy import copy
from string import Template
import ast

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import set_dict_item, get_func_arg_names, merge_dicts
from vectorbt.utils.docs import SafeToStr, prepare_for_doc

# Allowlist of attributes on mapped names that may be accessed/called from templates.
# Keys are names as they appear in the `mapping` (e.g. 'np'), values are sets of attribute names.
# Keep this intentionally small and conservative; expand only when necessary.
TEMPLATE_ALLOWED_ATTRS = {
    'np': {'prod'},
}


class Sub(SafeToStr):
    """Template to substitute parts of the string with the respective values from `mapping`.

    Returns a string."""

    def __init__(self, template: tp.Union[str, Template], mapping: tp.Optional[tp.Mapping] = None) -> None:
        self._template = template
        self._mapping = mapping

    @property
    def template(self) -> Template:
        """Template to be processed."""
        if not isinstance(self._template, Template):
            return Template(self._template)
        return self._template

    @property
    def mapping(self) -> tp.Mapping:
        """Mapping object passed to the initializer."""
        if self._mapping is None:
            return {}
        return self._mapping

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> str:
        """Substitute parts of `Sub.template` using `mapping`.

        Merges `mapping` and `Sub.mapping`.
        """
        mapping = merge_dicts(self.mapping, mapping)
        return self.template.substitute(mapping)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"template=\"{self.template.template}\", " \
               f"mapping={prepare_for_doc(self.mapping)})"


class Rep(SafeToStr):
    """Key to be replaced with the respective value from `mapping`."""

    def __init__(self, key: tp.Hashable, mapping: tp.Optional[tp.Mapping] = None) -> None:
        self._key = key
        self._mapping = mapping

    @property
    def key(self) -> tp.Hashable:
        """Key to be replaced."""
        return self._key

    @property
    def mapping(self) -> tp.Mapping:
        """Mapping object passed to the initializer."""
        if self._mapping is None:
            return {}
        return self._mapping

    def replace(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Replace `Rep.key` using `mapping`.

        Merges `mapping` and `Rep.mapping`."""
        mapping = merge_dicts(self.mapping, mapping)
        return mapping[self.key]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"key='{self.key}', " \
               f"mapping={prepare_for_doc(self.mapping)})"


class RepEval(SafeToStr):
    """Expression to be evaluated with `mapping` used as locals."""

    def __init__(self, expression: str, mapping: tp.Optional[tp.Mapping] = None) -> None:
        self._expression = expression
        self._mapping = mapping

    @property
    def expression(self) -> str:
        """Expression to be evaluated."""
        return self._expression

    @property
    def mapping(self) -> tp.Mapping:
        """Mapping object passed to the initializer."""
        if self._mapping is None:
            return {}
        return self._mapping

    def eval(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Evaluate `RepEval.expression` using `mapping`.

        Merges `mapping` and `RepEval.mapping`."""
        mapping = merge_dicts(self.mapping, mapping)
        # Use a restricted AST evaluator to avoid arbitrary code execution

        def _handle_constant(node):
            return node.value

        def _handle_name(node):
            if node.id in mapping:
                return mapping[node.id]
            raise NameError(f"name '{node.id}' is not defined")

        def _handle_binop(node):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"unsupported binary operator: {node.op}")

        def _handle_unaryop(node):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
            raise ValueError(f"unsupported unary operator: {node.op}")

        def _handle_boolop(node):
            values = [_eval_node(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError(f"unsupported boolean operator: {node.op}")

        def _handle_compare(node):
            left = _eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval_node(comparator)
                if isinstance(op, ast.Eq):
                    if not (left == right):
                        return False
                elif isinstance(op, ast.NotEq):
                    if not (left != right):
                        return False
                elif isinstance(op, ast.Is):
                    if not (left is right):
                        return False
                elif isinstance(op, ast.IsNot):
                    if not (left is not right):
                        return False
                elif isinstance(op, ast.In):
                    if not (left in right):
                        return False
                elif isinstance(op, ast.NotIn):
                    if not (left not in right):
                        return False
                elif isinstance(op, ast.Lt):
                    if not (left < right):
                        return False
                elif isinstance(op, ast.LtE):
                    if not (left <= right):
                        return False
                elif isinstance(op, ast.Gt):
                    if not (left > right):
                        return False
                elif isinstance(op, ast.GtE):
                    if not (left >= right):
                        return False
                else:
                    raise ValueError(f"unsupported comparison operator: {op}")
                left = right
            return True

        def _handle_attribute(node):
            # Only allow attribute access on top-level names from mapping and only allowed attrs
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                if base_name not in mapping:
                    raise NameError(f"name '{base_name}' is not defined")
                allowed = TEMPLATE_ALLOWED_ATTRS.get(base_name, set())
                if node.attr not in allowed:
                    raise ValueError(f"access to attribute '{node.attr}' of '{base_name}' is not allowed")
                base_obj = mapping[base_name]
                return getattr(base_obj, node.attr)
            raise ValueError("attribute access is only allowed on top-level mapped names")

        def _handle_call(node):
            # Allow calls only when calling an attribute of a mapped name, e.g. np.prod(...)
            func_node = node.func
            if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name):
                base_name = func_node.value.id
                if base_name not in mapping:
                    raise NameError(f"name '{base_name}' is not defined")
                allowed = TEMPLATE_ALLOWED_ATTRS.get(base_name, set())
                if func_node.attr not in allowed:
                    raise ValueError(f"call to '{func_node.attr}' of '{base_name}' is not allowed")
                base_obj = mapping[base_name]
                func = getattr(base_obj, func_node.attr)
                if not callable(func):
                    raise ValueError(f"object '{func_node.attr}' of '{base_name}' is not callable")
                args = [_eval_node(a) for a in node.args]
                kwargs = {kw.arg: _eval_node(kw.value) for kw in node.keywords}
                return func(*args, **kwargs)
            raise ValueError("only calls to mapped attributes are allowed")

        def _handle_subscript(node):
            val = _eval_node(node.value)
            # Handle slice objects properly
            s = node.slice
            if isinstance(s, ast.Slice):
                lower = _eval_node(s.lower) if s.lower is not None else None
                upper = _eval_node(s.upper) if s.upper is not None else None
                step = _eval_node(s.step) if s.step is not None else None
                return val[slice(lower, upper, step)]
            # Tuple of indices (multi-dimensional)
            if isinstance(s, ast.Tuple):
                idx = tuple(_eval_node(elt) for elt in s.elts)
                return val[idx]
            # Other single index types
            idx = _eval_node(s)
            return val[idx]

        def _handle_list(node):
            result = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    val = _eval_node(elt.value)
                    try:
                        result.extend(list(val))
                    except Exception:
                        raise ValueError("can't unpack starred expression")
                else:
                    result.append(_eval_node(elt))
            return result

        def _handle_tuple(node):
            result = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    val = _eval_node(elt.value)
                    try:
                        result.extend(list(val))
                    except Exception:
                        raise ValueError("can't unpack starred expression")
                else:
                    result.append(_eval_node(elt))
            return tuple(result)

        def _handle_joinedstr(node):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                elif isinstance(v, ast.FormattedValue):
                    val = _eval_node(v.value)
                    parts.append('' if val is None else str(val))
                else:
                    parts.append(str(_eval_node(v)))
            return ''.join(parts)

        def _handle_dict(node):
            return {_eval_node(k): _eval_node(v) for k, v in zip(node.keys, node.values)}

        def _handle_ifexp(node):
            # Ternary conditional expression: body if test else orelse
            test_val = _eval_node(node.test)
            if test_val:
                return _eval_node(node.body)
            return _eval_node(node.orelse)

        handlers = {
            ast.Constant: _handle_constant,
            ast.Name: _handle_name,
            ast.BinOp: _handle_binop,
            ast.UnaryOp: _handle_unaryop,
            ast.BoolOp: _handle_boolop,
            ast.Compare: _handle_compare,
            ast.Attribute: _handle_attribute,
            ast.Call: _handle_call,
            ast.Subscript: _handle_subscript,
            ast.List: _handle_list,
            ast.Tuple: _handle_tuple,
            ast.Dict: _handle_dict,
            ast.IfExp: _handle_ifexp,
            ast.JoinedStr: _handle_joinedstr,
        }

        def _eval_node(node):
            handler = handlers.get(type(node))
            if handler is not None:
                return handler(node)
            raise ValueError(f"unsupported expression: {type(node).__name__}")

        parsed = ast.parse(self.expression, mode="eval")
        return _eval_node(parsed.body)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"expression=\"{self.expression}\", " \
               f"mapping={prepare_for_doc(self.mapping)})"


class RepFunc(SafeToStr):
    """Function to be called with argument names from `mapping`."""

    def __init__(self, func: tp.Callable, mapping: tp.Optional[tp.Mapping] = None) -> None:
        self._func = func
        self._mapping = mapping

    @property
    def func(self) -> tp.Callable:
        """Replacement function to be called."""
        return self._func

    @property
    def mapping(self) -> tp.Mapping:
        """Mapping object passed to the initializer."""
        if self._mapping is None:
            return {}
        return self._mapping

    def call(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Call `RepFunc.func` using `mapping`.

        Merges `mapping` and `RepFunc.mapping`."""
        mapping = merge_dicts(self.mapping, mapping)
        func_arg_names = get_func_arg_names(self.func)
        func_kwargs = dict()
        for k, v in mapping.items():
            if k in func_arg_names:
                func_kwargs[k] = v
        return self.func(**func_kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"func={self.func}, " \
               f"mapping={prepare_for_doc(self.mapping)})"


def has_templates(obj: tp.Any) -> tp.Any:
    """Check if the object has any templates."""
    if isinstance(obj, RepFunc):
        return True
    if isinstance(obj, RepEval):
        return True
    if isinstance(obj, Rep):
        return True
    if isinstance(obj, Sub):
        return True
    if isinstance(obj, Template):
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            if has_templates(v):
                return True
    if isinstance(obj, (tuple, list, set, frozenset)):
        for v in obj:
            if has_templates(v):
                return True
    return False


def deep_substitute(obj: tp.Any,
                    mapping: tp.Optional[tp.Mapping] = None,
                    safe: bool = False,
                    make_copy: bool = True) -> tp.Any:
    """Traverses the object recursively and, if any template found, substitutes it using a mapping.

    Traverses tuples, lists, dicts and (frozen-)sets. Does not look for templates in keys.

    If `safe` is True, won't raise an error but return the original template.

    !!! note
        If the object is deep (such as a dict or a list), creates a copy of it if any template found inside,
        thus loosing the reference to the original. Make sure to do a deep or hybrid copy of the object
        before proceeding for consistent behavior, or disable `make_copy` to override the original in place.

    Usage:
        ```pycon
        >>> import vectorbt as vbt

        >>> vbt.deep_substitute(vbt.Sub('$key', {'key': 100}))
        100
        >>> vbt.deep_substitute(vbt.Sub('$key', {'key': 100}), {'key': 200})
        200
        >>> vbt.deep_substitute(vbt.Sub('$key$key'), {'key': 100})
        100100
        >>> vbt.deep_substitute(vbt.Rep('key'), {'key': 100})
        100
        >>> vbt.deep_substitute([vbt.Rep('key'), vbt.Sub('$key$key')], {'key': 100})
        [100, '100100']
        >>> vbt.deep_substitute(vbt.RepFunc(lambda key: key == 100), {'key': 100})
        True
        >>> vbt.deep_substitute(vbt.RepEval('key == 100'), {'key': 100})
        True
        >>> vbt.deep_substitute(vbt.RepEval('key == 100', safe=False))
        NameError: name 'key' is not defined
        >>> vbt.deep_substitute(vbt.RepEval('key == 100', safe=True))
        <vectorbt.utils.template.RepEval at 0x7fe3ad2ab668>
        ```
    """
    if mapping is None:
        mapping = {}
    if not has_templates(obj):
        return obj
    try:
        if isinstance(obj, RepFunc):
            return obj.call(mapping)
        if isinstance(obj, RepEval):
            return obj.eval(mapping)
        if isinstance(obj, Rep):
            return obj.replace(mapping)
        if isinstance(obj, Sub):
            return obj.substitute(mapping)
        if isinstance(obj, Template):
            return obj.substitute(mapping)
        if isinstance(obj, dict):
            if make_copy:
                obj = copy(obj)
            for k, v in obj.items():
                set_dict_item(obj, k, deep_substitute(v, mapping=mapping, safe=safe), force=True)
            return obj
        if isinstance(obj, list):
            if make_copy:
                obj = copy(obj)
            for i in range(len(obj)):
                obj[i] = deep_substitute(obj[i], mapping=mapping, safe=safe)
            return obj
        if isinstance(obj, (tuple, set, frozenset)):
            result = []
            for o in obj:
                result.append(deep_substitute(o, mapping=mapping, safe=safe))
            if checks.is_namedtuple(obj):
                return type(obj)(*result)
            return type(obj)(result)
    except Exception as e:
        if not safe:
            raise e
    return obj
