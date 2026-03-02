"""
Service 3: Calculator & Unit Converter (Function Calling)
Provides safe mathematical evaluation and unit conversions.
The LLM calls these functions through OpenAI function calling — it never
performs the arithmetic itself, it delegates to these Python functions.
"""

import ast
import math
import operator

# ---------------------------------------------------------------------------
# Safe mathematical expression evaluator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1 / 3),
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "factorial": math.factorial,
}

_SAFE_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _eval_node(node):
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        return node.value

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_CONSTS:
            return _SAFE_CONSTS[name]
        if name in _SAFE_FUNCS:
            return _SAFE_FUNCS[name]
        raise ValueError(f"Unknown name: '{name}'")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _SAFE_OPS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported.")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Unknown function: '{func_name}'")
        args = [_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCS[func_name](*args)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def _safe_eval(expression: str):
    """Parse and evaluate a mathematical expression safely."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree.body)


def calculate_expression(expression: str, description: str = "") -> str:
    """
    Evaluate a mathematical expression and return a human-readable result.
    This is the primary tool function for Service 3.
    """
    try:
        result = _safe_eval(expression)

        # Format result: use int if it's a whole number
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        elif isinstance(result, float):
            formatted = f"{result:.6g}"
        else:
            formatted = str(result)

        desc_part = f" ({description})" if description else ""
        return f"Result{desc_part}: {formatted}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

_CONVERSIONS = {
    # Temperature (special — handled separately)
    # Distance
    ("km", "miles"):    lambda x: x * 0.621371,
    ("miles", "km"):    lambda x: x * 1.60934,
    ("km", "m"):        lambda x: x * 1000,
    ("m", "km"):        lambda x: x / 1000,
    ("m", "feet"):      lambda x: x * 3.28084,
    ("feet", "m"):      lambda x: x * 0.3048,
    ("m", "inches"):    lambda x: x * 39.3701,
    ("inches", "m"):    lambda x: x * 0.0254,
    ("feet", "inches"): lambda x: x * 12,
    ("inches", "feet"): lambda x: x / 12,
    ("miles", "feet"):  lambda x: x * 5280,
    ("feet", "miles"):  lambda x: x / 5280,
    # Weight / mass
    ("kg", "lbs"):      lambda x: x * 2.20462,
    ("lbs", "kg"):      lambda x: x * 0.453592,
    ("kg", "g"):        lambda x: x * 1000,
    ("g", "kg"):        lambda x: x / 1000,
    ("kg", "oz"):       lambda x: x * 35.274,
    ("oz", "kg"):       lambda x: x * 0.0283495,
    ("lbs", "oz"):      lambda x: x * 16,
    ("oz", "lbs"):      lambda x: x / 16,
    # Volume
    ("l", "ml"):        lambda x: x * 1000,
    ("ml", "l"):        lambda x: x / 1000,
    ("l", "gallons"):   lambda x: x * 0.264172,
    ("gallons", "l"):   lambda x: x * 3.78541,
    # Speed
    ("kmh", "mph"):     lambda x: x * 0.621371,
    ("mph", "kmh"):     lambda x: x * 1.60934,
    ("ms", "kmh"):      lambda x: x * 3.6,
    ("kmh", "ms"):      lambda x: x / 3.6,
    # Area
    ("km2", "miles2"):  lambda x: x * 0.386102,
    ("miles2", "km2"):  lambda x: x * 2.58999,
    # Time
    ("hours", "minutes"): lambda x: x * 60,
    ("minutes", "hours"): lambda x: x / 60,
    ("days", "hours"):    lambda x: x * 24,
    ("hours", "days"):    lambda x: x / 24,
    ("years", "days"):    lambda x: x * 365.25,
    ("days", "years"):    lambda x: x / 365.25,
}

_TEMP_CONVERSIONS = {
    ("celsius", "fahrenheit"):  lambda x: x * 9 / 5 + 32,
    ("fahrenheit", "celsius"):  lambda x: (x - 32) * 5 / 9,
    ("celsius", "kelvin"):      lambda x: x + 273.15,
    ("kelvin", "celsius"):      lambda x: x - 273.15,
    ("fahrenheit", "kelvin"):   lambda x: (x - 32) * 5 / 9 + 273.15,
    ("kelvin", "fahrenheit"):   lambda x: (x - 273.15) * 9 / 5 + 32,
}

# Aliases for common unit names
_ALIASES = {
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "mile": "miles", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "foot": "feet", "pound": "lbs", "pounds": "lbs",
    "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg",
    "ounce": "oz", "ounces": "oz", "litre": "l", "litres": "l",
    "liter": "l", "liters": "l", "milliliter": "ml", "milliliters": "ml",
    "gallon": "gallons", "hour": "hours", "minute": "minutes",
    "day": "days", "year": "years",
    "c": "celsius", "f": "fahrenheit", "k": "kelvin",
    "km/h": "kmh", "kph": "kmh", "mph": "mph", "m/s": "ms",
    "sq km": "km2", "km²": "km2", "sq miles": "miles2", "mi²": "miles2",
}


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a numeric value between supported units.
    This is the secondary tool function for Service 3.
    """
    try:
        from_norm = _ALIASES.get(from_unit.lower(), from_unit.lower())
        to_norm = _ALIASES.get(to_unit.lower(), to_unit.lower())

        # Check temperature first
        temp_key = (from_norm, to_norm)
        if temp_key in _TEMP_CONVERSIONS:
            result = _TEMP_CONVERSIONS[temp_key](value)
            return f"{value} {from_unit} = {result:.4g} {to_unit}"

        # Check general conversions
        key = (from_norm, to_norm)
        if key in _CONVERSIONS:
            result = _CONVERSIONS[key](value)
            return f"{value} {from_unit} = {result:.6g} {to_unit}"

        # Same unit
        if from_norm == to_norm:
            return f"{value} {from_unit} = {value} {to_unit} (same unit)"

        return (
            f"Conversion from '{from_unit}' to '{to_unit}' is not supported. "
            "Supported categories: temperature, distance, weight, volume, speed, area, time."
        )

    except Exception as e:
        return f"Error during unit conversion: {e}"
