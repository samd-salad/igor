"""Math and unit conversion command using simpleeval and pint."""
from .base import Command

try:
    from simpleeval import simple_eval, EvalWithCompoundTypes
    SIMPLEEVAL_AVAILABLE = True
except ImportError:
    SIMPLEEVAL_AVAILABLE = False

try:
    import pint
    ureg = pint.UnitRegistry()
    ureg.default_format = "~P"  # Short, pretty format
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

import math
import re


# Safe math functions for expression evaluation
MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}


def try_unit_conversion(expression: str) -> str | None:
    """Try to parse as a unit conversion."""
    if not PINT_AVAILABLE:
        return None

    # Normalize temperature unit names
    temp_aliases = {
        'fahrenheit': 'degF', 'farenheit': 'degF', 'f': 'degF',
        'celsius': 'degC', 'celcius': 'degC', 'c': 'degC',
        'kelvin': 'kelvin', 'k': 'kelvin',
    }

    # Common patterns: "X unit to unit", "X unit in unit", "convert X unit to unit"
    patterns = [
        r"(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*(.+?)\s+(?:to|in|as)\s+(.+)",
        r"(-?\d+(?:\.\d+)?)\s*(.+?)\s*->\s*(.+)",
    ]

    for pattern in patterns:
        match = re.match(pattern, expression.strip(), re.IGNORECASE)
        if match:
            value, from_unit, to_unit = match.groups()
            from_unit = from_unit.strip().lower()
            to_unit = to_unit.strip().lower()

            # Normalize temperature units
            from_unit = temp_aliases.get(from_unit, from_unit)
            to_unit = temp_aliases.get(to_unit, to_unit)

            try:
                # Use Quantity constructor for proper temperature handling
                quantity = ureg.Quantity(float(value), from_unit)
                result = quantity.to(to_unit)
                # Format nicely
                magnitude = result.magnitude
                if isinstance(magnitude, float) and magnitude == int(magnitude):
                    magnitude = int(magnitude)
                elif isinstance(magnitude, float):
                    magnitude = round(magnitude, 4)
                return f"{value} {from_unit} = {magnitude} {result.units:~P}"
            except Exception:
                pass

    return None


def try_math_expression(expression: str) -> str | None:
    """Try to evaluate as a math expression."""
    if not SIMPLEEVAL_AVAILABLE:
        # No fallback - simpleeval is required for safety
        return None

    try:
        # Replace common notations
        expr = expression.replace('^', '**')
        expr = expr.replace('×', '*').replace('÷', '/')

        evaluator = EvalWithCompoundTypes(functions=MATH_FUNCTIONS, names=MATH_FUNCTIONS)
        result = evaluator.eval(expr)
        return format_result(result)
    except Exception:
        return None


def format_result(result) -> str:
    """Format a numeric result nicely."""
    if isinstance(result, float):
        if result == int(result) and abs(result) < 1e15:
            return str(int(result))
        elif abs(result) < 0.0001 or abs(result) > 1e10:
            return f"{result:.6e}"
        else:
            # Remove trailing zeros
            formatted = f"{result:.10f}".rstrip('0').rstrip('.')
            return formatted
    return str(result)


class CalculateCommand(Command):
    name = "calculate"
    description = "Evaluate math expressions or convert units. Examples: '2^16', 'sqrt(144)', '72 fahrenheit to celsius', '5 miles in kilometers'"

    @property
    def parameters(self) -> dict:
        return {
            "expression": {
                "type": "string",
                "description": "Math expression or unit conversion (e.g., '2+2', '50 * 1.08', '100 fahrenheit to celsius', '5 km in miles')"
            }
        }

    def execute(self, expression: str) -> str:
        expression = expression.strip()
        if not expression:
            return "No expression provided"

        # Try unit conversion first
        conversion_result = try_unit_conversion(expression)
        if conversion_result:
            return conversion_result

        # Try math expression
        math_result = try_math_expression(expression)
        if math_result:
            return f"{expression} = {math_result}"

        # Check if libraries are missing
        missing = []
        if not SIMPLEEVAL_AVAILABLE:
            missing.append("simpleeval")
        if not PINT_AVAILABLE:
            missing.append("pint")

        if missing:
            return f"Could not evaluate. Missing libraries: {', '.join(missing)}"

        return f"Could not evaluate: '{expression}'"
