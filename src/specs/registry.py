from .roman import RomanNumeralSpec
from .bit import BitBinarySpec
from .unit import UnitConversionSpec
from .gravity import GravitySpec
from .equation import EquationSpec
from .text import TextDecryptSpec

SPEC_REGISTRY = {
    "roman_numeral": RomanNumeralSpec(),
    "bit_binary": BitBinarySpec(),
    "unit_conversion": UnitConversionSpec(),
    "gravity": GravitySpec(),
    "equation": EquationSpec(),
    "text_decrypt": TextDecryptSpec(),
}