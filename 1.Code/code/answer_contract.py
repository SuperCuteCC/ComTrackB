"""Student-side whole-answer validation. Does not implement official scoring."""
import math
import re
from decimal import Decimal, InvalidOperation

POLICY_VERSION = 'unit25-whole-answer-v1'
NUMBER = r'[+\-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?'
NUMBER_ONLY = re.compile(r'\s*(' + NUMBER + r')\s*', re.ASCII)
ANSWER = re.compile(r'\s*(' + NUMBER + r')\s*(.+?)\s*', re.ASCII)
BASE = r'(?:millimeters?|millimetres?|centimeters?|centimetres?|kilometers?|kilometres?|meters?|metres?|mm|cm|km|m|seconds?|s)'
SUFFIX = r'(?:\s*/\s*(?:seconds?|s)(?:\s*(?:\^\s*2|\*\*\s*2|²|2))?|\s+per\s+seconds?(?:\s+squared)?)?'
REQUESTED = re.compile(r'\bin\s+(' + BASE + SUFFIX + r')(?![a-zA-Z0-9/^²³*])', re.I)
WORDS = {
    'millimeter': 'mm', 'millimeters': 'mm', 'millimetre': 'mm', 'millimetres': 'mm',
    'centimeter': 'cm', 'centimeters': 'cm', 'centimetre': 'cm', 'centimetres': 'cm',
    'kilometer': 'km', 'kilometers': 'km', 'kilometre': 'km', 'kilometres': 'km',
    'meter': 'm', 'meters': 'm', 'metre': 'm', 'metres': 'm',
    'second': 's', 'seconds': 's',
}


def canonical_unit(text):
    """Recognize equal-spelling SI units; never convert numerical scales."""
    s = text.strip().lower().replace('−', '-').replace('⁻²', '^-2').replace('⁻¹', '^-1')
    s = s.replace('²', '^2').replace('¹', '^1').replace('**', '^')
    s = re.sub(r'\b[a-z]+\b', lambda m: WORDS.get(m[0], m[0]), s)
    s = re.sub(r'\bper\b', '/', s)
    s = re.sub(r'\bsquared\b', '^2', s)
    match = re.fullmatch(r'(mm|cm|km|m|s)(.*)', s)
    if not match:
        return None
    base, suffix = match.groups()
    suffix = re.sub(r'\s+', '', suffix)
    if suffix.startswith(('·','⋅','*')):suffix=suffix[1:]
    if not suffix:
        return base
    if base == 's':
        return None
    if suffix in ('/s', '/s^1', 's^-1', 's^(-1)', 's^{-1}'):
        return base + '/s'
    if suffix in ('/s^2', '/s2', '/s/s', '/(s^2)', '/(s)^2', 's^-2', 's^(-2)', 's^{-2}'):
        return base + '/s^2'
    return None


def requested_unit(question):
    # Only this question's explicit "in <unit>" clauses; never priors or fallback m.
    found = [canonical_unit(m[1]) for m in REQUESTED.finditer(question)]
    if not found or None in found or len(set(found)) != 1:
        raise ValueError('Missing or ambiguous requested unit: ' + repr(question))
    return found[0]


def significant_digits(number):
    mantissa = re.split('[eE]', number.lstrip('+-'))[0]
    digits = mantissa.replace('.', '').lstrip('0').rstrip('0')
    return max(1, len(digits))


def nonnegative_quantity(question):
    q = question.casefold()
    # Signed velocity, displacement, coordinate and acceleration remain allowed.
    if re.search(r'\bmagnitude\b',q):return True
    if re.search(r'\b(velocity|acceleration|displacement|coordinate|position)\b',q):return False
    return bool(re.search(r'\b(speed|distance|length|width|height|diameter|radius|circumference|thickness|perimeter)\b',q))


def validate_answer(raw, question, *, new_tokens=None, ended_with_eos=None,
                    max_new_tokens=64, enforce_six_digits=False, require_known_eos=False):
    expected = requested_unit(question)
    normalized = raw.strip().replace('−', '-')
    reasons = []
    matched = ANSWER.fullmatch(normalized)
    number = unit = decimal = value = None
    if NUMBER_ONLY.fullmatch(normalized):
        reasons.append('MISSING_UNIT')
    elif not matched:
        reasons.append('MALFORMED_WHOLE_ANSWER')
    else:
        number, unit_text = matched.groups()
        unit = canonical_unit(unit_text)
        if unit is None:
            reasons.append('MALFORMED_OR_UNRECOGNIZED_UNIT')
        elif unit != expected:
            reasons.append('UNIT_CONFLICT')
        try:
            decimal = Decimal(number)
            value = float(decimal)
            if not decimal.is_finite() or not math.isfinite(value):
                reasons.append('NONFINITE_VALUE')
            elif decimal != 0 and value == 0:
                reasons.append('FLOAT_UNDERFLOW')
            elif decimal < 0 and nonnegative_quantity(question):
                reasons.append('NEGATIVE_SCALAR_MAGNITUDE')
        except (InvalidOperation, ValueError, OverflowError):
            reasons.append('INVALID_NUMBER')
        if enforce_six_digits and significant_digits(number) > 6:
            reasons.append('MORE_THAN_SIX_SIGNIFICANT_DIGITS')
    if re.search(r'(?<![a-z])(?:nan|[+\-]?inf(?:inity)?)(?![a-z])', normalized, re.I):
        reasons.append('NONFINITE_LITERAL')
    reached_limit = new_tokens is not None and new_tokens >= max_new_tokens
    if reached_limit:
        reasons.append('TOKEN_LIMIT_REACHED')
    if ended_with_eos is False:
        reasons.append('NO_NORMAL_EOS')
    if require_known_eos and ended_with_eos is not True:
        reasons.append('NORMAL_EOS_NOT_CONFIRMED')
    if require_known_eos and (type(new_tokens) is not int or not 0 < new_tokens <= max_new_tokens):
        reasons.append('INVALID_OR_MISSING_TOKEN_COUNT')
    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons
    return dict(accepted=accepted, reasons=reasons, requested_unit=expected,
                response_unit=unit, missing_unit='MISSING_UNIT' in reasons,
                token_limit_reached=reached_limit, ended_with_eos=ended_with_eos,
                numeric_string=number if accepted else None,
                decimal_string=str(decimal) if accepted else None,
                parsed_value=value if accepted else None,
                scale_conversion_applied=False)


def validate_new_generation(raw, question, *, new_tokens, ended_with_eos):
    """Only entry point for adopting a newly generated value under the frozen rule."""
    return validate_answer(raw, question, new_tokens=new_tokens,
        ended_with_eos=ended_with_eos, max_new_tokens=64,
        enforce_six_digits=True, require_known_eos=True)
