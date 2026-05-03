from src.json_literal_validators import StringValidator


def test_string_validator_rejects_nested_json_and_gt_junk_prefixes() -> None:
    validator = StringValidator()

    assert not validator.is_valid_prefix('"{')
    assert not validator.is_valid_prefix('">{')
    assert not validator.is_valid_prefix('"> ')
    assert validator.is_valid_prefix('">')
    assert validator.is_valid_prefix('">="')
    assert validator.is_valid_prefix('"[AEIOUaeiou]')
    assert validator.is_valid_prefix('"hello')
