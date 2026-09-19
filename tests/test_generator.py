from src.generator import FALLBACK, generate_answer


def test_generator_returns_fallback_without_context():
    assert generate_answer("What is diabetes?", []) == FALLBACK
