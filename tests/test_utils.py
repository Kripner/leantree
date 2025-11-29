"""
Tests for utility functions in leantree.utils
"""
import sys
import traceback

from leantree.utils import serialize_exception, deserialize_exception, RemoteException


def test_serialize_basic_exception():
    """Test serializing a basic picklable exception."""
    print("Running test_serialize_basic_exception...")
    exception = ValueError("Test error message")
    result = serialize_exception(exception)

    assert "exception" in result, "Should contain pickled exception"
    assert isinstance(result["exception"], str), "Exception should be base64-encoded string"
    assert len(result["exception"]) > 0, "Exception should not be empty"
    print("✓ test_serialize_basic_exception passed")


def test_deserialize_basic_exception():
    """Test deserializing a basic picklable exception."""
    print("Running test_deserialize_basic_exception...")
    original_exception = ValueError("Test error message")
    serialized = serialize_exception(original_exception)

    deserialized = deserialize_exception(serialized, "Wrapper message")

    assert isinstance(deserialized, RemoteException), "Should return RemoteException"
    assert isinstance(deserialized, RuntimeError), "Should be a RuntimeError subclass"
    assert deserialized.__cause__ is not None, "Should have original exception as cause"
    assert isinstance(deserialized.__cause__, ValueError), "Cause should be ValueError"
    assert str(deserialized.__cause__) == "Test error message", "Cause message should match"
    assert "Wrapper message" in str(deserialized), "Should contain wrapper message"
    assert deserialized.traceback_str is not None, "Should have traceback string"
    print("✓ test_deserialize_basic_exception passed")


def test_round_trip_exception():
    """Test round-trip serialization/deserialization."""
    print("Running test_round_trip_exception...")
    original_exception = RuntimeError("Original error")

    # Serialize
    serialized = serialize_exception(original_exception)

    # Deserialize
    deserialized = deserialize_exception(serialized)

    # Check that the original exception is preserved as cause
    assert deserialized.__cause__ is not None
    assert type(deserialized.__cause__) == type(original_exception)
    assert str(deserialized.__cause__) == str(original_exception)
    print("✓ test_round_trip_exception passed")


def test_custom_exception_type():
    """Test serializing/deserializing custom exception types."""
    print("Running test_custom_exception_type...")

    class CustomError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    original = CustomError("Custom error", code=42)
    serialized = serialize_exception(original)
    deserialized = deserialize_exception(serialized)

    assert deserialized.__cause__ is not None
    assert isinstance(deserialized.__cause__, CustomError)
    assert deserialized.__cause__.code == 42, "Custom attributes should be preserved"
    print("✓ test_custom_exception_type passed")


def test_exception_with_traceback():
    """Test that exceptions with tracebacks are preserved."""
    print("Running test_exception_with_traceback...")

    def raise_error():
        raise ValueError("Error with traceback")

    try:
        raise_error()
    except ValueError as e:
        original_exception = e
        serialized = serialize_exception(original_exception)
        deserialized = deserialize_exception(serialized)

        assert deserialized.__cause__ is not None
        assert isinstance(deserialized.__cause__, ValueError)
        # The traceback should be preserved in the exception
        assert deserialized.__cause__.__traceback__ is not None

    print("✓ test_exception_with_traceback passed")


def test_non_picklable_exception_fallback():
    """Test fallback to exception_info for non-picklable exceptions."""
    print("Running test_non_picklable_exception_fallback...")

    # Create an exception with non-picklable attributes
    class NonPicklableError(Exception):
        def __init__(self, message):
            super().__init__(message)
            # Add a non-picklable attribute (like a lambda or file handle)
            self.non_picklable = lambda x: x

    original = NonPicklableError("Non-picklable error")
    serialized = serialize_exception(original)

    # Should fall back to exception_info
    assert "exception_info" in serialized, "Should use exception_info fallback"
    assert "pickle_error" in serialized, "Should indicate pickle error"
    assert serialized["exception_info"]["type"] == "NonPicklableError"
    assert serialized["exception_info"]["message"] == "Non-picklable error"
    assert "traceback" in serialized["exception_info"]

    # Deserialize should create a proxy exception
    deserialized = deserialize_exception(serialized, "Wrapper")
    assert isinstance(deserialized, RemoteException)
    assert isinstance(deserialized, RuntimeError)
    assert deserialized.__cause__ is not None
    # The proxy exception should have the same type name
    assert type(deserialized.__cause__).__name__ == "NonPicklableError"
    assert deserialized.traceback_str is not None, "Should have traceback string"
    assert "Server traceback" in str(deserialized), "Should include server traceback in message"

    print("✓ test_non_picklable_exception_fallback passed")


def test_deserialize_with_exception_info_only():
    """Test deserializing when only exception_info is present."""
    print("Running test_deserialize_with_exception_info_only...")

    error_data = {
        "exception_info": {
            "type": "ValueError",
            "message": "Test error",
            "traceback": ["Traceback (most recent call last):\n", "  File \"test.py\", line 1\n",
                          "ValueError: Test error\n"],
        }
    }

    deserialized = deserialize_exception(error_data, "Wrapper message")

    assert isinstance(deserialized, RemoteException)
    assert isinstance(deserialized, RuntimeError)
    assert deserialized.__cause__ is not None
    assert type(deserialized.__cause__).__name__ == "ValueError"
    assert str(deserialized.__cause__) == "Test error"
    assert deserialized.traceback_str is not None
    assert "Traceback" in deserialized.traceback_str
    assert "Server traceback" in str(deserialized), "Should include server traceback in message"

    print("✓ test_deserialize_with_exception_info_only passed")


def test_deserialize_empty_error_data():
    """Test deserializing when error_data has no exception info."""
    print("Running test_deserialize_empty_error_data...")

    error_data = {}
    deserialized = deserialize_exception(error_data, "Custom error message")

    assert isinstance(deserialized, RemoteException)
    assert isinstance(deserialized, RuntimeError)
    assert str(deserialized) == "Custom error message"
    assert deserialized.__cause__ is None

    print("✓ test_deserialize_empty_error_data passed")


def test_deserialize_default_error_message():
    """Test deserializing with default error message."""
    print("Running test_deserialize_default_error_message...")

    exception = ValueError("Test")
    serialized = serialize_exception(exception)
    deserialized = deserialize_exception(serialized)  # No error_message provided

    assert isinstance(deserialized, RemoteException)
    assert isinstance(deserialized, RuntimeError)
    assert "Error from remote server" in str(deserialized)
    assert deserialized.traceback_str is not None, "Should have traceback string"

    print("✓ test_deserialize_default_error_message passed")


def test_exception_chaining():
    """Test that exception chaining works correctly."""
    print("Running test_exception_chaining...")

    try:
        try:
            raise ValueError("Inner error")
        except ValueError as e:
            raise RuntimeError("Outer error") from e
    except RuntimeError as e:
        original = e
        serialized = serialize_exception(original)
        deserialized = deserialize_exception(serialized)

        assert deserialized.__cause__ is not None
        assert isinstance(deserialized.__cause__, RuntimeError)
        # The original exception chain should be preserved
        assert deserialized.__cause__.__cause__ is not None
        assert isinstance(deserialized.__cause__.__cause__, ValueError)

    print("✓ test_exception_chaining passed")


def test_multiple_exception_types():
    """Test serializing/deserializing various exception types."""
    print("Running test_multiple_exception_types...")

    exception_types = [
        ValueError("Value error"),
        TypeError("Type error"),
        KeyError("Key error"),
        IndexError("Index error"),
        FileNotFoundError("File not found"),
        AssertionError("Assertion failed"),
    ]

    for original in exception_types:
        serialized = serialize_exception(original)
        deserialized = deserialize_exception(serialized)

        assert deserialized.__cause__ is not None
        assert type(deserialized.__cause__) == type(original)
        assert str(deserialized.__cause__) == str(original)

    print("✓ test_multiple_exception_types passed")


def test_exception_with_unicode():
    """Test serializing exceptions with unicode characters."""
    print("Running test_exception_with_unicode...")

    exception = ValueError("Error with unicode: 测试 🚀 émoji")
    serialized = serialize_exception(exception)
    deserialized = deserialize_exception(serialized)

    assert deserialized.__cause__ is not None
    assert str(deserialized.__cause__) == "Error with unicode: 测试 🚀 émoji"

    print("✓ test_exception_with_unicode passed")


def run_all_tests():
    """Run all tests sequentially."""
    tests = [
        ("test_serialize_basic_exception", test_serialize_basic_exception),
        ("test_deserialize_basic_exception", test_deserialize_basic_exception),
        ("test_round_trip_exception", test_round_trip_exception),
        ("test_custom_exception_type", test_custom_exception_type),
        ("test_exception_with_traceback", test_exception_with_traceback),
        ("test_non_picklable_exception_fallback", test_non_picklable_exception_fallback),
        ("test_deserialize_with_exception_info_only", test_deserialize_with_exception_info_only),
        ("test_deserialize_empty_error_data", test_deserialize_empty_error_data),
        ("test_deserialize_default_error_message", test_deserialize_default_error_message),
        ("test_exception_chaining", test_exception_chaining),
        ("test_multiple_exception_types", test_multiple_exception_types),
        ("test_exception_with_unicode", test_exception_with_unicode),
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running {test_name}...")
        print(f"{'=' * 60}")

        try:
            # Run the test
            test_func()
            print(f"✓ {test_name} passed")
        except AssertionError as e:
            print(f"\n❌ {test_name} failed with assertion error:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return 1
        except Exception as e:
            print(f"\n❌ {test_name} failed with error:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
