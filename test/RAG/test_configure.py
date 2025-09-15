import configure
import builtins
import pytest

def test_get_user_choice_valid(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    result = configure.get_user_choice("Pick one:", ["a", "b", "c"])
    assert result == "b"

def test_get_user_choice_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["5", "1"])  # اول اشتباه می‌زنه بعد درست
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = configure.get_user_choice("Pick one:", ["x", "y"])
    captured = capsys.readouterr()
    assert "Invalid choice" in captured.out
    assert result == "x"

def test_get_text_input_with_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    result = configure.get_text_input("Enter something", default="hi")
    assert result == "hi"

def test_get_text_input_without_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "custom")
    result = configure.get_text_input("Enter name")
    assert result == "custom"

def test_run_interactive_configuration_cancel(monkeypatch, capsys):
    inputs = iter([
        "MyTestProject",
        "1",
        "128",
        "2",
        "my-local-llm",
        "some-embedder",
        "no"
    ])

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    configure.run_interactive_configuration()
    captured = capsys.readouterr()
    assert "Build cancelled." in captured.out