from conftest import *
from leantree.repl_adapter.interaction import LeanServer

def test_can_load_env(fixture_env: LeanServer):
    pass

def test_send_simple_command(fixture_env: LeanServer):
    result = fixture_env.send_command("def f := 2")
    assert result["env"] == 0
