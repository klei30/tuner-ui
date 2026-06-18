import subprocess
import sys
import os


def test_backend_imports_without_tinker_or_chz():
    env = os.environ.copy()
    env.pop("TINKER_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'backend'); import main; print(main.TINKER_AVAILABLE)",
        ],
        cwd=".",
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "False" in result.stdout


def test_backend_refuses_simulation_when_tinker_key_is_set_without_stack():
    env = os.environ.copy()
    env["TINKER_API_KEY"] = "real-looking-test-key"
    env["ALLOW_ANON"] = "false"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'backend'); import main",
        ],
        cwd=".",
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "TINKER_API_KEY is set" in combined_output
