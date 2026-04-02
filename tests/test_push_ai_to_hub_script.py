from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "push-ai-to-hub.ps1"


def _run_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        capture_output=True,
        check=check,
    )


def _run_script(repo_root: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run_command(
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT_PATH),
        "-RepoRoot",
        str(repo_root),
        check=check,
    )


def _init_monorepo(repo_root: Path, remote_path: Path) -> None:
    (repo_root / "AI" / "app").mkdir(parents=True)
    (repo_root / "AI" / "docs").mkdir(parents=True)
    (repo_root / "AI" / "lab" / "merchant_info").mkdir(parents=True)
    (repo_root / "AI" / "README.md").write_text("project readme\n", encoding="utf-8")
    (repo_root / "AI" / "app" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "AI" / ".env").write_text("HF_TOKEN=hf_should_not_be_exported\n", encoding="utf-8")
    (repo_root / "AI" / "docs" / "private.md").write_text("ignore me\n", encoding="utf-8")
    (repo_root / "AI" / "lab" / "merchant_info" / "sample.csv").write_text("id,name\n1,test\n", encoding="utf-8")

    _run_command("git", "init", str(repo_root))
    _run_command("git", "-C", str(repo_root), "config", "user.name", "Test User")
    _run_command("git", "-C", str(repo_root), "config", "user.email", "test@example.com")
    _run_command("git", "init", "--bare", str(remote_path))
    _run_command("git", "-C", str(repo_root), "remote", "add", "hub", str(remote_path))


def test_push_ai_to_hub_exports_only_clean_ai_tree(tmp_path: Path) -> None:
    repo_root = tmp_path / "workspace"
    remote_path = tmp_path / "hub.git"
    _init_monorepo(repo_root, remote_path)

    result = _run_script(repo_root)

    assert result.returncode == 0, result.stderr or result.stdout

    ls_tree = _run_command(
        "git",
        "--git-dir",
        str(remote_path),
        "ls-tree",
        "-r",
        "--name-only",
        "dev/ai",
    )
    exported_paths = set(ls_tree.stdout.splitlines())

    assert "README.md" in exported_paths
    assert "app/main.py" in exported_paths
    assert ".env" not in exported_paths
    assert "docs/private.md" not in exported_paths
    assert "lab/merchant_info/sample.csv" not in exported_paths


def test_push_ai_to_hub_stops_when_secret_exists_in_exported_tree(tmp_path: Path) -> None:
    repo_root = tmp_path / "workspace"
    remote_path = tmp_path / "hub.git"
    _init_monorepo(repo_root, remote_path)
    fake_token = "hf_" + "abcdefghijklmnopqrstuvwxyz123456"
    (repo_root / "AI" / "app" / "token.py").write_text(
        f"HF_TOKEN = '{fake_token}'\n",
        encoding="utf-8",
    )

    result = _run_script(repo_root, check=False)

    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    assert result.returncode != 0
    assert "secret" in combined_output

    show_ref = _run_command(
        "git",
        "--git-dir",
        str(remote_path),
        "show-ref",
        check=False,
    )
    assert show_ref.stdout.strip() == ""
