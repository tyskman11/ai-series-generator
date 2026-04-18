#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    from pipeline_common import SCRIPT_DIR, error, headline, info, ok, warn
except Exception:
    SCRIPT_DIR = Path(__file__).resolve().parent

    def info(text: str) -> None:
        print(f"[INFO] {text}")

    def ok(text: str) -> None:
        print(f"[OK]   {text}")

    def warn(text: str) -> None:
        print(f"[WARN] {text}")

    def error(text: str) -> None:
        print(f"[ERROR] {text}")

    def headline(text: str) -> None:
        print()
        print("=" * 72)
        print(text)
        print("=" * 72)


class GitHubApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


REMOTE_PATTERNS = (
    re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"),
    re.compile(r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?/?$"),
)

DEFAULT_OWNER = "tyskman11"
DEFAULT_REPO = "ai-series-generator"
DEFAULT_REMOTE_URL = f"https://github.com/{DEFAULT_OWNER}/{DEFAULT_REPO}.git"
DEFAULT_GIT_USER_NAME = os.environ.get("GIT_USER_NAME", DEFAULT_OWNER)
DEFAULT_GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "baumscarry@gmail.com")
BLOCKED_GIT_COMMANDS = {"clone", "fetch", "pull"}
REPO_ABOUT_DESCRIPTION = (
    "Local AI pipeline for learning from TV episodes and generating new preview episodes; "
    "all scripts are AI-generated with GPT-5.4."
)


def ensure_upload_only_git_args(args: list[str]) -> None:
    if not args:
        return
    first = str(args[0]).strip().lower()
    if first in BLOCKED_GIT_COMMANDS:
        blocked = ", ".join(sorted(BLOCKED_GIT_COMMANDS))
        raise RuntimeError(
            "This sync helper is strictly local -> GitHub only. "
            f"Git commands for downloading are blocked: {blocked}."
        )


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    ensure_upload_only_git_args(args)
    result = subprocess.run(
        ["git", *args],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        message = (result.stdout or "").strip() or f"git {' '.join(args)} failed."
        raise RuntimeError(message)
    return result


def git_output(args: list[str]) -> str:
    return (run_git(args, check=False).stdout or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror only local scripts and the README to GitHub, never from GitHub to local.",
    )
    parser.add_argument(
        "--owner",
        default=os.environ.get("GITHUB_OWNER", DEFAULT_OWNER),
        help=f"GitHub owner or organization. Default: {DEFAULT_OWNER}",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPO", DEFAULT_REPO),
        help=f"GitHub repository name. Default: {DEFAULT_REPO}",
    )
    parser.add_argument("--branch", default="main", help="Target branch on GitHub. Default: main")
    parser.add_argument(
        "--message",
        default="Automatic script update",
        help="Commit message. Default: Automatic script update",
    )
    parser.add_argument(
        "--git-user-name",
        default=DEFAULT_GIT_USER_NAME,
        help="Git name for local commits. Alternatively via GIT_USER_NAME.",
    )
    parser.add_argument(
        "--git-user-email",
        default=DEFAULT_GIT_USER_EMAIL,
        help="Git email for local commits. Alternatively via GIT_USER_EMAIL.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "",
        help="GitHub token. Alternatively via GITHUB_TOKEN or GH_TOKEN.",
    )
    parser.add_argument(
        "--private",
        dest="private_repo",
        action="store_true",
        default=env_bool("GITHUB_PRIVATE", True),
        help="Create a new repository as private. Default: on",
    )
    parser.add_argument(
        "--public",
        dest="private_repo",
        action="store_false",
        help="Create a new repository as public.",
    )
    parser.add_argument(
        "--create-if-missing",
        dest="create_if_missing",
        action="store_true",
        default=True,
        help="Create the repository automatically if needed. Default: on",
    )
    parser.add_argument(
        "--no-create-if-missing",
        dest="create_if_missing",
        action="store_false",
        help="Fail if the repository does not yet exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would happen.",
    )
    return parser.parse_args()


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "ja", "on"}


def repo_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    slug = slug.strip("-")
    return slug or "ki-serien-training"


def allowed_repo_files() -> list[str]:
    files = sorted(path.name for path in SCRIPT_DIR.glob("*.py"))
    readme = SCRIPT_DIR / "README.md"
    if readme.exists():
        files.append(readme.name)
    return files


def tracked_repo_files() -> list[str]:
    result = run_git(["ls-files", "-z"], check=False)
    if result.returncode != 0:
        return []
    return [item for item in (result.stdout or "").split("\0") if item]


def path_has_changes(relative_path: str) -> bool:
    result = run_git(["status", "--porcelain", "--", relative_path], check=False)
    return bool((result.stdout or "").strip())


def ensure_readme_changed_with_scripts(files: list[str], dry_run: bool) -> None:
    if dry_run and not (SCRIPT_DIR / ".git").exists():
        return
    changed_scripts = [path for path in files if path.endswith(".py") and path_has_changes(path)]
    readme_changed = "README.md" in files and path_has_changes("README.md")
    if changed_scripts and not readme_changed:
        names = ", ".join(changed_scripts)
        raise RuntimeError(
            "README.md must be updated together with script changes. "
            f"Affected: {names}"
        )


def prune_disallowed_tracked_files(allowed_files: list[str], dry_run: bool) -> bool:
    allowed = set(allowed_files)
    disallowed = [path for path in tracked_repo_files() if path not in allowed]
    if not disallowed:
        return False
    preview = ", ".join(disallowed[:5])
    suffix = "" if len(disallowed) <= 5 else ", ..."
    if dry_run:
        warn(f"Would remove non-script files from the repo index: {preview}{suffix}")
        return True
    info("Removing already versioned non-script files from the Git index.")
    run_git(["rm", "--cached", "--ignore-unmatch", "--", *disallowed])
    return True


def ensure_git_repo(branch: str, dry_run: bool) -> None:
    if (SCRIPT_DIR / ".git").exists():
        return
    if dry_run:
        info(f"Would initialize a local Git repository with branch '{branch}'.")
        return
    info(f"Initializing local Git repository with branch '{branch}'.")
    run_git(["init", "-b", branch])


def current_remote_url() -> str:
    result = run_git(["remote", "get-url", "origin"], check=False)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def head_commit_exists() -> bool:
    result = run_git(["rev-parse", "--verify", "HEAD"], check=False)
    return result.returncode == 0


def parse_remote_url(url: str) -> tuple[str, str] | None:
    for pattern in REMOTE_PATTERNS:
        match = pattern.match(url.strip())
        if match:
            return match.group("owner"), match.group("repo")
    return None


def ensure_git_identity(name: str, email: str, dry_run: bool) -> None:
    current_name = git_output(["config", "--get", "user.name"])
    current_email = git_output(["config", "--get", "user.email"])
    final_name = current_name or name
    final_email = current_email or email
    if not final_name or not final_email:
        raise RuntimeError(
            "Git identity is missing. Please set --git-user-name and --git-user-email "
            "or configure git globally."
        )
    if not current_name:
        if dry_run:
            info(f"Would set local Git user.name to: {final_name}")
        else:
            run_git(["config", "user.name", final_name])
    if not current_email:
        if dry_run:
            info(f"Would set local Git user.email to: {final_email}")
        else:
            run_git(["config", "user.email", final_email])


def github_request(method: str, path: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"https://api.github.com{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "ki-serien-training-sync",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urlrequest.Request(url, data=data, headers=headers, method=method)
    try:
        with urlrequest.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
            return json.loads(body) if body else {}
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body
        try:
            payload = json.loads(body) if body else {}
            detail = str(payload.get("message") or body)
        except Exception:
            detail = body or str(exc)
        raise GitHubApiError(exc.code, f"GitHub API error {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"GitHub API is not reachable: {exc}") from exc


def github_request_optional(method: str, path: str, token: str) -> dict[str, Any] | None:
    try:
        return github_request(method, path, token)
    except GitHubApiError as exc:
        if exc.status == 404:
            return None
        raise


def github_login(token: str) -> str:
    data = github_request("GET", "/user", token)
    login = str(data.get("login") or "").strip()
    if not login:
        raise RuntimeError("Could not determine the GitHub login from the token.")
    return login


def repo_exists(owner: str, repo: str, token: str) -> bool:
    try:
        github_request("GET", f"/repos/{owner}/{repo}", token)
        return True
    except GitHubApiError as exc:
        if exc.status == 404:
            return False
        raise


def create_repo(owner: str, repo: str, private_repo: bool, token: str) -> None:
    login = github_login(token)
    payload = {
        "name": repo,
        "private": private_repo,
        "auto_init": False,
    }
    if owner == login:
        github_request("POST", "/user/repos", token, payload)
        ok(f"GitHub repository {owner}/{repo} was created.")
        return
    github_request("POST", f"/orgs/{owner}/repos", token, payload)
    ok(f"GitHub repository {owner}/{repo} was created in the organization.")


def update_repo_about(owner: str, repo: str, token: str, dry_run: bool) -> None:
    if dry_run:
        info(f"Would set GitHub About to: {REPO_ABOUT_DESCRIPTION}")
        return
    github_request(
        "PATCH",
        f"/repos/{owner}/{repo}",
        token,
        {"description": REPO_ABOUT_DESCRIPTION},
    )
    info("GitHub About was updated.")


def github_branch_head(owner: str, repo: str, branch: str, token: str) -> dict[str, Any] | None:
    return github_request_optional("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}", token)


def github_commit(owner: str, repo: str, commit_sha: str, token: str) -> dict[str, Any]:
    return github_request("GET", f"/repos/{owner}/{repo}/git/commits/{commit_sha}", token)


def github_tree(owner: str, repo: str, tree_sha: str, token: str) -> dict[str, Any]:
    return github_request("GET", f"/repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1", token)


def github_create_blob(owner: str, repo: str, relative_path: str, token: str) -> str:
    file_path = SCRIPT_DIR / relative_path
    content = file_path.read_text(encoding="utf-8")
    data = github_request(
        "POST",
        f"/repos/{owner}/{repo}/git/blobs",
        token,
        {"content": content, "encoding": "utf-8"},
    )
    sha = str(data.get("sha") or "").strip()
    if not sha:
        raise RuntimeError(f"Blob creation failed for {relative_path}.")
    return sha


def github_create_tree(
    owner: str,
    repo: str,
    base_tree_sha: str | None,
    tree_entries: list[dict[str, Any]],
    token: str,
) -> str:
    payload: dict[str, Any] = {"tree": tree_entries}
    if base_tree_sha:
        payload["base_tree"] = base_tree_sha
    data = github_request("POST", f"/repos/{owner}/{repo}/git/trees", token, payload)
    sha = str(data.get("sha") or "").strip()
    if not sha:
        raise RuntimeError("Could not create the GitHub tree.")
    return sha


def github_create_commit(
    owner: str,
    repo: str,
    message: str,
    tree_sha: str,
    parent_commit_sha: str | None,
    token: str,
) -> str:
    payload: dict[str, Any] = {"message": message, "tree": tree_sha}
    if parent_commit_sha:
        payload["parents"] = [parent_commit_sha]
    data = github_request("POST", f"/repos/{owner}/{repo}/git/commits", token, payload)
    sha = str(data.get("sha") or "").strip()
    if not sha:
        raise RuntimeError("Could not create the GitHub commit.")
    return sha


def github_update_branch(owner: str, repo: str, branch: str, commit_sha: str, token: str) -> None:
    existing = github_branch_head(owner, repo, branch, token)
    if existing is None:
        github_request(
            "POST",
            f"/repos/{owner}/{repo}/git/refs",
            token,
            {"ref": f"refs/heads/{branch}", "sha": commit_sha},
        )
        return
    github_request(
        "PATCH",
        f"/repos/{owner}/{repo}/git/refs/heads/{branch}",
        token,
        {"sha": commit_sha, "force": True},
    )


def build_github_mirror_tree(
    owner: str,
    repo: str,
    branch: str,
    files: list[str],
    token: str,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    head = github_branch_head(owner, repo, branch, token)
    if head is None:
        entries = []
        for relative_path in files:
            entries.append(
                {
                    "path": relative_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": github_create_blob(owner, repo, relative_path, token),
                }
            )
        return None, None, entries

    commit_sha = str((((head.get("object") or {})).get("sha")) or "").strip()
    if not commit_sha:
        raise RuntimeError(f"GitHub branch {branch} does not have a readable commit SHA.")
    commit_data = github_commit(owner, repo, commit_sha, token)
    base_tree_sha = str((((commit_data.get("tree") or {})).get("sha")) or "").strip()
    if not base_tree_sha:
        raise RuntimeError(f"GitHub commit {commit_sha} does not have a readable tree SHA.")
    remote_tree = github_tree(owner, repo, base_tree_sha, token)
    remote_paths = {
        str(item.get("path") or "")
        for item in (remote_tree.get("tree") or [])
        if item.get("type") == "blob" and item.get("path")
    }

    entries: list[dict[str, Any]] = []
    for relative_path in files:
        entries.append(
            {
                "path": relative_path,
                "mode": "100644",
                "type": "blob",
                "sha": github_create_blob(owner, repo, relative_path, token),
            }
        )
    for remote_path in sorted(remote_paths - set(files)):
        entries.append(
            {
                "path": remote_path,
                "mode": "100644",
                "type": "blob",
                "sha": None,
            }
        )
    return commit_sha, base_tree_sha, entries


def ensure_origin(owner: str, repo: str, dry_run: bool) -> str:
    clean_url = f"https://github.com/{owner}/{repo}.git"
    remote_url = current_remote_url()
    if remote_url == clean_url:
        return clean_url
    if dry_run:
        action = "set" if not remote_url else f"change to {clean_url}"
        info(f"Would {action} Git remote 'origin'.")
        return clean_url
    if remote_url:
        run_git(["remote", "set-url", "origin", clean_url])
    else:
        run_git(["remote", "add", "origin", clean_url])
    return clean_url


def stage_and_commit(files: list[str], message: str, dry_run: bool) -> bool:
    pruned = prune_disallowed_tracked_files(files, dry_run)
    if dry_run:
        info(f"Would version only these files: {', '.join(files)}")
        info("Would stage the allowed files and commit if there are changes.")
        return True
    run_git(["add", "--all", "--", *files])
    staged = run_git(["diff", "--cached", "--quiet"], check=False)
    if staged.returncode == 0:
        info("No new Git changes found to commit.")
        return False
    info("Creating Git commit.")
    run_git(["commit", "-m", message])
    return True


def push_changes(owner: str, repo: str, branch: str, token: str, dry_run: bool) -> None:
    if dry_run:
        info(f"Would mirror the current files to {owner}/{repo}:{branch} via the GitHub API.")
        return
    files = allowed_repo_files()
    if not git_output(["rev-parse", "--verify", "HEAD"]):
        raise RuntimeError("No local commit found for mirroring.")
    parent_commit_sha, base_tree_sha, tree_entries = build_github_mirror_tree(owner, repo, branch, files, token)
    tree_sha = github_create_tree(owner, repo, base_tree_sha, tree_entries, token)
    message = git_output(["log", "-1", "--pretty=%s"]) or "Automatic script update"
    commit_sha = github_create_commit(owner, repo, message, tree_sha, parent_commit_sha, token)
    github_update_branch(owner, repo, branch, commit_sha, token)
    ok(f"GitHub was updated: {owner}/{repo}:{branch}")


def resolve_owner_repo(args: argparse.Namespace) -> tuple[str, str]:
    owner = str(args.owner or DEFAULT_OWNER).strip()
    repo = str(args.repo or DEFAULT_REPO).strip()
    return owner, repo


def main() -> None:
    args = parse_args()
    headline("GitHub Synchronization")
    info("Mode: upload/mirror from local to GitHub only, never download.")
    files = allowed_repo_files()
    ensure_git_repo(args.branch, args.dry_run)
    ensure_git_identity(args.git_user_name.strip(), args.git_user_email.strip(), args.dry_run)
    ensure_readme_changed_with_scripts(files, args.dry_run)
    owner, repo = resolve_owner_repo(args)
    target_remote_url = f"https://github.com/{owner}/{repo}.git"
    remote_url = current_remote_url()
    missing_remote_without_auth = not remote_url and not args.token
    info(f"Target: {owner}/{repo}:{args.branch}")
    info(f"Remote: {target_remote_url}")

    if remote_url:
        remote_parts = parse_remote_url(remote_url)
        if remote_parts and remote_parts != (owner, repo):
            warn(
                f"Existing origin remote points to {remote_parts[0]}/{remote_parts[1]}. "
                f"It will be changed to {owner}/{repo}."
            )

    if not remote_url:
        if not args.token:
            if args.dry_run:
                warn(
                    "Without GITHUB_TOKEN or an existing origin remote, a real GitHub push would not be possible yet."
                )
            else:
                raise RuntimeError(
                    "No origin remote was found and no token is available. "
                    "Please set GITHUB_TOKEN or configure a remote first."
                )
        else:
            exists = repo_exists(owner, repo, args.token)
            if not exists:
                if not args.create_if_missing:
                    raise RuntimeError(f"GitHub repository {owner}/{repo} does not exist yet.")
                if args.dry_run:
                    info(
                        f"Would create GitHub repository {owner}/{repo} "
                        f"({'private' if args.private_repo else 'public'})."
                    )
                else:
                    create_repo(owner, repo, args.private_repo, args.token)
    elif args.token and not repo_exists(owner, repo, args.token):
        if not args.create_if_missing:
            raise RuntimeError(f"GitHub repository {owner}/{repo} does not exist yet.")
        if args.dry_run:
            info(
                f"Would create missing GitHub repository {owner}/{repo} "
                f"({'private' if args.private_repo else 'public'})."
            )
        else:
            create_repo(owner, repo, args.private_repo, args.token)

    if args.token:
        update_repo_about(owner, repo, args.token, args.dry_run)

    ensure_origin(owner, repo, args.dry_run)
    has_commit = stage_and_commit(files, args.message, args.dry_run)
    if not has_commit:
        if not head_commit_exists():
            ok("There are no new changes for GitHub.")
            return
        info("No new allowed file changes were found. Checking whether an existing local commit still needs to be mirrored.")

    if missing_remote_without_auth:
        warn("For a real run, set GITHUB_TOKEN or configure an origin remote first.")
        return

    if args.token:
        push_changes(owner, repo, args.branch, args.token, args.dry_run)
        return

    if args.dry_run:
        info("Would mirror via force-push using existing Git credentials without an embedded token.")
        return

    info("Mirroring local state via force-push using configured Git credentials.")
    result = run_git(["push", "--force", "origin", f"HEAD:refs/heads/{args.branch}"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Git push via origin failed. Please check credentials or the remote.")
    ok(f"GitHub was updated: {owner}/{repo}:{args.branch}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        sys.exit(1)

