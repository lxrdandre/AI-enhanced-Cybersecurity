import os


def detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)

    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, "..", ".."))


def resolve_data_csv(project_root: str, relative_path: str) -> str:
    return os.path.normpath(os.path.join(project_root, relative_path))


def resolve_artifacts_dir(project_root: str, subdir: str) -> str:
    return os.path.normpath(os.path.join(project_root, "artifacts", subdir))
