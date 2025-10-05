import sys


def main() -> int:
    """Basic environment sanity checks."""
    print("Python version:", sys.version)
    # Add more checks as needed (e.g., import ultralytics)
    try:
        import ultralytics  # noqa: F401
        print("ultralytics: OK")
    except Exception as e:
        print("ultralytics import failed:", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
