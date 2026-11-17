import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request


REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "langchain_core": "langchain-core",
    "langchain_ollama": "langchain-ollama",
}


def check_package(import_name: str, package_name: str) -> bool:
    if importlib.util.find_spec(import_name):
        print(f"[OK] {package_name}")
        return True

    print(f"[ERROR] {package_name} is not installed")
    return False


def check_ollama() -> bool:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags",
            timeout=5,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        models = [
            model.get("name", "")
            for model in payload.get("models", [])
        ]

        print("[OK] Ollama is running")

        if any(name.startswith("mistral") for name in models):
            print("[OK] Mistral is available")
            return True

        print("[ERROR] Mistral is not installed")
        print("        Run: ollama pull mistral")
        return False

    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        print("[ERROR] Ollama is not responding")
        print("        Start Ollama and run: ollama pull mistral")
        return False


def main() -> None:
    print("\nAI Lead Qualification Workshop")
    print("=" * 38)
    print(f"[OK] Python {sys.version.split()[0]}")

    results = [
        check_package(import_name, package_name)
        for import_name, package_name in REQUIRED_PACKAGES.items()
    ]

    results.append(check_ollama())

    print("=" * 38)

    if all(results):
        print("[SUCCESS] The workshop environment is ready.")
        print("Run: streamlit run app.py")
    else:
        print("[ACTION REQUIRED] Correct the errors listed above.")
        sys.exit(1)


if __name__ == "__main__":
    main()