"""
End-to-end runner: fetch -> features -> models -> Excel report -> charts.
"""
from fetch_data import fetch_panel
from build_features import build_features
from run_models import run_all
from plot_results import main as plot_main


def main():
    print("=" * 70)
    print("STEP 1/4  Fetch raw panel from FRED + EIA")
    print("=" * 70)
    fetch_panel()

    print()
    print("=" * 70)
    print("STEP 2/4  Build feature matrix")
    print("=" * 70)
    build_features()

    print()
    print("=" * 70)
    print("STEP 3/4  Fit four OLS models")
    print("=" * 70)
    run_all()

    print()
    print("=" * 70)
    print("STEP 4/4  Generate evaluation charts")
    print("=" * 70)
    plot_main()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
