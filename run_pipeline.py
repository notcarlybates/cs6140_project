"""
Local pipeline runner for the HAR Random Forest pipeline.
Runs each step locally; /scratch paths are transparently accessed via SSH by scratch_io.

Usage:
    python run_pipeline.py             # run all steps in order
    python run_pipeline.py preprocess  # run a single step
    python run_pipeline.py features
    python run_pipeline.py train
"""

import subprocess
import sys
import os

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    "preprocess": f"{LOCAL_DIR}/rf_1_preprocess.py",
    "features":   f"{LOCAL_DIR}/rf_2_features.py",
    "train":      f"{LOCAL_DIR}/rf_3_train.py",
}


def run_step(step_name: str, script_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"[{step_name}] {script_path}")
    print(f"{'='*60}")
    subprocess.run([sys.executable, script_path], check=True)


def main():
    steps_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(STEPS.keys())

    for step in steps_to_run:
        if step not in STEPS:
            print(f"Unknown step '{step}'. Valid steps: {', '.join(STEPS)}")
            sys.exit(1)
        run_step(step, STEPS[step])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
