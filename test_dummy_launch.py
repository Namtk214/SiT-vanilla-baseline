import sys
import subprocess
try:
    cmd = [
        "python", "train.py",
        "--online-encode", "./test_dummy_dir",
        "--batch-size", "8",
        "--online-batch-size", "8",
        "--epochs", "1",
        "--steps-per-epoch", "1",
        "--learning-rate", "1e-4",
        "--ckpt-dir", "./checkpoints",
        "--wandb-project", "selfflow-tpu",
        "--model", "DiT-B/2",
        "--fid-freq", "20000",
        "--num-fid-samples", "4000",
        "--log-freq", "100",
        "--sample-freq", "5000"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env={"JAX_PLATFORMS": "cpu", "WANDB_MODE": "offline", **os.environ})
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    print("Return Code:", result.returncode)
except Exception as e:
    print(e)
