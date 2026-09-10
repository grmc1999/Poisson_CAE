"""Quick SLURM status + log tail for Poisson_CAE sweeps on SDumont / ICA.

Reads credentials from environment variables (never commit secrets):
  SD_HOST, SD_USER, SD_PASS  (SDumont, defaults host/user from skill docs)
  ICA_HOST, ICA_USER, ICA_PASS

Usage
-----
    python cluster/status.py --user guillermo.carrillo          # queue only
    python cluster/status.py --user guillermo.carrillo --tail 15  # + tail logs
    python cluster/status.py --hosts sdumont --tail 15
    python cluster/status.py --logs-dir cluster/logs/mnist_diffusion --tail 20
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:  # pragma: no cover
    print("paramiko is required: pip install paramiko", file=sys.stderr)
    sys.exit(1)


CLUSTERS = {
    "sdumont": {
        "host": os.environ.get("SD_HOST", "146.134.176.5"),
        "user_env": "SD_USER",
        "pass_env": "SD_PASS",
        "code": "/petrobr/parceirosbr/proxy-sim/users/guillermo.carrillo/Proxy-FNO",
    },
    "ica": {
        "host": os.environ.get("ICA_HOST", "139.82.152.10"),
        "port": 22,
        "user_env": "ICA_USER",
        "pass_env": "ICA_PASS",
        "code": "/share_zeta/Proxy-Sim/guillermo.carrillo/Proxy-FNO",
    },
}


def ssh_run(host: str, username: str, password: str, command: str, port: int = 22):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=username, password=password,
                   timeout=30)
    _, stdout, stderr = client.exec_command(command, timeout=60)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    client.close()
    return out, err


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Check SLURM queues + tail logs")
    ap.add_argument("--hosts", nargs="*", default=["sdumont", "ica"])
    ap.add_argument("--user", default=None, help="SLURM user for squeue")
    ap.add_argument("--logs-dir", default=None,
                    help="local clone of cluster/logs/<sweep> to tail")
    ap.add_argument("--tail", type=int, default=0,
                    help="tail N lines of the latest .out logs")
    args = ap.parse_args(argv)

    for name in args.hosts:
        info = CLUSTERS[name]
        username = args.user or os.environ.get(info["user_env"], "")
        password = os.environ.get(info["pass_env"], "")
        if not username or not password:
            print(f"[{name}] set {info['user_env']}/{info['pass_env']} env vars "
                  f"(or --user) to check the queue")
            continue
        host = info.get("host", info.get("addr", ""))
        port = info.get("port", 22)
        print(f"=== {name} ({host}) queue for {username} ===")
        queue_cmd = (
            f"squeue -u {username} "
            f'--format="%.18i %.9P %.30j %.2t %.10M %R"'
        )
        try:
            out, err = ssh_run(host, username, password, queue_cmd, port)
            print(out if out.strip() else "(queue empty)")
            if err.strip():
                print("[stderr]", err)
        except Exception as e:
            print(f"[{name}] SSH/queue failed: {e}")

    if args.logs_dir and args.tail > 0:
        logs = sorted(Path(args.logs_dir).glob("*.out"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"\n=== latest logs in {args.logs_dir} ===")
        for log in logs[:3]:
            print(f"--- {log.name} (last {args.tail} lines) ---")
            lines = log.read_text(errors="replace").splitlines()
            print("\n".join(lines[-args.tail:]))


if __name__ == "__main__":
    main()