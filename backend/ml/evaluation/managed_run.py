"""Start an isolated loopback server, evaluate it, and stop only that server.

    python -m ml.evaluation.managed_run --backend trained

No existing database, server, credentials, or results are reused or overwritten.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["trained", "rule_based", "bert"], default="trained")
    parser.add_argument("--bert-model-path", type=Path, help="Explicit completed BERT export")
    parser.add_argument("--input-type", choices=["url", "email_text"], default="url")
    parser.add_argument("--limit", type=int, help="Optional smoke-test size; omit for full evaluation")
    args = parser.parse_args()
    if args.backend == "bert" and args.input_type != "email_text":
        parser.error("BERT supports --input-type email_text only")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.input_type + "_" + args.backend + "_" + secrets.token_hex(3)
    run_dir = ROOT / "ml/evaluation/runs" / run_id
    run_dir.mkdir(parents=True)
    db_path = run_dir / "evaluation.db"
    env = os.environ.copy()
    # Isolate the selected modality; never inherit the other component's backend.
    env.update(DATABASE_URL="sqlite:///" + db_path.as_posix(),
               URL_CLASSIFIER_BACKEND=args.backend if args.input_type == "url" else "rule_based",
               EMAIL_CLASSIFIER_BACKEND=args.backend if args.input_type == "email_text" else "rule_based",
               SECRET_KEY=secrets.token_urlsafe(48), EVALUATION_PASSWORD=secrets.token_urlsafe(24),
               PYTHONUNBUFFERED="1", ENVIRONMENT="development")
    if args.bert_model_path:
        env["EMAIL_BERT_MODEL_PATH"] = str(args.bert_model_path.resolve())
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.run([sys.executable, "-c", "import os; from scripts.create_admin import create_admin; "
                    "create_admin('evaluation@example.com', 'Evaluation', os.environ['EVALUATION_PASSWORD'])"],
                   cwd=ROOT, env=env, check=True, creationflags=flags)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    with (run_dir / "server.log").open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                   "--port", str(port), "--no-access-log"], cwd=ROOT, env=env,
                                  stdout=server_log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            deadline = time.monotonic() + 180
            with httpx.Client(trust_env=False, timeout=1) as client:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Evaluation server exited; inspect {run_dir / 'server.log'}")
                    try:
                        if client.get(base_url + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("Evaluation server did not become ready within 180 seconds")
                    time.sleep(.25)
            command = [sys.executable, "-m", "ml.evaluation.run_evaluation", "--base-url", base_url,
                       "--admin-email", "evaluation@example.com", "--classifier-backend", args.backend,
                       "--input-type", args.input_type,
                       "--db-path", str(db_path), "--output", str(run_dir / "results.json")]
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            with (run_dir / "evaluation.log").open("w", encoding="utf-8") as evaluation_log:
                evaluator = subprocess.Popen(command, cwd=ROOT, env=env, creationflags=flags,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace")
                try:
                    for line in evaluator.stdout:
                        evaluation_log.write(line)
                        evaluation_log.flush()
                        print(line, end="", flush=True)
                    if evaluator.wait() != 0:
                        raise RuntimeError(f"Evaluation failed; inspect {run_dir / 'evaluation.log'}")
                finally:
                    if evaluator.poll() is None:
                        evaluator.terminate()
                        try:
                            evaluator.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            evaluator.kill()
                            evaluator.wait()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print(f"Evaluation server stopped. Evidence: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
