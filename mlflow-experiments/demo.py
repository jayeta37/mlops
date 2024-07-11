import os
import mlflow
import argparse
import time

def eval(p1, p2):
    output_metric = p1**2 + p2**2
    return output_metric

def main(input1, input2):
    mlflow.set_experiment("Demo Experiment")
    with mlflow.start_run(run_name="Example Demo"):
        mlflow.set_tag("version", "1.0.0")
        mlflow.log_param("param1", input1)
        mlflow.log_param("param2", input2)
        metric = eval(input1, input2)
        mlflow.log_metric("Eval Metric", metric)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(f"Artifact created at {time.asctime()}")
        mlflow.log_artifacts("dummy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param1", "-p1", type=int, default=5)
    parser.add_argument("--param2", "-p2", type=int, default=10)
    parsed_args = parser.parse_args()
    main(input1=parsed_args.param1, input2=parsed_args.param2)