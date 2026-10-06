import json
import subprocess
import sys


def verify(base: dict, candidate: dict) -> None:
    old_layers = base["RootFS"]["Layers"]
    new_layers = candidate["RootFS"]["Layers"]
    if new_layers[:-1] != old_layers:
        raise ValueError(
            "Candidate must preserve every base layer and add exactly one layer"
        )
    for key in ("Entrypoint", "Cmd", "WorkingDir", "User"):
        if base["Config"][key] != candidate["Config"][key]:
            raise ValueError(f"Runtime configuration changed: {key}")
    if candidate["Os"] != "linux" or candidate["Architecture"] != "amd64":
        raise ValueError("Expected a linux/amd64 image")
    cache_values = [
        entry
        for entry in candidate["Config"]["Env"]
        if entry.startswith("AMD_GPU_GET_CACHE_TTL=")
    ]
    if cache_values != ["AMD_GPU_GET_CACHE_TTL=0s"]:
        raise ValueError("Exporter cache bypass must be explicit in the image")


if __name__ == "__main__":
    images = json.loads(
        subprocess.check_output(["docker", "image", "inspect", *sys.argv[1:]])
    )
    if len(images) != 2:
        raise ValueError("Expected base and candidate image references")
    verify(*images)
    print("Base layers, runtime entrypoint and cache bypass verified")
