"""Download licensed, pinned artifacts; prepare an offline runtime without changing weights."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

GR00T_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory")
    groot = Path(
        snapshot_download(
            "nvidia/GR00T-N1.7-LIBERO",
            revision=GR00T_REVISION,
            allow_patterns=["libero_spatial/*.json", "libero_spatial/*.safetensors"],
            ignore_patterns=["libero_spatial/experiment_cfg/*"],
        )
    )
    cosmos = Path(snapshot_download("nvidia/Cosmos-Reason2-2B", revision=COSMOS_REVISION))
    output.mkdir(parents=True)
    target = output / "gr00t-n1.7-libero" / "libero_spatial"
    shutil.copytree(groot / "libero_spatial", target)
    shutil.copytree(cosmos, output / "Cosmos-Reason2-2B")
    # The container mount is stable; original safetensors bytes remain unchanged.
    config = target / "config.json"
    data = json.loads(config.read_text())
    data["model_name"] = "/weights/Cosmos-Reason2-2B"
    config.write_text(json.dumps(data, indent=2) + "\n")
    processor = target / "processor_config.json"
    data = json.loads(processor.read_text())
    data["processor_kwargs"]["model_name"] = "/weights/Cosmos-Reason2-2B"
    processor.write_text(json.dumps(data, indent=2) + "\n")
    from moravec_policy.provenance import tree_hash

    print(
        json.dumps(
            {
                "gr00t_prepared_sha256": tree_hash(target),
                "cosmos_sha256": tree_hash(output / "Cosmos-Reason2-2B"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
