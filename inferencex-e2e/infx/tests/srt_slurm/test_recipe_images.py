"""Keep checked-in srt-slurm recipe images identical to their master-config images."""

import sys
from collections import defaultdict
from pathlib import Path

import yaml

from infx.srt_slurm.synthetic_acceptance import selected_recipes

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
MULTI_NODE_RECIPES = "benchmarks/multi_node/srt-slurm-recipes/"
# Their recipes still name a stale image on main; remove a key once its recipe is aligned.
KNOWN_STALE_KEYS = {
    "glm5.2-fp8-mi325x-sglang-agentic-mtp",
    "minimaxm3-fp8-mi300x-vllm-agentic-mtp",
}


def recipe_references(node):
    """Yield (multinode, reference) for every srt-slurm recipe a master entry names."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "srt-recipe":
                yield False, value
            else:
                yield from recipe_references(value)
    elif isinstance(node, list):
        for item in node:
            yield from recipe_references(item)
    elif isinstance(node, str) and node.startswith("CONFIG_FILE="):
        path = node.removeprefix("CONFIG_FILE=")
        # Launchers stage multi-node recipes under recipes/ in the srt-slurm checkout.
        for staged in ("recipes/", MULTI_NODE_RECIPES):
            if path.startswith(staged):
                yield True, MULTI_NODE_RECIPES + path.removeprefix(staged)


def master_references():
    """Yield (config key, image, multinode, reference) from every master config."""
    for master in sorted((ROOT / "configs").glob("*-master.yaml")):
        for key, entry in yaml.safe_load(master.read_text()).items():
            for multinode, reference in sorted(set(recipe_references(entry))):
                yield key, entry["image"], multinode, reference


def variants(reference):
    path, _, selector = reference.partition(":")
    recipe = yaml.safe_load((ROOT / path).read_text())
    return path, selected_recipes(recipe, selector or None)


def canonical_image(image):
    # Enroot spells the nvcr.io registry separator as "#"; launchers key aliases either way.
    return image.replace("nvcr.io#", "nvcr.io/", 1)


def test_single_node_recipes_use_their_master_image():
    references = [
        (key, image, reference)
        for key, image, multinode, reference in master_references()
        if not multinode and key not in KNOWN_STALE_KEYS
    ]
    images = defaultdict(set)
    for _, image, reference in references:
        images[reference.partition(":")[0]].add(image)
    problems = set()
    for key, image, reference in references:
        path, selected = variants(reference)
        containers = {recipe["model"]["container"] for _, recipe in selected}
        if image not in containers:
            problems.add(f"{key}: no variant of {reference} uses master image {image}")
        for container in containers - images[path]:
            problems.add(f"{path}: {container} is not the image of any master key using it")
    assert not problems, "\n".join(sorted(problems))


def test_multi_node_recipes_use_their_master_image():
    problems = set()
    for key, image, multinode, reference in master_references():
        if not multinode:
            continue
        path, selected = variants(reference)
        for name, recipe in selected:
            label = f"{key}: {path}" + (f":{name}" if name else "")
            container = recipe["model"]["container"]
            # srtctl pulls a literal missing from the alias map, so a stale one runs silently.
            if ":" in container and canonical_image(container) != canonical_image(image):
                problems.add(f"{label} model.container {container} != master image {image}")
            identity = ((recipe.get("identity") or {}).get("container") or {}).get("image")
            if identity is not None and canonical_image(identity) != canonical_image(image):
                problems.add(f"{label} identity.container.image {identity} != master image {image}")
    assert not problems, "\n".join(sorted(problems))
