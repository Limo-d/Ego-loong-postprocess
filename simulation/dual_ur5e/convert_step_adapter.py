#!/usr/bin/env python3
"""Convert a STEP flange adapter into a MuJoCo-compatible binary STL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gmsh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--target_facets_across_diagonal",
        type=float,
        default=80.0,
        help="Approximate surface resolution; larger produces a denser STL",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("ur5e_omnipicker_adapter")
        entities = gmsh.model.occ.importShapes(str(args.input.resolve()), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        bounds = gmsh.model.getBoundingBox(-1, -1)
        extent = [bounds[index + 3] - bounds[index] for index in range(3)]
        diagonal = sum(value * value for value in extent) ** 0.5
        mesh_size = diagonal / max(args.target_facets_across_diagonal, 1.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.35)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(args.output.resolve()))
        print(
            json.dumps(
                {
                    "input": str(args.input.resolve()),
                    "output": str(args.output.resolve()),
                    "volume_entities": len(entities),
                    "bounds_native": list(bounds),
                    "extent_native": extent,
                    "mesh_size_native": mesh_size,
                    "nodes": len(gmsh.model.mesh.getNodes()[0]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
