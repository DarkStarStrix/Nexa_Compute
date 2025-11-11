"""High-level helpers for building and publishing curated container images."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from nexa_infra.containers import ContainerSpec, available_containers, get_container


def build_image(
    spec: ContainerSpec,
    *,
    tag: str | None = None,
    build_args: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Build a container image for the provided specification and tag.

    Args:
        spec: Container specification describing Dockerfile and repository.
        tag: Optional tag to apply; defaults to the spec's default tag.
        build_args: Optional build arguments forwarded to the Docker build command.
        labels: Optional labels added to the image metadata.

    Returns:
        The image reference that was built.
    """

    image_ref = spec.ref(tag)
    cmd = [
        "docker",
        "build",
        "-f",
        str(spec.dockerfile),
        "-t",
        image_ref,
        str(spec.context),
    ]
    for key, value in (build_args or {}).items():
        cmd.extend(["--build-arg", f"{key}={value}"])
    for key, value in (labels or {}).items():
        cmd.extend(["--label", f"{key}={value}"])

    subprocess.run(cmd, check=True)
    return image_ref


def push_image(image_ref: str) -> None:
    """Push an image reference to the configured container registry."""

    subprocess.run(["docker", "push", image_ref], check=True)


def tag_image(spec: ContainerSpec, source_tag: str, target_tag: str, *, push: bool = False) -> str:
    """Create an additional tag for an existing container image."""

    source_ref = spec.ref(source_tag)
    target_ref = spec.ref(target_tag)
    subprocess.run(["docker", "tag", source_ref, target_ref], check=True)
    if push:
        push_image(target_ref)
    return target_ref


def build_release(
    targets: Sequence[str] | None = None,
    *,
    variant_tag: str = "cu121-py311-pt22",
    include_latest: bool = True,
    include_date: bool = True,
    push: bool = False,
    additional_tags: Iterable[str] | None = None,
    build_args: Mapping[str, str] | None = None,
) -> None:
    """Build (and optionally push) curated images for the selected targets.

    The release pipeline always builds the variant tag first, then applies
    canonical aliases such as ``latest`` and a UTC datestamp.
    """

    specs = (
        [get_container(target) for target in targets]
        if targets
        else list(available_containers())
    )

    extra_tags = list(additional_tags or [])
    if include_date:
        extra_tags.append(datetime.now(timezone.utc).strftime("%Y%m%d"))

    for spec in specs:
        built_ref = build_image(spec, tag=variant_tag, build_args=build_args)
        if push:
            push_image(built_ref)

        aliases = []
        if include_latest:
            aliases.append("latest")
        aliases.extend(extra_tags)

        for alias in aliases:
            tag_image(spec, variant_tag, alias, push=push)


