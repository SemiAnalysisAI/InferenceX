#!/usr/bin/env bash

# Enroot 3.x does not parse Docker's tag@digest syntax. For digest-pinned
# images, use its explicit registry syntax and pass the digest as the
# manifest reference so the import remains immutable.
enroot_uri_for_image() {
    local image="$1"
    local image_without_digest="$image"
    local digest=""
    local first_component registry repository repository_dir repository_name

    if [[ "$image" == *@sha256:* ]]; then
        image_without_digest="${image%@*}"
        digest="${image##*@}"
    fi

    first_component="${image_without_digest%%/*}"
    if [[ "$image_without_digest" == */* && ( "$first_component" == *.* || "$first_component" == *:* || "$first_component" == "localhost" ) ]]; then
        registry="$first_component"
        repository="${image_without_digest#*/}"
    else
        registry="registry-1.docker.io"
        repository="$image_without_digest"
    fi

    if [[ -z "$digest" ]]; then
        if [[ "$registry" == "registry-1.docker.io" ]]; then
            printf 'docker://%s\n' "$image"
        else
            printf 'docker://%s#%s\n' "$registry" "$repository"
        fi
        return
    fi

    repository_dir="${repository%/*}"
    repository_name="${repository##*/}"
    repository_name="${repository_name%%:*}"
    if [[ "$repository" == */* ]]; then
        repository="${repository_dir}/${repository_name}"
    else
        repository="$repository_name"
    fi
    if [[ "$registry" == "registry-1.docker.io" && "$repository" != */* ]]; then
        repository="library/$repository"
    fi

    printf 'docker://%s#%s:%s\n' "$registry" "$repository" "$digest"
}
