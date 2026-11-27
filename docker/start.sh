#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ref https://github.com/runpod/containers/blob/main/container-template/start.sh

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #


# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        local key
        install -d -m 0700 ~/.ssh
        touch ~/.ssh/authorized_keys
        chmod 0600 ~/.ssh/authorized_keys
        while IFS= read -r key; do
            if [[ -n $key ]] && ! grep -Fxq -- "$key" ~/.ssh/authorized_keys; then
                printf '%s\n' "$key" >> ~/.ssh/authorized_keys
            fi
        done <<< "$PUBLIC_KEY"

        # The image carries no host private keys. Create this pod's identity on first boot.
        ssh-keygen -A
        service ssh start
    fi
}

# SSH sessions do not inherit the pod environment. Keep only the connection
# and resource information needed there. Other RUNPOD_* variables may be keys.
write_ssh_env() {
    local name
    for name in PATH RUNPOD_POD_ID RUNPOD_POD_HOSTNAME RUNPOD_PUBLIC_IP RUNPOD_DC_ID \
        RUNPOD_VOLUME_ID RUNPOD_CPU_COUNT RUNPOD_MEM_GB RUNPOD_GPU_COUNT; do
        if [[ -v $name ]]; then
            printf 'export %s=%q\n' "$name" "${!name}"
        fi
    done
    while IFS= read -r name; do
        if [[ $name =~ ^RUNPOD_TCP_PORT_[0-9]+$ ]]; then
            printf 'export %s=%q\n' "$name" "${!name}"
        fi
    done < <(compgen -A variable RUNPOD_TCP_PORT_)
}

export_env_vars() {
    local env_file=/etc/rp_environment tmp
    tmp=$(mktemp "${env_file}.XXXXXX")
    write_ssh_env > "$tmp"
    mv -f "$tmp" "$env_file"
    if ! grep -Fxq 'source /etc/rp_environment' ~/.bashrc; then
        echo 'source /etc/rp_environment' >> ~/.bashrc
    fi
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #


echo "Pod Started"

export_env_vars
setup_ssh
echo "Starting AI Toolkit UI..."
cd /app/ai-toolkit/ui && npm run start
