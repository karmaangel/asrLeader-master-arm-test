#!/usr/bin/env bash
set -e

prepend_env_path() {
  var_name="$1"
  new_path="$2"
  current_value="${!var_name:-}"

  if [ -z "$new_path" ] || [ ! -e "$new_path" ]; then
    return
  fi

  case ":${current_value}:" in
    *":${new_path}:"*) ;;
    *)
      if [ -n "$current_value" ]; then
        export "${var_name}=${new_path}:${current_value}"
      else
        export "${var_name}=${new_path}"
      fi
      ;;
  esac
}

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-${ASCEND_HOME_PATH}}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-${ASCEND_HOME_PATH}/toolkit}"

# Match the old working image behavior: source CANN before Python starts.
for env_file in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  "${ASCEND_HOME_PATH}/set_env.sh"; do
  if [ -f "$env_file" ]; then
    # shellcheck disable=SC1090
    . "$env_file"
    break
  fi
done

prepend_env_path LD_LIBRARY_PATH /usr/local/Ascend/driver/lib64/common
prepend_env_path LD_LIBRARY_PATH /usr/local/Ascend/driver/lib64/driver
prepend_env_path LD_LIBRARY_PATH /usr/local/Ascend/driver/lib64
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/lib64"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/lib64/plugin/opskernel"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/lib64/plugin/nnengine"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/aarch64-linux/devlib"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/aarch64-linux/lib64"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/aarch64-linux/lib64/stub"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/runtime/lib64"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/runtime/lib64/stub"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/compiler/lib64"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/hccl/lib64"
prepend_env_path LD_LIBRARY_PATH "${ASCEND_HOME_PATH}/tools/aml/lib64"
prepend_env_path LD_LIBRARY_PATH /usr/local/Ascend/atb/latest/atb/cxx_abi_1/lib
prepend_env_path PYTHONPATH "${ASCEND_HOME_PATH}/python/site-packages"
prepend_env_path PYTHONPATH "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe"

# FunASR 1.3.1 in the working production image loads ASR models through the
# ModelScope registry/cache aliases. Resolving aliases to /app/models paths
# makes it fail with "is not registered".
export ASR_RESOLVE_LOCAL_MODELS="${ASR_RESOLVE_LOCAL_MODELS:-false}"

if [ "${POSTPROCESS_PRELOAD:-false}" = "true" ]; then
  postprocess_model_dir="${POSTPROCESS_MODEL_DIR:-}"
  if [ -n "$postprocess_model_dir" ] && [ ! -d "$postprocess_model_dir" ]; then
    export POSTPROCESS_PRELOAD=false
    echo "Disabled post-process preload: missing ${postprocess_model_dir}"
  fi
fi

# The old production pod worked with ASCEND_VISIBLE_DEVICES set to the physical
# NPU id from the device plugin. Keep that default and only normalize when
# explicitly requested.
normalize="${ASR_NORMALIZE_ASCEND_DEVICES:-false}"

visible_devices="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-}}"

if [ "$normalize" != "false" ] && [ -n "$visible_devices" ]; then
  should_normalize=false
  if [ "$normalize" = "true" ]; then
    should_normalize=true
  fi

  if [ "$should_normalize" = "true" ] && [[ "$visible_devices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    original_devices="$visible_devices"
    IFS=',' read -r -a devices <<< "$original_devices"
    local_devices=""
    for index in "${!devices[@]}"; do
      if [ -n "$local_devices" ]; then
        local_devices="${local_devices},${index}"
      else
        local_devices="${index}"
      fi
    done

    export ASCEND_RT_VISIBLE_DEVICES="$local_devices"
    echo "Normalized Ascend runtime devices: ${original_devices} -> ${local_devices}"
  fi
fi

echo "ASR NPU runtime: ASR_DEVICE=${ASR_DEVICE:-} ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-} ASR_RESOLVE_LOCAL_MODELS=${ASR_RESOLVE_LOCAL_MODELS:-} POSTPROCESS_PRELOAD=${POSTPROCESS_PRELOAD:-}"

exec "$@"
