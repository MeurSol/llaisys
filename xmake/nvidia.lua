-- NVIDIA CUDA support for LLAISYS
-- Enable with: xmake f --nv-gpu=y
--
-- Future extensibility notes:
-- - To add AMD GPU support, create xmake/amd.lua with similar structure
-- - To add other vendors, follow the same pattern: xmake/<vendor>.lua
-- - Each vendor implements the same LlaisysRuntimeAPI interface

-- Option: CUDA compute capability (can be overridden via xmake f --cuda-arch=sm_90)
option("cuda-arch")
    set_default("sm_80")  -- Default to A100 (SM80)
    set_showmenu(true)
    set_description("CUDA compute capability, e.g., sm_80 for A100, sm_90 for H100")
option_end()

-- Helper: get compute capability from option
local function get_cuda_arch()
    local arch = get_config("cuda-arch")
    if arch then
        -- Remove sm_ prefix if present, we'll add it back
        arch = arch:gsub("^sm_", "")
        return arch
    end
    return "80"  -- Default A100
end

local cuda_arch = get_cuda_arch()

-- CUDA device runtime
target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")

    add_rules("cuda")

    -- Support multiple architectures for broader compatibility
    -- Generates code for both virtual architecture (PTX) and real hardware (SASS)
    add_cuflags("-gencode=arch=compute_" .. cuda_arch .. ",code=sm_" .. cuda_arch)
    add_cuflags("-O3")
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")

    add_files("../src/device/nvidia/*.cu")

    add_links("cudart")

    on_install(function (target) end)
target_end()

-- NVIDIA operators
target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-device-nvidia")

    set_languages("cxx17")
    add_rules("cuda")
    add_cuflags("-gencode=arch=compute_" .. cuda_arch .. ",code=sm_" .. cuda_arch)
    add_cuflags("-O3")
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")

    add_files("../src/ops/*/nvidia/*.cu")

    -- CUDA libraries for accelerated operators
    add_links("cudart", "cublas")

    on_install(function (target) end)
target_end()
