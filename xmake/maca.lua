-- MetaX MACA support for LLAISYS
-- Enable with: xmake f --mx-gpu=y

local maca_path = get_config("maca-path") or "/opt/maca"
local mxdriver_path = get_config("mxdriver-path") or "/opt/mxdriver"
local mxcc = path.join(maca_path, "mxgpu_llvm/bin/mxcc")
local cu_bridge_include = path.join(maca_path, "tools/cu-bridge/include")

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")

    set_toolset("cu", mxcc)
    set_toolset("culd", mxcc)
    add_includedirs(cu_bridge_include, {public = true})
    add_linkdirs(path.join(maca_path, "lib"))
    add_linkdirs(path.join(mxdriver_path, "lib"))

    add_cuflags("-x", "maca")
    add_cuflags("-offload-arch", "native")
    add_cuflags("--maca-path=" .. maca_path)
    add_cuflags("-O3")
    add_cuflags("-fgpu-rdc")
    add_cuflags("-fPIC")
    add_cuflags("-Wno-unknown-pragmas")

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-device-nvidia")

    set_languages("cxx17")
    set_toolset("cu", mxcc)
    set_toolset("culd", mxcc)
    add_includedirs(cu_bridge_include, {public = true})
    add_linkdirs(path.join(maca_path, "lib"))
    add_linkdirs(path.join(mxdriver_path, "lib"))

    add_cuflags("-x", "maca")
    add_cuflags("-offload-arch", "native")
    add_cuflags("--maca-path=" .. maca_path)
    add_cuflags("-O3")
    add_cuflags("-fgpu-rdc")
    add_cuflags("-fPIC")
    add_cuflags("-Wno-unknown-pragmas")

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
