-- MetaX MACA backend integration.

local function push_unique(list, value)
    if not value or value == "" then
        return
    end
    for _, item in ipairs(list) do
        if item == value then
            return
        end
    end
    table.insert(list, value)
end

local function sorted_files(pattern)
    local files = os.files(pattern)
    table.sort(files)
    return files
end

local function build_metax_config()
    local projectdir = os.projectdir()
    local maca_path = get_config("maca-path") or os.getenv("MACA_HOME") or "/opt/maca"
    local mxdriver_path = get_config("mxdriver-path") or "/opt/mxdriver"

    local roots = {}
    push_unique(roots, maca_path)
    push_unique(roots, os.getenv("MACA_HOME"))
    push_unique(roots, "/opt/maca")
    push_unique(roots, "/usr/local/maca")
    push_unique(roots, "/opt/maca-3.3.0")
    push_unique(roots, "/opt/maca-3.2.1")
    push_unique(roots, "/opt/maca-3.2.0")
    push_unique(roots, "/opt/maca-3.1.0")

    local include_dirs = {
        path.join(projectdir, "include"),
        path.join(projectdir, "src")
    }
    local link_dirs = {}
    for _, root in ipairs(roots) do
        push_unique(include_dirs, os.isdir(path.join(root, "include")) and path.join(root, "include") or nil)
        push_unique(include_dirs, os.isdir(path.join(root, "tools", "cu-bridge", "include")) and path.join(root, "tools", "cu-bridge", "include") or nil)
        push_unique(include_dirs, os.isdir(path.join(root, "mxgpu_llvm", "include")) and path.join(root, "mxgpu_llvm", "include") or nil)

        push_unique(link_dirs, os.isdir(path.join(root, "lib")) and path.join(root, "lib") or nil)
        push_unique(link_dirs, os.isdir(path.join(root, "lib64")) and path.join(root, "lib64") or nil)
        push_unique(link_dirs, os.isdir(path.join(root, "mxgpu_llvm", "lib")) and path.join(root, "mxgpu_llvm", "lib") or nil)
        push_unique(link_dirs, os.isdir(path.join(root, "mxgpu_llvm", "lib64")) and path.join(root, "mxgpu_llvm", "lib64") or nil)
    end
    push_unique(link_dirs, os.isdir(path.join(mxdriver_path, "lib")) and path.join(mxdriver_path, "lib") or nil)

    local mxcc = os.getenv("MXCC")
    if not mxcc or mxcc == "" then
        local candidate = path.join(maca_path, "mxgpu_llvm", "bin", "mxcc")
        mxcc = os.isfile(candidate) and candidate or "mxcc"
    end

    return {
        projectdir = projectdir,
        maca_path = maca_path,
        mxdriver_path = mxdriver_path,
        mxcc = mxcc,
        include_dirs = include_dirs,
        link_dirs = link_dirs,
        syslinks = {"mcruntime", "mxc-runtime64", "runtime_cu", "mxsml", "mcblas"},
        common_cxflags = is_plat("windows") and {} or {"-fPIC", "-Wno-unknown-pragmas"},
        maca_compile_flags = {
            "-std=c++17",
            "-x", "maca",
            "-offload-arch", "native",
            "--maca-path=" .. maca_path,
            "-O3",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-DENABLE_METAX_API"
        }
    }
end

local function configure_current_target(cfg)
    set_languages("cxx17")
    set_warnings("all", "error")
    for _, flag in ipairs(cfg.common_cxflags) do
        add_cxflags(flag)
    end
    for _, includedir in ipairs(cfg.include_dirs) do
        add_includedirs(includedir, {public = true})
    end
    for _, linkdir in ipairs(cfg.link_dirs) do
        add_linkdirs(linkdir, {public = true})
    end
    add_syslinks(table.unpack(cfg.syslinks), {public = true})
end

local function wrapper_path(cfg, group_name, source)
    local filename = path.basename(source) .. "_wrapper.cpp"
    return path.join(cfg.projectdir, "build", "_gen", "metax", group_name, filename)
end

local function object_path(cfg, group_name, source)
    local op_name = path.basename(path.directory(path.directory(source)))
    local filename = op_name .. "_" .. path.basename(source) .. ".o"
    return path.join(cfg.projectdir, "build", "_gen", "metax", group_name, filename)
end

local function generate_wrapper_sources(cfg, group_name, sources)
    local wrappers = {}
    for _, source in ipairs(sources) do
        local wrapper = wrapper_path(cfg, group_name, source)
        os.mkdir(path.directory(wrapper))
        io.writefile(wrapper, "#include \"" .. path.translate(source) .. "\"\n")
        table.insert(wrappers, wrapper)
    end
    return wrappers
end

local function compile_maca_objects(cfg, sources, objects)
    for i, source in ipairs(sources) do
        local object = objects[i]
        os.mkdir(path.directory(object))

        local args = {}
        for _, flag in ipairs(cfg.maca_compile_flags) do
            table.insert(args, flag)
        end
        for _, includedir in ipairs(cfg.include_dirs) do
            table.insert(args, "-I" .. includedir)
        end
        table.insert(args, "-c")
        table.insert(args, source)
        table.insert(args, "-o")
        table.insert(args, object)

        os.vrunv(cfg.mxcc, args)
    end
end

local function archive_objects(target, objects)
    local targetfile = target:targetfile()
    local ar = target:tool("ar") or "ar"

    os.mkdir(path.directory(targetfile))
    os.rm(targetfile)

    local args = {"-cr", targetfile}
    for _, object in ipairs(objects) do
        table.insert(args, object)
    end
    os.vrunv(ar, args)
end

local metax = build_metax_config()

target("llaisys-device-metax")
    set_kind("static")
    add_deps("llaisys-utils")
    configure_current_target(metax)

    on_load(function (target)
        local sources = sorted_files(path.join(metax.projectdir, "src", "device", "metax", "*.maca"))
        local wrappers = generate_wrapper_sources(metax, "device", sources)
        for _, wrapper in ipairs(wrappers) do
            target:add("files", wrapper)
        end
    end)

    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    configure_current_target(metax)

    on_load(function (target)
        local sources = sorted_files(path.join(metax.projectdir, "src", "ops", "*", "metax", "*.maca"))
        local objects = {}
        for _, source in ipairs(sources) do
            table.insert(objects, object_path(metax, "ops", source))
        end

        target:data_set("metax_sources", sources)
        target:data_set("metax_objects", objects)
    end)

    on_build(function (target)
        local sources = target:data("metax_sources") or {}
        local objects = target:data("metax_objects") or {}

        compile_maca_objects(metax, sources, objects)
        archive_objects(target, objects)
    end)

    on_install(function (target) end)
target_end()
