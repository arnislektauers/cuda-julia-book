using Printf

"""
    extract_snippet(input, output, tag)

Extract lines between `# --- begin:TAG ---` and `# --- end:TAG ---` markers
from `input` file and write them to `output` file.
"""
function extract_snippet(input, output, tag)
    begin_tag = "# --- begin:$tag ---"
    end_tag   = "# --- end:$tag ---"

    inside = false
    lines = String[]

    for line in eachline(input)
        if occursin(begin_tag, line)
            inside = true
            continue
        elseif occursin(end_tag, line)
            inside = false
            continue
        end
        inside && push!(lines, line)
    end

    open(output, "w") do io
        for l in lines
            println(io, l)
        end
    end

    @printf("  %s [%s] -> %s\n", input, tag, output)
end

"""
    discover_tags(filepath)

Scan a source file for all `# --- begin:TAG ---` markers and return
the list of tag names found.
"""
function discover_tags(filepath)
    tags = String[]
    for line in eachline(filepath)
        m = match(r"^#\s*---\s*begin:(\S+)\s*---", line)
        if m !== nothing
            push!(tags, m.captures[1])
        end
    end
    return tags
end

"""
    infer_chapter(filepath)

Extract chapter number from source path like `src/chapter05/foo.jl` → `"ch05"`,
or `src/appendix/foo.jl` → `"appx"`.
"""
function infer_chapter(filepath)
    m = match(r"chapter(\d+)", filepath)
    if m !== nothing
        return @sprintf("ch%02d", parse(Int, m.captures[1]))
    end
    if occursin("appendix", filepath)
        return "appx"
    end
    error("Cannot infer chapter from path: $filepath")
end

"""
    batch_extract(src_files)

Auto-discover all tagged regions in the given source files and extract them
to `listings/CHNN_TAG.jl`.
"""
function batch_extract(src_files)
    mkpath("listings")
    total = 0

    for src in src_files
        if !isfile(src)
            @printf("Warning: %s not found, skipping\n", src)
            continue
        end

        tags = discover_tags(src)
        if isempty(tags)
            continue
        end

        ch = infer_chapter(src)
        @printf("Processing %s (%d tags)\n", src, length(tags))

        for tag in tags
            output = joinpath("listings", "$(ch)_$(tag).jl")
            extract_snippet(src, output, tag)
            total += 1
        end
    end

    @printf("\nExtracted %d snippets total.\n", total)
end

# --- CLI dispatch ---
if !isempty(ARGS)
    if ARGS[1] == "--all"
        # Batch mode: julia extract.jl --all src/chapter01/foo.jl src/chapter02/bar.jl ...
        batch_extract(ARGS[2:end])
    elseif length(ARGS) == 3
        # Legacy single mode: julia extract.jl input output tag
        extract_snippet(ARGS[1], ARGS[2], ARGS[3])
    else
        println("""
Usage:
  julia tools/extract.jl --all SRC_FILE [SRC_FILE ...]
      Auto-discover all tagged regions and extract to listings/

  julia tools/extract.jl INPUT OUTPUT TAG
      Extract a single tagged region from INPUT to OUTPUT
""")
        exit(1)
    end
end
