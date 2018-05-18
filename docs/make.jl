using Documenter, UniversalTensorBoard

makedocs(;
    modules=[UniversalTensorBoard],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/oxinabox/UniversalTensorBoard.jl/blob/{commit}{path}#L{line}",
    sitename="UniversalTensorBoard.jl",
    authors="Lyndon White (aka oxinabox)",
    assets=[],
)

deploydocs(;
    repo="github.com/oxinabox/UniversalTensorBoard.jl",
    target="build",
    julia="0.6",
    deps=nothing,
    make=nothing,
)
