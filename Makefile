JULIA ?= julia
SRC_FILES := $(shell find src -name '*.jl' | sort)

# Regenerate the book's numbered snippets into listings/ from the tagged
# regions in src/. listings/ is deliberately not committed; run this to
# reproduce the exact code printed in the book.
listings:
	$(JULIA) tools/extract.jl --all $(SRC_FILES)

# Remove the regenerated snippets.
clean:
	rm -rf listings

.PHONY: listings clean
