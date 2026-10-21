CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic -fPIC
CPP_SOURCE ?= src/numerics.cpp
.PHONY: all test experiments submission clean
all: build/libnumerics.so
build/libnumerics.so: $(CPP_SOURCE) include/numerics.h
	mkdir -p build
	CCACHE_DISABLE=1 $(CXX) $(CXXFLAGS) -shared -Iinclude $(CPP_SOURCE) -o $@
test: all
	PYTHONDONTWRITEBYTECODE=1 python3 tests/check.py
experiments: all
	MPLCONFIGDIR=$(CURDIR)/build/mpl PYTHONDONTWRITEBYTECODE=1 python3 scripts/experiments.py
submission:
	python3 -c "import zipfile; z=zipfile.ZipFile('submission.zip','w',zipfile.ZIP_DEFLATED); [z.write(p,p.split('/')[-1]) for p in ['src/numerics.cpp','python/bindings.py','python/modeling.py']]; z.close()"
clean:
	rm -rf build
