PY_FILES = __init__.py blender_importer.py $(wildcard tor2jka/*.py)

# Folder name inside the zip = the add-on's module name in Blender (Install from Disk keeps it).
PACKAGE = Blender_JA_SWTOR_Tools

ZIP_CONTENTS = $(PY_FILES) README.md CHANGELOG.md

PEP8_FILES = $(PY_FILES) tests/*.py

.PHONY: all format pep8 clean

all: build/$(PACKAGE).zip

format:
	autopep8 --in-place $(PEP8_FILES)

pep8:
	pycodestyle --config=.pep8 $(PEP8_FILES)

build/$(PACKAGE).zip: $(ZIP_CONTENTS)
	rm -rf build/$(PACKAGE) $@
	mkdir -p build/$(PACKAGE)
	cp --parents $(ZIP_CONTENTS) build/$(PACKAGE)
	(cd build; zip -r $(PACKAGE).zip $(PACKAGE))

clean:
	rm -rf build
