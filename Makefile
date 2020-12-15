all: init build

init:
	pip3 install -r requirements.txt


build: setup.py
	python3 setup.py build


PACKAGE_DIR = package/


.PHONY: install
install:
	python3 setup.py install --user


.PHONY: clean
clean:
ifeq ($(OS), Windows_NT)


ifneq (,$(wildcard build/*))
	rmdir /S /Q build
endif
ifneq (,$(wildcard dist/*))
	rmdir /S /Q dist
endif
ifneq (,$(wildcard nPDyn.egg-info/*))
	rmdir /S /Q nPDyn.egg-info
endif
ifneq (,$(wildcard $(PACKAGE_DIR)nPDyn/lib/*.pyd))
	del package\nPDyn\lib\*.pyd
endif
ifneq (,$(wildcard $(PACKAGE_DIR)nPDyn/lib/*.c))
	del package\nPDyn\lib\*.c
endif


else

ifneq (,$(wildcard build/*))
	rm -rf build
endif
ifneq (,$(wildcard dist/*))
	rm -rf dist
endif
ifneq (,$(wildcard nPDyn.egg-info/*))
	rm -rf nPDyn.egg-info
endif
ifneq (,$(wildcard $(PACKAGE_DIR)nPDyn/lib/*.pyd))
	rm $(PACKAGE_DIR)nPDyn/lib/*.pyd
endif
ifneq (,$(wildcard $(PACKAGE_DIR)nPDyn/lib/*.c))
	rm $(PACKAGE_DIR)nPDyn/lib/*.c
endif


endif
