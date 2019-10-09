all: init build 

init:
	pip3 install -r requirements.txt


build: setup.py
	python3 setup.py build 



.PHONY: install
install:
	python3 setup.py install --user


.PHONY: clean
clean:
ifeq ($(OS), Windows_NT)


ifneq (,$(wildcard ./build/*))
	rmdir /S /Q build
endif
ifneq (,$(wildcard ./dist/*))
	rmdir /S /Q dist
endif
ifneq (,$(wildcard ./nPDyn.egg-info/*))
	rmdir /S /Q nPDyn.egg-info
endif
ifneq (,$(wildcard ./nPDyn/lib/*.pyd))
	del .\nPDyn\lib\*.pyd
endif
ifneq (,$(wildcard ./nPDyn/lib/*.c))
	del .\nPDyn\lib\*.c
endif


else

ifneq (,$(wildcard ./build/*))
	rm -rf build
endif
ifneq (,$(wildcard ./dist/*))
	rm -rf dist
endif
ifneq (,$(wildcard ./nPDyn.egg-info/*))
	rm -rf nPDyn.egg-info
endif
ifneq (,$(wildcard ./nPDyn/lib/*.pyd))
	rm ./nPDyn/lib/*.pyd
endif
ifneq (,$(wildcard ./nPDyn/lib/*.c))
	rm ./nPDyn/lib/*.c
endif


endif
