all: init nPDyn 

init:
	pip3 install -r requirements.txt

nPDyn: setup.py
ifeq ($(OS), Windows_NT)
	python setup.py build --compiler=mingw32
else
	python setup.py build
endif



.PHONY: install
install:
	python setup.py install



.PHONY: clean
clean:
	rm -r build
	cd nPDyn/lib/ && rm *.c 

