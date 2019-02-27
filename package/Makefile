all: init nPDyn 

init:
	pip3 install -r requirements.txt

nPDyn: setup.py
	python setup.py build 
	

.PHONY: install
install:
	python setup.py install


.PHONY: clean
ifeq ($(OS), Windows_NT)
clean:
	rmdir build /s /q 
else
clean:
	rm -r build
endif

