all: init build 

init:
	pip3 install -r requirements.txt


build: setup.py
	python setup.py build 



.PHONY: install
install:
	python setup.py install 


.PHONY: clean
clean:
ifeq ($(OS), Windows_NT)
	rmdir /S /Q build  
	cd nPDyn/lib && del *.c *.cpp
else
	rm -rf build  
	rm nPDyn/lib/*.c 
endif

