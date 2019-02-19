init:
	pip3 install -r requirements.txt

nPDyn:
	ipython setup.py && del nPDyn.c && rmdir build /q /s
