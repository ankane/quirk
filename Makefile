test:
	pytest

install:
	pip3 install -r requirements.txt

publish:
	python3 setup.py bdist_wheel --universal
	ls dist
	# twine upload dist/*
	rm -fr build dist quirk.egg-info
