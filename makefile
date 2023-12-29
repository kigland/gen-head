# source ./venv/bin/activate

activate:
	source ./venv/bin/activate

install:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt