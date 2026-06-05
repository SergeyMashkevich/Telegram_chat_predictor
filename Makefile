PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: run sync authorize report status select-chat check install

run: select-chat
	$(PYTHON) -m src.live_sync

sync:
	$(PYTHON) -m src.history_sync

authorize:
	$(PYTHON) -m src.authorize

report:
	$(PYTHON) -m src.quality_report

status:
	$(PYTHON) -m src.app_status

select-chat:
	$(PYTHON) -m src.select_chat

check:
	$(PYTHON) -m py_compile src/*.py

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
