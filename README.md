# Telegram Predictor

Local desktop/CLI assistant that learns from a private Telegram chat and predicts likely replies in the user's writing style.

The project uses:

- Telegram TDLib for chat sync and live updates
- SQLite for per-chat local state
- Ollama for local LLM predictions
- MLX-LM for optional Apple Silicon LoRA fine-tuning
- Manual and auto-send modes
- Feedback collection and candidate ranking

## MVP Features

- Select a private Telegram chat
- Sync historical messages
- Listen to live incoming/outgoing messages
- Group messages into reply batches
- Generate candidate replies
- Rank candidates
- Send replies manually
- Optional auto-send mode
- Store feedback from chosen/skipped/external replies
- Export per-chat training dataset
- Train and test per-chat MLX LoRA adapter

## Project Structure

```text
src/                 Python source code
docs/                Documentation
Makefile             Common commands
.env.example         Example configuration
requirements.txt     Runtime dependencies
requirements-train.txt Optional training dependencies


cat > Makefile <<'EOF'
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: help install install-train authorize run sync status report select-chat \
        export-training inspect-training prepare-training train-chat-model \
        continue-chat-model test-chat-model check clean

help:
	@echo "Telegram Predictor"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install runtime dependencies"
	@echo "  make install-train        Install runtime + training dependencies"
	@echo "  make authorize            Authorize Telegram TDLib session"
	@echo ""
	@echo "App:"
	@echo "  make run                  Select chat and run live predictor"
	@echo "  make sync                 Sync selected chat history"
	@echo "  make status               Show app status"
	@echo "  make report               Show quality report"
	@echo ""
	@echo "Training:"
	@echo "  make export-training      Export chat training dataset"
	@echo "  make inspect-training     Inspect/filter training dataset"
	@echo "  make prepare-training     Prepare MLX train/valid data"
	@echo "  make train-chat-model     Train MLX LoRA adapter"
	@echo "  make continue-chat-model  Continue MLX LoRA training"
	@echo "  make test-chat-model      Test trained adapter"
	@echo ""
	@echo "Dev:"
	@echo "  make check                Compile-check Python files"
	@echo "  make clean                Remove Python caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-train:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-train.txt

authorize:
	$(PYTHON) -m src.authorize

run: select-chat
	$(PYTHON) -m src.live_sync

sync:
	$(PYTHON) -m src.history_sync

status:
	$(PYTHON) -m src.app_status

report:
	$(PYTHON) -m src.quality_report

select-chat:
	$(PYTHON) -m src.select_chat

export-training:
	$(PYTHON) -m src.export_training_data

inspect-training:
	$(PYTHON) -m src.inspect_training_data

prepare-training:
	$(PYTHON) -m src.prepare_mlx_training_data

train-chat-model:
	$(PYTHON) -m src.train_chat_model

continue-chat-model:
	$(PYTHON) -m src.continue_chat_model

test-chat-model:
	$(PYTHON) -m src.test_chat_model

check:
	$(PYTHON) -m py_compile src/*.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
