# Telegram Predictor

Telegram Predictor is a local desktop/CLI assistant that learns from a private
Telegram chat and predicts likely replies in the user's writing style.

It uses:

- Telegram TDLib for authorization, history sync, and live updates
- SQLite for isolated per-chat state
- Ollama for local reply generation and embedding-based ranking
- MLX-LM for optional Apple Silicon LoRA fine-tuning and generation
- Manual and automatic response modes
- Feedback collection and response matching

## Features

- Select a private Telegram chat
- Sync historical messages and listen for live updates
- Group incoming and outgoing messages into reply batches
- Generate and rank multiple candidate replies
- Send a candidate or custom response from the live CLI
- Optionally send high-scoring candidates automatically
- Store selected, skipped, and externally written responses
- Export, inspect, and prepare per-chat training datasets
- Train, continue, and test a per-chat MLX LoRA adapter
- Inspect application state and response-quality reports

## Requirements

- Python 3 with virtual-environment support
- Telegram API credentials from `https://my.telegram.org`
- A compatible TDLib `libtdjson` shared library
- Ollama with the configured chat and embedding models
- Apple Silicon and MLX-LM only if LoRA training or MLX generation is required

The application is local-first, but TDLib connects to Telegram, Ollama serves
local models, and MLX-LM may download a base model when it is first used.

## Project Structure

```text
src/
  live_sync.py                 Main live application
  telegram_client.py          TDLib client and request handling
  authorize.py                Interactive Telegram authorization
  select_chat.py              Active private-chat selection
  history_sync.py             Historical message synchronization
  message_normalizer.py       TDLib message normalization
  storage.py                  Core per-chat SQLite storage
  prediction_worker.py        Background candidate generation
  predictor.py                Ollama reply generator
  mlx_predictor.py            MLX-LM reply generator
  predictor_factory.py        Generation provider and fallback selection
  ranking.py                  Candidate ranking
  response_matcher.py         Response attribution and feedback matching
  cli.py                      Interactive runtime commands
  export_training_data.py     Per-chat dataset export
  inspect_training_data.py    Interactive dataset review
  prepare_mlx_training_data.py
                               MLX train/validation preparation
  train_chat_model.py         Initial MLX LoRA training
  continue_chat_model.py      Continued MLX LoRA training
  test_chat_model.py          Interactive adapter test
  app_status.py               Application status
  quality_report.py           Stored quality report
Makefile                      Common commands
.env.example                  Example configuration
requirements.txt              Runtime dependencies
requirements-train.txt        Runtime plus MLX-LM dependencies
```

Runtime data, Telegram sessions, datasets, adapters, and local environment files
are intentionally excluded from Git.

## Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   # Runtime only
   make install

   # Runtime plus MLX-LM training and generation support
   make install-train
   ```

   `make install-train` includes the runtime dependencies, so it can be used
   instead of running both installation commands.

3. Install TDLib and configure its library path. See
   [TDLib on macOS](#tdlib-on-macos).

4. Create the local configuration:

   ```bash
   cp .env.example .env
   ```

   Set at least `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`,
   `TDLIB_LIBRARY_PATH`, and a private `TDLIB_DATABASE_ENCRYPTION_KEY`.

5. Start Ollama and make the configured models available:

   ```bash
   ollama pull qwen3:8b
   ollama pull qwen3-embedding:0.6b
   ```

   Use different names if `OLLAMA_CHAT_MODEL` or `OLLAMA_EMBED_MODEL` has been
   changed in `.env`.

6. Verify TDLib without logging in:

   ```bash
   .venv/bin/python -m src.tdlib_smoke_test
   ```

7. Authorize the local TDLib session:

   ```bash
   make authorize
   ```

8. Select a chat and start the live application:

   ```bash
   make run
   ```

### TDLib on macOS

Homebrew's stable TDLib 1.8.0 is too old for current phone-number login. Install
the upstream HEAD build:

```bash
brew install --HEAD tdlib
```

If the stable formula is already installed, unlink it first:

```bash
brew unlink tdlib
brew install --HEAD tdlib
```

Configure the matching library path in `.env`:

```dotenv
# Apple Silicon
TDLIB_LIBRARY_PATH=/opt/homebrew/opt/tdlib/lib/libtdjson.dylib

# Intel
TDLIB_LIBRARY_PATH=/usr/local/opt/tdlib/lib/libtdjson.dylib
```

## Configuration

Copy `.env.example` to `.env` before changing values. Empty required values
cause startup errors.

### Telegram and TDLib

| Variable | Purpose |
| --- | --- |
| `TELEGRAM_API_ID` | Telegram application ID |
| `TELEGRAM_API_HASH` | Telegram application hash |
| `TDLIB_LIBRARY_PATH` | Path to `libtdjson.dylib` or the platform equivalent |
| `TDLIB_DATABASE_DIR` | TDLib database directory |
| `TDLIB_FILES_DIR` | TDLib downloaded-file directory |
| `TDLIB_DATABASE_ENCRYPTION_KEY` | Local TDLib database key; choose a private stable value |
| `TARGET_USER_ID` | Optional fallback when no chat has been selected |
| `TARGET_TDLIB_CHAT_ID` | Optional TDLib chat-ID fallback |

`make run` or `make select-chat` stores the selected IDs in local state, so the
two `TARGET_*` variables normally do not need to be set manually.

### Generation

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHAT_GENERATION_PROVIDER` | `auto` | `auto` selects MLX for a trained chat and Ollama otherwise; explicit values are `ollama` or `mlx` |
| `GENERATION_FALLBACK_PROVIDER` | `ollama` | Provider used when a different primary provider cannot start or generate |
| `GENERATION_FALLBACK_ON_ERROR` | `true` | Enable startup and generation-time provider fallback |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | Ollama reply model |
| `DEFAULT_OLLAMA_CHAT_MODEL` | `qwen3:8b` | Default model used by model resolution |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | Embedding model used for ranking |
| `OLLAMA_KEEP_ALIVE` | `30m` | Ollama model keep-alive duration |
| `MLX_CHAT_MODEL` | training model | Optional MLX base-model override |
| `MLX_MAX_TOKENS` | `700` | Maximum tokens generated by MLX |
| `MLX_TEMPERATURE` | `0.55` | MLX generation temperature |
| `CANDIDATE_COUNT` | `3` | Candidates generated for each incoming batch |
| `MAX_MESSAGES_PER_CANDIDATE` | `4` | Maximum Telegram messages in one candidate |
| `RECENT_CONTEXT_MAX_MESSAGES` | `100` | Recent messages supplied to the generator |

Ollama is still required for embedding-based ranking when MLX generates the
candidate text.

### Sync, Batching, and Sending

| Variable | Example default | Purpose |
| --- | --- | --- |
| `HISTORY_SYNC_LIMIT` | `3000` | Maximum messages fetched by a full sync |
| `STARTUP_SYNC_ON_RUN` | `true` | Sync recent history when `make run` starts |
| `STARTUP_SYNC_LIMIT` | `1000` | Startup-sync message limit |
| `RESULT_JSON_MAX_MESSAGES` | sync limit | Maximum messages retained in `result.json` |
| `INCOMING_BATCH_DELAY_SECONDS` | `5` | Wait before closing an incoming batch |
| `OUTGOING_BATCH_DELAY_SECONDS` | `5` | Wait before closing an outgoing batch |
| `APP_MODE` | `manual` | Initial sending mode: `manual` or `auto` |
| `AUTO_SEND_DELAY_SECONDS` | `2` | Delay before automatic sending |
| `AUTO_MIN_FINAL_SCORE` | `0.0` | Minimum score required for automatic sending |
| `REPLY_USE_PROBABILITY` | `0.4` | Probability of using a Telegram reply reference |

The live `auto` and `manual` commands persist the current mode in the selected
chat's database and override `APP_MODE`.

### Matching and Ranking

| Variable | Default | Purpose |
| --- | --- | --- |
| `MATCH_INFERRED_THRESHOLD` | `0.88` | Similarity threshold for inferred candidate matches |
| `MATCH_INDEPENDENT_THRESHOLD` | `0.55` | Threshold for classifying an independent response |
| `RANK_STYLE_WEIGHT` | `0.35` | Style score contribution |
| `RANK_RELEVANCE_WEIGHT` | `0.35` | Relevance score contribution |
| `RANK_LENGTH_WEIGHT` | `0.10` | Length score contribution |
| `RANK_NATURALNESS_WEIGHT` | `0.20` | Naturalness score contribution |

### Training

| Variable | Example default | Purpose |
| --- | --- | --- |
| `TRAIN_CONTEXT_BLOCKS` | `8` | Conversation blocks included in each example |
| `TRAIN_MIN_TARGET_CHARS` | `1` | Minimum target-response length |
| `TRAIN_MAX_TARGET_CHARS` | `800` | Maximum target-response length |
| `TRAIN_VALID_RATIO` | `0.10` | Validation split ratio |
| `TRAIN_BASE_MODEL` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | MLX base model |
| `TRAIN_ITERS` | `100` | Initial LoRA training iterations |
| `CONTINUE_TRAIN_ITERS` | `20` | Additional continuation iterations |
| `TRAIN_BATCH_SIZE` | `1` | Training batch size |
| `TRAIN_NUM_LAYERS` | `4` | Transformer layers adapted by LoRA |
| `TEST_MAX_TOKENS` | `120` | Maximum tokens during interactive adapter testing |
| `TEST_TEMPERATURE` | `0.55` | Adapter-test generation temperature |

## Make Commands

Run `make help` to print the command list.

| Command | Purpose |
| --- | --- |
| `make install` | Install runtime dependencies |
| `make install-train` | Install runtime and MLX-LM dependencies |
| `make authorize` | Authorize the Telegram TDLib session |
| `make run` | Select a chat and run the live predictor |
| `make sync` | Sync the selected chat's history |
| `make status` | Show state, message, batch, and response summaries |
| `make report` | Show the detailed stored quality report |
| `make select-chat` | Select or keep the active private chat |
| `make export-training` | Export the selected chat's SFT dataset |
| `make inspect-training` | Interactively review and filter the dataset |
| `make prepare-training` | Create MLX train/validation files |
| `make train-chat-model` | Train a new MLX LoRA adapter |
| `make continue-chat-model` | Back up and continue an existing adapter |
| `make test-chat-model` | Test the selected chat's active adapter |
| `make check` | Compile-check all Python source files |
| `make clean` | Remove only Python cache files under `src/` |

`make clean` does not delete `.env`, Telegram sessions, SQLite databases,
datasets, downloaded models, or trained adapters.

## Live Application

`make run` first opens chat selection. Press Enter to keep the active chat or
choose another private chat. The application then performs the configured
startup sync, starts prediction and response-matching workers, and waits for
new messages.

Incoming messages are grouped for `INCOMING_BATCH_DELAY_SECONDS`. Candidate
generation starts when the batch closes.

### Live Commands

| Command | Purpose |
| --- | --- |
| `help`, `h`, `?` | Show live commands |
| `list`, `l` | Show batches waiting for a decision |
| `1`, `2`, `3` | Send that candidate for the oldest pending batch |
| `send <batch> <number>` | Send a candidate from a specific batch |
| `e <text>` | Send custom text for the oldest pending batch |
| `write <batch> <text>` | Send custom text for a specific batch |
| `s` | Skip the oldest pending batch |
| `skip <batch>` | Skip a specific batch |
| `sync [limit]` | Update the selected chat's history |
| `status`, `st` | Show the current application status |
| `model`, `provider`, `generation` | Show provider, model, adapter, and ranking status |
| `mode` | Show the current manual/automatic mode |
| `manual` | Enable manual sending |
| `auto` | Enable automatic sending |
| `quit`, `q`, `exit` | Stop the application |

Automatic mode can send Telegram messages without an additional confirmation.
Test candidate quality in manual mode before enabling it.

## Training Workflow

Training is per selected chat. Each chat receives a separate dataset and
adapter directory.

1. Install MLX-LM support:

   ```bash
   make install-train
   ```

2. Select the intended private chat and sync its history:

   ```bash
   make select-chat
   make sync
   ```

3. Export supervised fine-tuning examples:

   ```bash
   make export-training
   ```

   The raw dataset is written to
   `datasets/chats/<chat_id>/sft.jsonl`.

4. Review the examples:

   ```bash
   make inspect-training
   ```

   Inspector commands:

   | Command | Purpose |
   | --- | --- |
   | Enter, `k` | Keep the current example and continue |
   | `d` | Drop the current example |
   | `u` | Undo a drop |
   | `n`, `p` | Move to the next or previous example |
   | `r` | Open a random example |
   | `j <number>` | Jump to an example |
   | `f` | Write `sft.filtered.jsonl` |
   | `q` | Save review state and quit |

   Filtering is optional. Preparation uses `sft.filtered.jsonl` when it exists
   and falls back to `sft.jsonl`.

5. Prepare deterministic train and validation files:

   ```bash
   make prepare-training
   ```

   At least five examples are required. Outputs are written to
   `datasets/chats/<chat_id>/mlx/train.jsonl` and `valid.jsonl`.

6. Train the adapter:

   ```bash
   make train-chat-model
   ```

   The first run may download `TRAIN_BASE_MODEL`. The adapter is written to
   `adapters/chats/<chat_id>/lora/`.

7. Test the adapter interactively:

   ```bash
   make test-chat-model
   ```

8. Keep automatic per-chat provider selection enabled in `.env`:

   ```dotenv
   CHAT_GENERATION_PROVIDER=auto
   GENERATION_FALLBACK_PROVIDER=ollama
   GENERATION_FALLBACK_ON_ERROR=true
   ```

   `auto` resolves to MLX when the selected chat has
   `adapters.safetensors` and to Ollama when it does not. If MLX cannot start or
   fails while generating, Ollama is used as the fallback. Run `make run`, then
   enter `model` to see the configured and resolved providers.

9. Continue training later if needed:

   ```bash
   make continue-chat-model
   ```

   The current `adapters.safetensors` file is backed up before continuation.

## Status and Diagnostics

```bash
make status
make report
make check
.venv/bin/python -m src.tdlib_smoke_test
```

- `make status` summarizes the active chat database, mode, messages, batches,
  responses, and generation configuration.
- `make report` prints detailed message, prediction, candidate, manual-choice,
  and external-response statistics.
- `make check` checks Python syntax without running Telegram or model services.
- `src.tdlib_smoke_test` verifies that the configured TDLib library loads and
  accepts a static request.

## Local Data

| Path | Contents |
| --- | --- |
| `.env` | Local secrets and configuration |
| `tdlib_data/` | Telegram session database and downloaded files |
| `state/active_chat.json` | Currently selected chat |
| `state/chats/<chat_id>/app.db` | Per-chat messages, batches, candidates, responses, and state |
| `data/chats/<chat_id>/result.json` | Latest normalized per-chat history snapshot |
| `datasets/chats/<chat_id>/` | Exported and prepared training data |
| `adapters/chats/<chat_id>/lora/` | Trained LoRA adapter and metadata |

Back up `.env`, `tdlib_data/`, `state/`, and `adapters/` carefully. They may
contain credentials, authenticated session data, private messages, or learned
information from private conversations.

## Current Limitations

- Chat selection supports private chats only.
- Chat filtering currently matches display titles, not Telegram usernames.
- Empty chat search results are alphabetically ordered rather than using
  Telegram's recent-chat ordering.
- The live `model` command reports generation status; per-chat Ollama model
  selection helpers exist internally but are not exposed through the CLI.
- `src/predictor_smoke_test.py` still expects the legacy global
  `data/result.json` path and is not part of the supported Make workflow.
- `RANK_FEEDBACK_WEIGHT` and the `FEEDBACK_*` values in `.env.example` are
  currently reserved and are not read by the source.

## Localization Compatibility

Source-controlled UI text, logs, errors, prompts, comments, and documentation
use English. Telegram message content remains in its original language, and
generation prompts instruct the model to follow the language implied by the
conversation.

Normalized message format version 2 uses English media markers such as
`[photo]`, `[voice message]`, and `[sticker]`. Existing version 1 records with
non-English markers remain readable and are not migrated automatically. A
subsequent history sync may refresh stored records using version 2 markers.

Training dataset format version 2 uses English prompt scaffolding and role
labels. Re-export and inspect a dataset before new training. Existing adapters
remain usable, but their behavior may reflect the earlier prompt format.
