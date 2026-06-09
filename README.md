# LLMSecEvalPipeline

A modular, resume-safe pipeline for evaluating the security of LLM-generated Python
code with automated static analysis. Prompts are loaded and cleaned, sent to a local
LLM for code generation, and the generated code is scanned with a SAST tool (Bandit).

This is the research artifact for a BSc thesis at LIACS, Leiden University:
*Assessing the Security of Large Language Model-Generated Code Through Automated
Static Analysis.*

## Pipeline overview

| Stage | Status | Input | Output |
|---|---|---|---|
| **Data Loader** | Done | DevGPT JSON snapshots | `prompts_clean.jsonl` |
| **Code Generator** | Done | `prompts_clean.jsonl` | `generated_code.jsonl` + `code_files/*.py` |
| **SAST Executor** | Done | `generated_code.jsonl` | `sast_findings.jsonl` |
| **Aggregator** | Planned (config stub only) | the three JSONL outputs | merged metrics |

Each stage reads the previous stage's output file, writes its own, and persists
incrementally, so an interrupted run resumes by re-invoking the same command. Stages
share one validated `config.yaml` and a common adapter pattern (an abstract base class
plus concrete backends), so swapping a model or SAST tool means writing one class
rather than re-engineering the pipeline.

A detailed design write-up with per-stage flow diagrams lives in
[walkthrough.md](walkthrough.md).

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) (or LM Studio) serving a code-generation model, for
  the Code Generator stage
- A GPU is recommended for generation but not required for the rest of the pipeline
- The DevGPT dataset (see below), for a real run

The test suite mocks the network and the Bandit subprocess, so it runs offline with no
GPU, no Ollama, and no Bandit binary.

## Installation

```bash
git clone https://github.com/HarryTheNoobyPlayer/LLMSecEvalPipeline.git
cd LLMSecEvalPipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Core + SAST + dev dependencies
pip install -e ".[all]"
```

Optional dependency groups (see `pyproject.toml`): `sast` (Bandit), `dev`
(pytest + coverage), `codegen` (no extras, the Ollama backend uses `requests`), and
`all` (everything).

## Get the dataset

The pipeline is built around [DevGPT](https://github.com/NAIST-SE/DevGPT), a snapshot
of shared ChatGPT conversations. It is not bundled with this repo. Download the DevGPT
JSON snapshots and point `data_loader.dataset_path` in `config.yaml` at the directory
that holds them (default: `./DataSet/DevGPT/`).

## Configuration

All behaviour is driven by `config.yaml`, which is heavily commented. The keys you are
most likely to change:

| Key | What it controls |
|---|---|
| `data_loader.dataset_path` | Where the DevGPT JSON files live |
| `data_loader.sample_size` | `null` for the whole dataset, or an integer to subsample |
| `code_generator.backend` | `ollama` (default) or `lmstudio` |
| `code_generator.ollama_host` | URL of the running Ollama server |
| `code_generator.model_name` | The model tag to pull and serve |
| `sast_executor.tool` | SAST tool to run (`bandit`) |
| `sast_executor.parallel_workers` | Concurrency for scanning |

The config is validated with Pydantic on load, so typos and bad values fail fast with a
clear error.

## Running the pipeline

A unified `cli.py` is planned; until then each stage has a standalone entry point. Run
them in order:

```bash
# 1. Data Loader -> results/data_loader_output/prompts_clean.jsonl
python -m llmseceval.data_loader.loader config.yaml

# 2. Code Generator (needs Ollama reachable; to use a remote GPU host, SSH-forward it:
#    ssh -L 11434:localhost:11434 your-gpu-host)
python -m llmseceval.code_generator.runner config.yaml \
    ./results/data_loader_output/prompts_clean.jsonl \
    ./results/code_generator_output

# 3. SAST Executor -> results/sast_executor_output/sast_findings.jsonl
python -m llmseceval.sast_executor.runner config.yaml \
    ./results/code_generator_output/generated_code.jsonl \
    ./results/sast_executor_output
```

For a quick visual check against a live model or tool there are two `rich`-formatted
smoke scripts:

```bash
python scripts/smoke_test_codegen.py 5   # sample 5 prompts, generate, pretty-print
python scripts/smoke_test_sast.py        # run Bandit on the codegen smoke output
```

## Tests

```bash
pytest          # 81 tests, runs offline in a couple of seconds
```

## Project layout

```
src/llmseceval/
  config.py              Pydantic config models + YAML loader
  data_loader/           DevGPT loading, cleaning, filtering
  code_generator/        Backend-agnostic generation (Ollama, LM Studio)
  sast_executor/         Tool-agnostic SAST scanning (Bandit)
scripts/                 Live smoke tests
tests/                   Unit + integration tests, synthetic fixtures
config.yaml              Default pipeline configuration
walkthrough.md           Detailed design write-up
prd.md                   Product requirements document
```

## Status and roadmap

The data loading, code generation, and SAST stages are implemented, tested, and wired
together end-to-end. Still to come (see the "What's Left" section of `walkthrough.md`):

- **Aggregator** that merges the three JSONL outputs on `prompt_id` and computes the
  thesis metrics (vulnerability prevalence, CWE / severity / confidence distributions,
  MITRE Top-25 overlap).
- **CLI + orchestrator** (`cli.py`, `pipeline.py`) to chain the stages behind a single
  command.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

This is an academic research artifact, but issues and pull requests are welcome. The
adapter pattern makes it straightforward to add a new generation backend (subclass
`BaseCodeGenerator`) or a new SAST tool (subclass `BaseSASTExecutor`).
