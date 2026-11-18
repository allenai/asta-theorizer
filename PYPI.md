# asta-theorizer

A Python library that generates literature-grounded scientific theories from a corpus of papers.
You run it a stage at a time (plan, import papers, extract, generate theories), and every stage saves its output to a project directory.

```
pip install asta-theorizer
```

Symbolic law discovery needs the `symbolic` extra: `pip install "asta-theorizer[symbolic]"`.

**Keys.** It calls OpenAI or Anthropic models through litellm. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`,
or put `{"openai": "sk-...", "anthropic": "sk-ant-..."}` in `api_keys.donotcommit.json` in the working
directory or `~/.config/asta/`, which `th.Config.init()` loads; a key already in the environment wins
over the file. With `LITELLM_BASE_URL` and `LITELLM_USER_KEY` set, every call goes through that
LiteLLM proxy instead. The default models are OpenAI's (`gpt-6-sol`, `gpt-6-luna`);
`th.Config.set_models(primary=..., extraction=...)` changes them.

**Papers** are markdown or text files. Convert PDFs to markdown first, or import them with
`ocr="mistral"` (Mistral OCR, needs `MISTRAL_API_KEY`).

```python
import theorizer as th
th.Config.init(project_dir="vip-run")

plan  = th.Pipeline.plan_theory_query("Build a theory of how VIP interneurons control V1 gain")
query = plan["output"]

papers = th.PaperImport.import_papers("/path/to/papers")["papers"]

ext     = th.Pipeline.extract_from_papers(papers, query["extraction_query"],
                                          query["extraction_schema"])
records = th.Pipeline.flatten_records(ext)

theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"])
found    = theories["theory_response"]["theories_general"] + theories["theory_response"]["theories_specific"]
th.Pipeline.to_theorystore(found, query["theory_request_normalized"])   # into the project's theorystore.json

for t in found:
    print(t["theory_name"])

th.Export.export_html()        # vip-run/html/index.html: the theories, their laws, and the evidence behind each
```

The same export from the shell: `python -m theorizer.Export --project vip-run`.

- [AGENTS.md](https://github.com/allenai/asta-theorizer/blob/main/AGENTS.md): the library guide (every stage, the other workflows, settings)
- [README.md](https://github.com/allenai/asta-theorizer/blob/main/README.md): the paper and the project
