# PaperImport.py
# Bring-your-own papers: load papers the user already has (markdown, text, or PDF) into the shape
# the extraction stages expect.
#
# PaperStore is built around submitting a paper to Semantic Scholar by corpus id and fetching it,
# so it supplies title, authors, abstract and a link as a matter of course.  A user-supplied file
# has none of that -- and an extraction record that says only "paper-doi101016jcell201401050" does
# not tell you where the evidence came from.  So every imported paper gets its metadata read out of
# the document text, and optionally completed from Semantic Scholar by title.
#
# The title is read by an LLM, not by pattern matching.  Picking the title out of a converted PDF is
# a judgement -- headings, journal names, "OPEN ACCESS" banners and preprint watermarks all sit at
# the top of the text -- and a rule-based attempt at this produced titles like "Abstract",
# "OPEN ACCESS" and "Neuron" for 88 papers of 934, with an author-affiliation fallback that was
# worse.  One LLM call per paper gets it right, and costs about $0.003 (measured on a small
# reasoning model, which spends reasoning tokens) -- roughly $3 for a 1000-paper corpus.

from __future__ import annotations

import glob
import hashlib
import json
import os

from . import Config
from . import Legacy
from . import Parallel
from . import Project


#
#   Reading files
#

DEFAULT_EXTENSIONS = [".md", ".markdown", ".txt", ".pdf"]


def find_files(path_or_paths, extensions:list=None):
    # Accepts a directory, a glob, a single file, or a list of any of those.
    if (extensions is None):
        extensions = DEFAULT_EXTENSIONS

    if (isinstance(path_or_paths, str)):
        path_or_paths = [path_or_paths]

    found = []
    for path in path_or_paths:
        if (os.path.isdir(path)):
            for extension in extensions:
                found.extend(sorted(glob.glob(os.path.join(path, "**", "*" + extension), recursive=True)))
        elif (os.path.isfile(path)):
            found.append(path)
        else:
            # Treat anything else as a glob pattern
            found.extend(sorted(glob.glob(path)))

    # Remove duplicates while keeping the order
    unique = []
    seen = set()
    for path in found:
        if (path not in seen):
            seen.add(path)
            unique.append(path)

    return unique


def file_hash(filename:str):
    # A short digest of the file's bytes; of its name when it cannot be read
    digest = hashlib.sha1()
    try:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        digest.update(os.path.basename(filename).encode("utf-8", errors="replace"))
    return digest.hexdigest()[:8]


def make_paper_id(filename:str):
    # A stable, readable id: the start of the filename, then a short hash of the file's content.
    # Papers fetched through PaperStore keep their Semantic Scholar ids; imported papers get one of
    # these instead.
    #
    # The id depends on the file alone, never on which other files are imported with it, so adding
    # a file to a corpus never changes another file's id.  Two files whose names only differ past
    # the 50 characters kept still get different ids, from their content.  Editing a file gives it
    # a new id, which is right: it is a different text, and it is extracted again.
    stem = os.path.splitext(os.path.basename(filename))[0]
    cleaned = ""
    for character in stem.lower():
        if (character.isalnum()):
            cleaned += character
        elif (len(cleaned) > 0) and (not cleaned.endswith("-")):
            cleaned += "-"

    return "paper-" + cleaned.strip("-")[:50].strip("-") + "-" + file_hash(filename)


def make_paper_ids(filenames:list):
    # {filename: id} for a batch.  Each id is make_paper_id()'s, which depends on the file alone, so
    # two files share an id only when their names clean to the same stem AND their content is
    # identical -- the same paper twice.  import_papers() keeps the first of those.
    ids = {}
    for filename in filenames:
        ids[filename] = make_paper_id(filename)

    return ids


def import_signature(read_metadata:bool, model_str:str, ocr:str):
    # What the import checkpoint is keyed to: the settings that change what an import produces.
    # The model is the one actually used, so a defaulted model and the same model named outright
    # share a checkpoint, and changing the default does not reuse the old model's metadata.  Each
    # file within the checkpoint is keyed to its id, which carries a hash of its content.
    if (read_metadata):
        effective_model = model_str if (model_str is not None) else Config.DEFAULT_MODEL_EXTRACTION
    else:
        effective_model = None
    return [read_metadata, effective_model, ocr]


#
#   Metadata
#

METADATA_PROMPT_HEAD = ""
METADATA_PROMPT_HEAD += "You are reading the beginning of a scientific paper, converted to text.\n"
METADATA_PROMPT_HEAD += "Your task is to identify its bibliographic metadata.\n"
METADATA_PROMPT_HEAD += "\n"
METADATA_PROMPT_HEAD += "The text may begin with journal banners, 'OPEN ACCESS' notices, preprint watermarks, DOIs,\n"
METADATA_PROMPT_HEAD += "copyright lines, running heads, or author affiliations. None of those are the title.\n"
METADATA_PROMPT_HEAD += "The title is the specific descriptive name of THIS study.\n"
METADATA_PROMPT_HEAD += "\n"
METADATA_PROMPT_HEAD += "Return your results in JSON format, between a single set of codeblocks (```).\n"
METADATA_PROMPT_HEAD += "Your JSON response must be the last thing you write, and must contain these keys:\n"
METADATA_PROMPT_HEAD += "- `title`(str): The full title of the paper. null if you genuinely cannot find one.\n"
METADATA_PROMPT_HEAD += "- `authors`(list of str): The author names, in order. Empty list if not present.\n"
METADATA_PROMPT_HEAD += "- `year`(int): The publication or preprint year. null if not stated.\n"
METADATA_PROMPT_HEAD += "- `abstract`(str): The abstract, verbatim. null if the paper has none.\n"
METADATA_PROMPT_HEAD += "- `venue`(str): Journal, conference or preprint server. null if not stated.\n"
METADATA_PROMPT_HEAD += "\n"
METADATA_PROMPT_HEAD += "Do not guess. If a field is not present in the text, use null (or an empty list for authors).\n"
METADATA_PROMPT_HEAD += "\n"
METADATA_PROMPT_HEAD += "# Beginning of the paper\n"


def read_metadata_from_text(paper_text:str, model_str:str=None, head_characters:int=6000):
    # One LLM call over the head of the document.  Returns the packed dict from the LLM helper.
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_EXTRACTION

    prompt = METADATA_PROMPT_HEAD + "```\n" + paper_text[:head_characters] + "\n```\n"
    response, response_raw, cost = Legacy.get_llm_response_json(prompt, model_str, temperature=0.0, maxTokens=2000, jsonOut=True)

    metadata = {"title": None, "authors": [], "year": None, "abstract": None, "venue": None}
    if (isinstance(response, dict)):
        for key in metadata:
            if (key in response):
                metadata[key] = response[key]

    # Pack output
    packed = {
        "metadata": metadata,
        "model": model_str,
        "cost": cost,
        "response_raw": response_raw,
    }
    return packed


# The interval comes from Config, which reads the S2_RATE_LIMIT environment variable.  Throttle is
# per-domain, so this does not slow anything else down.  Note that find_paper_by_title() also sleeps
# one second per call in the calling thread, so the lookup pass needs a few workers to reach the
# configured rate at all.
S2_API_HOST = "https://api.semanticscholar.org"

_s2_throttle = None
_s2_throttle_interval = None


def get_s2_throttle():
    # Rebuilt if the setting changed, so S2_RATE_LIMIT can be set after the module was imported.
    global _s2_throttle, _s2_throttle_interval

    interval = Config.get_s2_seconds_between_requests()
    if (_s2_throttle is None) or (interval != _s2_throttle_interval):
        _s2_throttle = Legacy.Throttle(interval)
        _s2_throttle_interval = interval

    return _s2_throttle


def make_semantic_scholar_client():
    # Always pass the key.  Without it the client is unauthenticated and Semantic Scholar returns
    # 429 for anything but the slowest trickle of requests.
    return Legacy.SemanticScholar(api_key=Legacy.load_s2_api_key())


def complete_metadata_from_semantic_scholar(metadata:dict, semantic_scholar=None):
    # Optional second step: look the title up on Semantic Scholar to fill in the things a
    # user-supplied file does not carry -- corpus id, a link, the canonical author list, venue.
    # Uses SemanticScholar.find_paper_by_title(), the title-match endpoint.
    title = metadata.get("title")
    if (title is None) or (len(str(title).strip()) == 0):
        return {"matched": False, "reason": "no title to search with", "metadata": metadata}

    if (semantic_scholar is None):
        semantic_scholar = make_semantic_scholar_client()

    try:
        # Semantic Scholar rate-limits by IP, and returns 429 when several requests arrive at once.
        # Throttle reserves a slot per caller and sleeps outside its lock, so workers queue up
        # rather than all retrying into the same limit.
        get_s2_throttle().wait(S2_API_HOST)
        response = semantic_scholar.find_paper_by_title(str(title))
    except Exception as e:
        return {"matched": False, "reason": "Semantic Scholar lookup failed: " + str(e), "metadata": metadata}

    # The match endpoint returns {"data": [ {...} ]}, and 404s when nothing matches
    match = None
    if (isinstance(response, dict)):
        data = response.get("data")
        if (isinstance(data, list)) and (len(data) > 0):
            match = data[0]

    if (match is None):
        return {"matched": False, "reason": "no Semantic Scholar match for that title", "metadata": metadata}

    completed = dict(metadata)
    completed["corpus_id"] = match.get("corpusId")
    completed["s2_paper_id"] = match.get("paperId")
    completed["url"] = match.get("url")
    completed["s2_title"] = match.get("title")

    # Only fill fields the document itself did not supply
    for key, s2_key in (("year", "year"), ("venue", "venue"), ("abstract", "abstract")):
        if (completed.get(key) is None) and (match.get(s2_key)):
            completed[key] = match.get(s2_key)

    if (len(completed.get("authors") or []) == 0) and (match.get("authors")):
        names = []
        for author in match["authors"]:
            if (isinstance(author, dict)) and (author.get("name")):
                names.append(author["name"])
        completed["authors"] = names

    return {"matched": True, "reason": None, "metadata": completed}


#
#   Importing
#

# The lookup pass is still kept narrow: S2_THROTTLE spaces the requests out, so extra workers would
# only queue up behind it, and a smaller pool makes that obvious in the progress output.
S2_MAX_WORKERS = 4


def title_from_markdown_heading(text:str):
    # The first level-1 heading ("# Title") in a markdown file, or None if it has none.  Only used for
    # files the user wrote as markdown; a PDF's converted text is left to the metadata call (see top).
    for line in text.splitlines():
        stripped = line.strip()
        if (stripped.startswith("# ")):
            title = stripped[2:].strip().rstrip("#").strip()
            if (len(title) > 0):
                return title
    return None


def import_papers(path_or_paths, model_str:str=None, max_workers:int=None, read_metadata:bool=True, lookup_semantic_scholar:bool=False, ocr:str=None, show_progress:bool=True, save_as:str=None):
    # Load every paper under the given path(s) and return them as a list of paper dicts.
    #
    # read_metadata            run the metadata LLM call (leave True unless you already have titles)
    # lookup_semantic_scholar  additionally search S2 by title for corpus id, link and venue
    # ocr                      how PDFs become markdown.  None (default): PDFs are refused -- convert
    #                          them to markdown first.  "mistral": Mistral OCR, which needs
    #                          MISTRAL_API_KEY and costs money per page.
    #
    # Reading files and their metadata is done wide, because it is LLM work.  The Semantic Scholar
    # pass is a separate, narrow pass afterwards, because it is rate-limited.
    Config.ensure_initialized()

    files = find_files(path_or_paths)
    if (len(files) == 0):
        # Silently importing nothing looks identical to a corpus that failed to extract, three
        # stages later.  A wrong path should say so here.
        print("PaperImport.import_papers(): WARNING: No files found at " + str(path_or_paths) + " -- check the path, or the extensions (" + ", ".join(DEFAULT_EXTENSIONS) + ")", flush=True)
    elif (show_progress):
        print("PaperImport.import_papers(): Found " + str(len(files)) + " file(s)", flush=True)

    # Refuse PDFs before any work is done, rather than failing them one by one
    check_pdf_route(files, ocr)

    paper_ids = make_paper_ids(files)

    # The same file twice (same name stem, same bytes) would be extracted, and paid for, twice
    unique_files = []
    seen_ids = set()
    for filename in files:
        if (paper_ids[filename] in seen_ids):
            continue
        seen_ids.add(paper_ids[filename])
        unique_files.append(filename)
    if (len(unique_files) < len(files)) and (show_progress):
        print("PaperImport.import_papers(): " + str(len(files) - len(unique_files)) + " file(s) are copies of another file and were skipped", flush=True)
    files = unique_files

    def load_one(filename):
        text = read_paper_text(filename, ocr=ocr)
        paper = {
            "paper_id": paper_ids[filename],
            "source_file": filename,
            "markdown": text,
            "title": None,
            "authors": [],
            "year": None,
            "abstract": None,
            "venue": None,
            "metadata_cost": 0.0,
        }

        if (read_metadata) and (len(text.strip()) > 0):
            result = read_metadata_from_text(text, model_str=model_str)
            paper.update(result["metadata"])
            paper["metadata_cost"] = result["cost"]

        # A markdown file's own "# Title" heading, when no title was read
        if ((paper["title"] is None) or (len(str(paper["title"]).strip()) == 0)) and (os.path.splitext(filename)[1].lower() in (".md", ".markdown")):
            paper["title"] = title_from_markdown_heading(text)

        # Fall back to the filename so that a paper always has something identifying it
        if (paper["title"] is None) or (len(str(paper["title"]).strip()) == 0):
            paper["title"] = os.path.splitext(os.path.basename(filename))[0]
            paper["title_is_filename_fallback"] = True

        return paper

    # Reading a paper costs an OCR call and a metadata LLM call, so a crash partway through a large
    # corpus is expensive.  Checkpoint per file, keyed to the settings that change what is produced
    # (import_signature()) and, per file, to its id -- which carries a hash of its content, so an
    # edited file is read again.  A file that moved keeps its result and gets its new path.
    def refresh(filename, cached):
        cached["source_file"] = filename
        return cached

    checkpoint = Project.checkpoint_path(save_as if (save_as is not None) else "paper_import",
                                         import_signature(read_metadata, model_str, ocr))
    batch = Parallel.run_over_items(load_one, files, max_workers=max_workers, description="paper import", show_progress=show_progress,
                                    checkpoint_path=checkpoint, item_key=lambda filename: paper_ids[filename], refresh=refresh)

    papers = batch["results"]

    # Second pass: fill in what Semantic Scholar knows that the document did not carry
    if (lookup_semantic_scholar) and (len(papers) > 0):
        semantic_scholar = make_semantic_scholar_client()

        def look_up_one(paper):
            looked_up = complete_metadata_from_semantic_scholar(paper, semantic_scholar=semantic_scholar)
            paper.update(looked_up["metadata"])
            paper["semantic_scholar_matched"] = looked_up["matched"]
            if (not looked_up["matched"]):
                paper["semantic_scholar_reason"] = looked_up["reason"]
            return paper

        Parallel.run_over_items(look_up_one, papers, max_workers=S2_MAX_WORKERS, description="Semantic Scholar lookup", show_progress=show_progress)
    total_cost = 0.0
    for paper in papers:
        total_cost += paper.get("metadata_cost", 0.0)

    if (show_progress):
        num_fallback = 0
        for paper in papers:
            if (paper.get("title_is_filename_fallback")):
                num_fallback += 1
        print("PaperImport.import_papers(): Imported " + str(len(papers)) + " paper(s), $" + str(round(total_cost, 2)) + " (" + str(num_fallback) + " fell back to the filename for a title)", flush=True)

    # Add them to the project's PaperStore under their paper_id, in the record shape fetch_papers()
    # stores, so Evaluate.novelty() and the HTML export find them as they find fetched papers.  No
    # Semantic Scholar call: only what the lookup above found, if it ran, is kept from it.
    paper_store = Project.get_paper_store()
    records = {}
    for paper in papers:
        s2_metadata = {}
        if (paper.get("semantic_scholar_matched")):
            s2_metadata = {"corpusId": paper.get("corpus_id"), "paperId": paper.get("s2_paper_id"), "title": paper.get("s2_title"), "url": paper.get("url")}
        records[paper["paper_id"]] = {
            "status": Legacy.PaperStatus.FULL_TEXT.value,
            "s2_id": paper.get("s2_paper_id"),
            "corpus_id": paper.get("corpus_id"),
            "title": paper.get("title"),
            "publication_year": paper.get("year"),
            "publication_month": None,
            "s2_metadata": s2_metadata,
            "paper_urls": [],
            "paper_markdown": paper.get("markdown"),
        }
        if (paper.get("title_is_filename_fallback")):
            records[paper["paper_id"]]["title"] = None      # filled from the store below, else the filename
    with paper_store.THREAD_LOCK_PAPERSTORE:
        # merge_from_dict() replaces a stored record whole.  A re-import made with less (no Semantic
        # Scholar lookup, read_metadata=False) must not wipe what an earlier import stored, so each
        # field the new record leaves empty keeps its stored value.
        for paper in papers:
            record = records[paper["paper_id"]]
            stored = paper_store.paperstore.get(paper["paper_id"]) or {}
            for field, value in stored.items():
                if (record.get(field) is None) or (record.get(field) == "") or (record.get(field) == []) or (record.get(field) == {}):
                    record[field] = value
            if (record.get("title") is None):
                record["title"] = paper.get("title")
        paper_store.merge_from_dict({"paperstore": records})
    paper_store_path = Project.save_paper_store()

    # Save the corpus, so an import that cost real money is not lost with the session
    if (Project.is_open()):
        saved_path = Project.save(save_as if (save_as is not None) else "papers", papers)
        if (show_progress):
            print("  saved: " + str(saved_path), flush=True)
            if (paper_store_path is not None):
                print("  paperstore: " + str(paper_store_path) + " (" + str(len(paper_store.paperstore)) + " papers)", flush=True)

    # Pack output
    packed = {
        "papers": papers,
        "failures": batch["failures"],
        "num_papers": len(papers),
        "total_cost": total_cost,
        "read_metadata": read_metadata,
        "lookup_semantic_scholar": lookup_semantic_scholar,
        "ocr": ocr,
    }
    if (paper_store_path is not None):
        packed["paper_store_path"] = paper_store_path
    return packed


PDF_HELP = "convert PDFs to markdown first, or pass ocr=\"mistral\" (Mistral OCR; needs MISTRAL_API_KEY)"


def check_pdf_route(files:list, ocr:str):
    # PDFs are read only through Mistral OCR.  Anything else is refused with a message that says
    # what to do, before a single file is read.
    if (ocr is not None) and (ocr != "mistral"):
        raise ValueError("PaperImport: ocr must be None or \"mistral\", not " + repr(ocr) + " -- " + PDF_HELP)

    pdfs = [filename for filename in files if (os.path.splitext(filename)[1].lower() == ".pdf")]
    if (len(pdfs) > 0) and (ocr != "mistral"):
        raise ValueError("PaperImport: " + str(len(pdfs)) + " PDF file(s) found (e.g. " + os.path.basename(pdfs[0]) + ") -- " + PDF_HELP)
    if (len(pdfs) > 0) and (not os.environ.get("MISTRAL_API_KEY")):
        raise ValueError("PaperImport: ocr=\"mistral\" needs MISTRAL_API_KEY, which is not set -- set it, or convert the PDFs to markdown first")


def read_paper_text(filename:str, ocr:str=None):
    # Markdown and text are read directly.  PDFs need ocr="mistral" (see read_pdf_text).
    extension = os.path.splitext(filename)[1].lower()

    if (extension == ".pdf"):
        return read_pdf_text(filename, ocr=ocr)

    with open(filename, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_pdf_text(filename:str, max_pages:int=30, ocr:str=None):
    # PDFs are converted with Mistral OCR and nothing else.  Without ocr="mistral" a PDF is refused:
    # convert it to markdown first.
    if (ocr != "mistral"):
        raise ValueError("PaperImport.read_pdf_text(): " + os.path.basename(filename) + " is a PDF -- " + PDF_HELP)

    size_mb = os.path.getsize(filename) / (1024.0 * 1024.0)
    if (size_mb > MAX_PDF_MB):
        raise RuntimeError("PaperImport.read_pdf_text(): PDF too large to send to Mistral OCR: " + os.path.basename(filename) + " is " + str(round(size_mb, 1)) + " MB, limit is " + str(MAX_PDF_MB) + " MB. Split the PDF, or convert it to markdown first.")

    text = _read_pdf_with_mistral(filename, max_pages)
    if (len(text.strip()) == 0):
        raise RuntimeError("PaperImport.read_pdf_text(): Mistral OCR returned no text for " + str(filename))
    return text


# Mistral will not take a document past this size.  Bigger files are refused with a clear message.
MAX_PDF_MB = 25

# Base64 adds about a third to the size, so files near the limit are uploaded and OCR'd through a
# signed URL rather than sent inline.
MAX_INLINE_PDF_MB = 18


def _read_pdf_with_mistral(filename:str, max_pages:int):
    # MistralOCRStore.process_pdf() only accepts a URL, so a local file needs a different way in.
    # Everything around the call -- the cache, the throttle, turning the OCR response into markdown
    # -- is MistralOCRStore's, so an imported PDF is cached and rate-limited exactly like a fetched
    # one, and re-importing a corpus costs nothing.
    api_key = os.environ.get("MISTRAL_API_KEY")
    if (api_key is None) or (len(api_key) == 0):
        raise RuntimeError("MISTRAL_API_KEY is not set")

    ocr_store = Legacy.MistralOCRStore(api_key=api_key)

    # MistralOCRStore keys its cache by URL; a file:// path gives a stable key for a local file that
    # cannot collide with a real URL.
    cache_key = "file://" + os.path.abspath(filename)
    cached = ocr_store.load_cache(cache_key)
    if (cached is not None):
        return cached

    ocr_store._ocr_throttle.wait()

    size_mb = os.path.getsize(filename) / (1024.0 * 1024.0)
    if (size_mb <= MAX_INLINE_PDF_MB):
        response = _ocr_inline(api_key, filename, max_pages)
    else:
        response = _ocr_by_upload(api_key, filename, max_pages)

    # save_cache() converts the response to markdown and returns it, same as process_pdf() does
    return ocr_store.save_cache(cache_key, response.dict())


def _ocr_inline(api_key:str, filename:str, max_pages:int):
    # One call, nothing left behind on Mistral's side.
    import base64

    from mistralai import Mistral

    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    client = Mistral(api_key=api_key)
    return client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64," + encoded},
        pages=range(0, max_pages + 1),
    )


def _ocr_by_upload(api_key:str, filename:str, max_pages:int):
    # For files too big to send inline: upload, OCR the signed URL, then delete the upload.
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    with open(filename, "rb") as f:
        uploaded = client.files.upload(file={"file_name": os.path.basename(filename), "content": f.read()}, purpose="ocr")

    try:
        signed = client.files.get_signed_url(file_id=uploaded.id)
        return client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed.url},
            pages=range(0, max_pages + 1),
        )
    finally:
        # Clean up the upload whatever happened, but never let cleanup hide the real error
        try:
            client.files.delete(file_id=uploaded.id)
        except Exception:
            pass


#
#   Saving and loading an imported set
#

def save_papers(papers:list, filename:str):
    # One JSON object per line, so a large corpus can be streamed and grepped.
    directory = os.path.dirname(os.path.abspath(filename))
    if (len(directory) > 0):
        os.makedirs(directory, exist_ok=True)

    with open(filename, "w") as f:
        for paper in papers:
            f.write(json.dumps(paper, default=str) + "\n")

    print("PaperImport.save_papers(): Wrote " + str(len(papers)) + " paper(s) to " + str(filename), flush=True)
    return filename


def load_papers(filename:str):
    papers = []
    with open(filename) as f:
        for line in f:
            if (len(line.strip()) > 0):
                papers.append(json.loads(line))

    return papers
