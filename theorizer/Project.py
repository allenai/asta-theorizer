# Project.py
# Where stage output is kept, so an exploration can be put down and picked up later.
#
# The stages are meant to be run piecewise and out of order: extract a little, look at what came
# back, refine the schema, extract again at scale, come back next week and generate more theories
# from what is already on disk.  That only works if everything a stage produces is saved without
# having to remember to save it, which is what this does.
#
# It is a directory of numbered JSON files and nothing else.  There is no database, no object model,
# and no index file that can fall out of step with the directory -- contents() reads the directory
# itself, so moving, copying, renaming or deleting a file with normal tools all work.
#
# Files are numbered in the order they were made and nothing is ever overwritten, so extracting
# three times leaves three files to compare rather than one survivor.

from __future__ import annotations

import hashlib
import json
import os
import time


DIRECTORY = None                        # None: nothing is being saved
DEFAULT_DIRECTORY = "theorizer-run"


#
#   Opening and closing
#

def use(path:str=None):
    # Open a project directory, creating it if it does not exist.  Everything the stages produce
    # from here on is saved into it.
    global DIRECTORY

    if (path is None):
        path = DEFAULT_DIRECTORY

    os.makedirs(path, exist_ok=True)
    DIRECTORY = path

    # The engine's own caches and dumps go here too (see Legacy.point_engine_at())
    from . import Legacy
    Legacy.point_engine_at(DIRECTORY)

    # A different project means different stores; drop the ones held for the previous project so
    # they are reloaded from the new directory on next use.
    global _theory_store, _paper_store
    _theory_store = None
    _paper_store = None

    existing = contents()
    message = "Project.use(): Saving stage output to " + str(os.path.abspath(path))
    if (len(existing) > 0):
        message += " (" + str(len(existing)) + " item(s) already there -- Project.show() to see them)"
    print(message, flush=True)
    return DIRECTORY


def close():
    # Stop saving stage output.  The stages still return their packed dicts; they just stop being
    # written to disk.
    #
    # This also drops the session's TheoryStore and PaperStore, since they belong to the project:
    # leaving them in place would carry one project's ids, extraction results and papers into the
    # next.
    global DIRECTORY, _theory_store, _paper_store

    DIRECTORY = None
    _theory_store = None
    _paper_store = None

    from . import Legacy
    Legacy.point_engine_at(None)
    return None


def is_open():
    return (DIRECTORY is not None)


#
#   The session's TheoryStore
#
# One store per project, held for the session and written to the project directory.
#
# WHY THIS MATTERS.  The store is what assigns ids -- and TheoryStore.set_id() is what stamps a uuid
# onto every extracted record.  Those uuids are how a law cites its evidence: the theory-generation
# prompts ask the model for "the UUIDs that correspond to the specific pieces of evidence", and
# records with no uuid produce theories whose evidence lists are empty and whose laws cannot be
# traced back to a paper.  Extraction that bypasses the store silently breaks that chain.
#
# It is saved and reloaded because TheoryStore.save() persists all_ids and next_ids: reopening a
# project continues the numbering instead of restarting at 1, so theories from a later session
# cannot collide with, or overwrite, the ones already there.

THEORY_STORE_FILENAME = "theorystore.json"

_theory_store = None

# The PaperStore is the other half of that chain, and is saved for the same reason.  It holds every
# paper fetched for the project: the full text, and the Semantic Scholar record behind it (venue,
# abstract, tldr, url, corpus id).  fetch_papers() copies a handful of those fields into the packed
# output it returns, but that is an extract, not the store -- everything else it holds, and the
# ability to reload and add to it in a later session, lives here.
#
# Without this the store was constructed per call and dropped when fetch_papers() returned, so a
# project on disk had no record of its papers beyond that extract, and the HTML exporter had no
# paperstore to render paper titles and links from.

PAPER_STORE_FILENAME = "paperstore.json"

_paper_store = None


def get_theory_store():
    # The session's store, created on first use and reloaded from the project directory if one was
    # saved there before.
    global _theory_store

    if (_theory_store is not None):
        return _theory_store

    from . import Legacy
    _theory_store = Legacy.TheoryStore()

    if (is_open()):
        path = os.path.join(DIRECTORY, THEORY_STORE_FILENAME)
        if (os.path.isfile(path)):
            _theory_store.load(path)
            print("Project.get_theory_store(): Continuing the store in " + str(path) + " (" + str(len(_theory_store.theories)) + " theories, " + str(len(_theory_store.extraction_results)) + " extraction results)", flush=True)

    return _theory_store


def save_theory_store():
    # Write the session's store into the project directory, so the next session continues its ids.
    if (not is_open()) or (_theory_store is None):
        return None

    path = os.path.join(DIRECTORY, THEORY_STORE_FILENAME)
    _theory_store.save(path)
    return path


def get_paper_store():
    # The session's PaperStore, created on first use and reloaded from the project directory if one
    # was saved there before -- so a second run of fetch_papers() adds to the papers already fetched
    # rather than starting empty, and papers fetched last week are still there this week.
    global _paper_store

    if (_paper_store is not None):
        return _paper_store

    from . import Legacy
    _paper_store = Legacy.PaperStore()

    if (is_open()):
        path = os.path.join(DIRECTORY, PAPER_STORE_FILENAME)
        if (os.path.isfile(path)):
            _paper_store.load(path)
            print("Project.get_paper_store(): Continuing the store in " + str(path) + " (" + str(len(_paper_store.paperstore)) + " papers)", flush=True)

    return _paper_store


def save_paper_store():
    # Write the session's PaperStore into the project directory.  This is what keeps the fetched
    # papers -- full text and Semantic Scholar metadata -- after the process exits.
    if (not is_open()) or (_paper_store is None):
        return None

    path = os.path.join(DIRECTORY, PAPER_STORE_FILENAME)
    _paper_store.save(path)
    return path


#
#   Saving and loading
#

def make_slug(name:str):
    # Filenames are made from the name the caller gave, so that a directory listing is readable.
    slug = ""
    for character in str(name).lower():
        if (character.isalnum()):
            slug += character
        elif (len(slug) > 0) and (not slug.endswith("-")):
            slug += "-"

    slug = slug.strip("-")
    if (len(slug) == 0):
        slug = "stage"
    return slug[:60]


def next_index():
    # The next number in the directory.  Reads the directory rather than keeping a counter, so it is
    # still right after files have been moved in, copied out, or deleted by hand.
    highest = 0
    for filename in os.listdir(DIRECTORY):
        if (not filename.endswith(".json")):
            continue
        parts = filename.split("-", 1)
        try:
            index = int(parts[0])
        except ValueError:
            continue
        if (index > highest):
            highest = index

    return highest + 1


def save(name:str, data):
    # Save a stage's output (a packed dict, or a list of records) under a name.  Returns the path it
    # was written to, or None if no project is open.
    #
    # The data is saved exactly as the stage returned it, with nothing wrapped around it, so what
    # load() gives back is what the next stage expects to be handed.
    if (not is_open()):
        return None

    filename = str(next_index()).zfill(2) + "-" + make_slug(name) + ".json"
    full_path = os.path.join(DIRECTORY, filename)

    # Write to a temporary file first, so a crash mid-write leaves no half-written file
    tmp_path = full_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, full_path)

    return full_path


def checkpoint_path(name:str, signature):
    # Where a long parallel stage appends its per-item results while it runs.
    #
    # The filename includes a hash of what the work actually is (the query, the schema, the model),
    # not just the name.  Change the schema and re-run, and you get a fresh checkpoint rather than
    # silently resuming results extracted with the old one -- which would be worse than no
    # checkpointing at all.
    if (not is_open()):
        return None

    directory = os.path.join(DIRECTORY, "checkpoints")
    os.makedirs(directory, exist_ok=True)

    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    return os.path.join(directory, make_slug(name) + "-" + digest + ".jsonl")


def find(name_or_index):
    # Find one saved file by index (4), by name ("extraction-v2"), or by filename.  Raises if the
    # name matches nothing, or more than one thing, rather than guessing which was meant.
    if (not is_open()):
        raise RuntimeError("Project.find(): No project directory is open.  Call Project.use() first.")

    items = contents()
    if (len(items) == 0):
        raise FileNotFoundError("Project.find(): The project directory (" + str(DIRECTORY) + ") is empty.")

    matches = []
    for item in items:
        if (isinstance(name_or_index, int)):
            if (item["index"] == name_or_index):
                matches.append(item)
        elif (str(name_or_index) == item["filename"]) or (str(name_or_index) == item["name"]):
            matches.append(item)

    # Fall back to a partial name match, which is what a caller half-remembering a name will type
    if (len(matches) == 0) and (not isinstance(name_or_index, int)):
        for item in items:
            if (make_slug(name_or_index) in item["name"]):
                matches.append(item)

    if (len(matches) == 0):
        available = []
        for item in items:
            available.append(str(item["index"]) + ":" + item["name"])
        raise FileNotFoundError("Project.find(): Nothing matching '" + str(name_or_index) + "'.  Available: " + ", ".join(available))

    if (len(matches) > 1):
        ambiguous = []
        for item in matches:
            ambiguous.append(str(item["index"]) + ":" + item["name"])
        raise ValueError("Project.find(): '" + str(name_or_index) + "' matches more than one item (" + ", ".join(ambiguous) + ").  Use the index instead.")

    return matches[0]


def load(name_or_index):
    # Read back what a stage saved, by index or by name.  Returns it exactly as the stage returned
    # it, ready to hand to the next stage.
    item = find(name_or_index)
    with open(item["path"]) as f:
        data = json.load(f)

    return data


#
#   Seeing what is there
#

def contents():
    # Everything saved in the project, oldest first.  Reads the directory, so it is always accurate.
    if (not is_open()):
        return []

    items = []
    for filename in sorted(os.listdir(DIRECTORY)):
        if (not filename.endswith(".json")):
            continue

        parts = filename[:-len(".json")].split("-", 1)
        try:
            index = int(parts[0])
        except ValueError:
            continue

        name = parts[1] if (len(parts) > 1) else ""
        full_path = os.path.join(DIRECTORY, filename)

        # Pack output
        item = {
            "index": index,
            "name": name,
            "filename": filename,
            "path": full_path,
            "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 2),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(full_path))),
        }
        items.append(item)

    items = sorted(items, key=lambda item: item["index"])
    return items


def describe(data):
    # A one-line description of a saved item, worked out from the data itself rather than from
    # anything recorded alongside it.
    if (isinstance(data, list)):
        return str(len(data)) + " record(s)"

    if (not isinstance(data, dict)):
        return str(type(data).__name__)

    parts = []
    if ("papers" in data) and (isinstance(data.get("papers"), list)):
        parts.append(str(len(data["papers"])) + " papers")
        if (data.get("num_candidates") is not None):
            parts.append("of " + str(data["num_candidates"]) + " found")
    elif ("theory_response" in data):
        response = data.get("theory_response") or {}
        count = len(response.get("theories_general") or []) + len(response.get("theories_specific") or [])
        parts.append(str(count) + " theories")
    elif ("results" in data):
        parts.append(str(len(data.get("results") or [])) + " results")
    elif ("output" in data):
        # Count whatever the stage produced, rather than just naming the key it lives under -- "5
        # suggestions" is what you want to see when scanning a directory, not "suggestions".
        output = data.get("output")
        if (isinstance(output, dict)):
            counted = []
            for key in sorted(output.keys()):
                value = output[key]
                if (isinstance(value, list)):
                    counted.append(str(len(value)) + " " + key)
            if (len(counted) > 0):
                parts.append(", ".join(counted)[:60])
            else:
                parts.append(", ".join(sorted(output.keys()))[:60])

    if (data.get("model")):
        parts.append(str(data["model"]))
    if (data.get("total_cost")):
        parts.append("$" + str(round(data["total_cost"], 3)))

    return "  ".join(parts)


def show():
    # Print what is in the project.  This is what you run when you come back to something a week
    # later and need to see what you already have.
    items = contents()
    if (not is_open()):
        print("Project.show(): No project directory is open.  Call Project.use() first.", flush=True)
        return items

    print("Project.show(): " + str(os.path.abspath(DIRECTORY)) + " (" + str(len(items)) + " item(s))", flush=True)
    for item in items:
        try:
            with open(item["path"]) as f:
                data = json.load(f)
            summary = describe(data)
        except Exception as e:
            summary = "UNREADABLE (" + str(e) + ")"

        line = "  " + str(item["index"]).zfill(2) + "  " + item["name"].ljust(34) + item["modified"] + "  " + str(item["size_mb"]).rjust(6) + " MB  " + summary
        print(line, flush=True)

    return items
