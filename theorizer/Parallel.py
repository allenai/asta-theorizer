# Parallel.py
# Run a per-paper function over many papers at once, and hand back the results.
#
# The engine's SchemaExtractionQueue / VariableDiscoveryQueue submit work and are polled for
# completion -- the right shape for a long-running service.  A caller working
# piecewise wants the opposite: "run these papers now and give me the results when they are done".
# This is that, over the same single-paper functions the queues call.
#
# One paper failing does not stop the others.  Failures come back in their own list, so a run of
# 1000 papers with 3 bad PDFs still returns 997 results plus 3 recorded failures.

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import Config


#
#   Checkpoints
#
# A run over a thousand papers takes tens of minutes and costs real money.  Without a checkpoint, a
# crash (or a laptop lid) at paper 900 throws away all 900 -- so each result is appended to a file
# the moment it arrives, and a re-run skips whatever is already in there.
#
# The file is JSONL: one line per finished item, appended under a lock, so it stays readable while
# the run is going and a half-written last line cannot corrupt the rest.

def load_checkpoint(checkpoint_path:str):
    # Returns {item key: result} for everything already finished.  A truncated final line (the
    # process was killed mid-write) is skipped rather than being allowed to fail the whole load.
    done = {}
    if (checkpoint_path is None) or (not os.path.isfile(checkpoint_path)):
        return done

    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if (len(line) == 0):
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if ("key" in entry):
                done[entry["key"]] = entry.get("result")

    return done


def append_checkpoint(checkpoint_path:str, lock, key, result):
    if (checkpoint_path is None):
        return None

    with lock:
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps({"key": key, "result": result}, default=str) + "\n")
            f.flush()

    return checkpoint_path


#
#   Running work in parallel
#

def run_over_items(work_function, items:list, max_workers:int=None, description:str="work", show_progress:bool=True, progress_every:int=None, checkpoint_path:str=None, item_key=None, refresh=None):
    # work_function is called with one item and returns whatever that item's result is.
    # Returns a packed dict with the successes (in the original item order) and the failures.
    #
    # checkpoint_path  append each result as it finishes, and skip anything already there on a re-run
    # item_key         how to identify an item across runs (e.g. lambda paper: paper["paper_id"]).
    #                  Defaults to the item's position, which is only right if the list is unchanged.
    #                  It must change whenever the item changes in a way that changes its result.
    # refresh          refresh(item, cached_result) -> result, applied to a result taken from the
    #                  checkpoint (e.g. to update a file path that moved); None leaves it as saved
    if (max_workers is None):
        max_workers = Config.DEFAULT_MAX_WORKERS

    # A fixed interval of 25 means a 6-paper run prints nothing at all between "starting" and
    # "finished", which is exactly when you are watching it most closely.  Scale it to the work.
    if (progress_every is None):
        progress_every = max(1, min(25, len(items) // 4))

    if (item_key is None):
        item_key = lambda item: None

    results = [None] * len(items)
    failures = []
    num_done = 0
    start_time = time.time()

    # Work out what is already done, and what still needs running
    already_done = load_checkpoint(checkpoint_path)
    checkpoint_lock = threading.Lock()

    todo = []
    num_resumed = 0
    for idx, item in enumerate(items):
        key = item_key(item)
        if (key is None):
            key = idx
        if (key in already_done):
            results[idx] = already_done[key] if (refresh is None) else refresh(item, already_done[key])
            num_resumed += 1
        else:
            todo.append((idx, item, key))

    if (show_progress):
        message = "Parallel.run_over_items(): Starting " + description + " over " + str(len(items)) + " item(s), " + str(max_workers) + " workers"
        if (num_resumed > 0):
            message += " -- " + str(num_resumed) + " already done, " + str(len(todo)) + " to run"
        print(message, flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Keep the index with each future, so results can be put back in the original order
        futures = {}
        for idx, item, key in todo:
            futures[executor.submit(work_function, item)] = (idx, item, key)

        for future in as_completed(futures):
            idx, item, key = futures[future]
            try:
                results[idx] = future.result()
                append_checkpoint(checkpoint_path, checkpoint_lock, key, results[idx])
            except Exception as e:
                # An exception may say why it happened in a word (e.g. "timeout"); keep it if so
                failures.append({"index": idx, "item": item, "error": str(e), "error_type": type(e).__name__, "reason": getattr(e, "reason", None)})

            num_done += 1
            if (show_progress) and ((num_done % progress_every == 0) or (num_done == len(todo))):
                elapsed = time.time() - start_time
                rate = num_done / elapsed if (elapsed > 0) else 0.0
                remaining = ""
                if (rate > 0) and (num_done < len(todo)):
                    remaining = ", ~" + str(round((len(todo) - num_done) / rate)) + "s left"
                print("Parallel.run_over_items(): " + str(num_done) + "/" + str(len(todo)) + " done (" + str(len(failures)) + " failed, " + str(round(elapsed)) + "s elapsed" + remaining + ")", flush=True)

    # Drop the gaps left by failures, keeping the successful results in order
    successes = []
    for result in results:
        if (result is not None):
            successes.append(result)

    elapsed = time.time() - start_time
    if (show_progress):
        print("Parallel.run_over_items(): Finished " + description + ": " + str(len(successes)) + " succeeded, " + str(len(failures)) + " failed, " + str(round(elapsed)) + "s", flush=True)

    # Pack output
    packed = {
        "description": description,
        "num_items": len(items),
        "num_resumed": num_resumed,
        "max_workers": max_workers,
        "results": successes,
        "failures": failures,
        "elapsed_seconds": elapsed,
    }
    return packed
