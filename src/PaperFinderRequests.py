# PaperFinderRequests.py
# A class to handle a queue of requests to the PaperFinder API.

import os
import json
import requests
import time

import threading

from SemanticScholar import *


class PaperFinderRequests():
    #
    #   Constructor
    #
    def __init__(self):
        self.queue = []
        self.finished_work = {}
        self.next_request_id = 1

        # Parameters
        self.MIN_TIME_BETWEEN_REQUESTS_SECS = 2.5        # The minimum time between subsequent Paperfinder requests
        self.last_request_time = 0                     # The timestamp of the last request

        # Update message: Show the user the PaperFinder heartbeat every 10 seconds
        self.heartbeat_interval = 10
        self.heartbeat_last_time = 0

        # Thread lock
        self.THREAD_LOCK_PAPERFINDER_REQUESTS = threading.Lock()
        self.THREAD_ACTIVE = False

        # Start a background thread to process the queue
        self.worker_thread = None
        self.actively_processing_work = False
        self.start_thread()


    # Submit a request to the PaperFinder API, which will add it to the request queue.
    def submit_request(self, query:str):
        with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
            # Generate an ID for the request
            request_id = "paperfinder-request-" + str(self.next_request_id)
            self.next_request_id += 1

            # Add the request to the queue
            self.queue.append({
                'id': request_id,
                'query': query
            })

            return request_id


    # Get finished work
    def get_work(self, request_id:str):
        with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
            # Return a copy of the finished work
            if (request_id in self.finished_work):
                return self.finished_work[request_id]
            else:
                return None

    # Get the number of requests in the queue
    def get_num_requests(self):
        with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
            return len(self.queue)

    def get_num_completed_requests(self):
        return len(self.finished_work)

    # Returns true if there is anything in the queue, or anything currently being processed
    def is_busy(self):
        # Check 1: Are there any requests in the queue?
        if (self.get_num_requests() > 0):
            return True
        # Check 2: Are we actively processing work?
        with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
            if (self.actively_processing_work):
                return True

        # If we reach here, then we're not busy
        return False




    # Add finished work to the directory of finished work
    def _set_work(self, request_id:str, work:dict):
        with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
            # Set the finished work
            self.finished_work[request_id] = work

    # Stop the worker thread
    def stop_thread(self):
        print("PaperFinderRequests.start_thread(): Stopping worker thread...")
        self.THREAD_ACTIVE = False
        if (self.worker_thread is not None):
            self.worker_thread.join()
        self.worker_thread = None

    def start_thread(self):
        if (self.worker_thread is not None) and (self.worker_thread.is_alive()):
            print("PaperFinderRequests.start_thread(): WARNING: Worker thread is already running -- not starting a new one.")
            return
        print("PaperFinderRequests.start_thread(): Starting worker thread...")

        self.THREAD_ACTIVE = True
        # Start the worker thread
        self.worker_thread = threading.Thread(target=self.process_work_monitor, daemon=True)
        self.worker_thread.start()

        print("PaperFinderRequests.start_thread(): Worker thread started.")



    # Perform the paperfinder request
    def perform_work(self, query:str):
        # TODO: Call Paperfinder
        print("PaperFinderRequests.perform_work(): Performing work for query: " + query)

        # Simulate performing work
        result = get_paperfinder_results(query)

        # Create a faux result
        work_result = {
            "result": result
        }

        return work_result


    # Main thread: Process the request queue
    # This should be running in the background.
    def process_work_monitor(self):
        print("PaperFinderRequests.process_work_monitor(): Worker thread started.")
        while (self.THREAD_ACTIVE):
            # Step 0: Check if the heartbeat interval has passed
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                # Print a heartbeat message
                print("PaperFinderRequests.process_work_monitor(): Heartbeat: Worker thread is alive. (queue size: " + str(len(self.queue)) + ")")
                self.heartbeat_last_time = time.time()


            #print("PaperFinderRequests.process_work_monitor(): Worker thread is running...")
            # Step 1: See if enough time has passed since the last request
            time_since_last_request = time.time() - self.last_request_time
            if (time_since_last_request < self.MIN_TIME_BETWEEN_REQUESTS_SECS):
                # Sleep for a short time to avoid busy waiting
                time.sleep(1)
                continue

            # Step 2: If we're here, then enough time has passed since the last request.  Look for work to perform
            request = None
            with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
                # Show queue size periodically
                #print("PaperFinderRequests.process_work_monitor(): Checking for requests in the queue... (queue size: " + str(len(self.queue)) + ")")

                # Check if there are any requests in the queue
                if (len(self.queue)) == 0:
                    # No requests in the queue, so wait a bit before checking again
                    time.sleep(2)
                    continue

                # Get the next request
                request = self.queue.pop(0)


            # Step 2A: If the request is not valid, then sleep and check again in the next iteration
            if (request is None):
                # No request to process, so wait a bit before checking again
                time.sleep(2)
                continue


            # Step 2B: If we reach here, then it's time to process the request
            # TODO: Process request
            with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
                self.actively_processing_work = True

            try:
                work_result = self.perform_work(request['query'])
                # Step 2C: Store the work result
                self._set_work(request['id'], work_result)
            except Exception as e:
                import traceback
                error_str = "PaperFinderRequests.process_work_monitor(): Error processing request '" + request['id'] + "': " + str(e) + "\n" + traceback.format_exc()
                self._set_work(request['id'], {"error": error_str})

            with self.THREAD_LOCK_PAPERFINDER_REQUESTS:
                self.actively_processing_work = False

            # Step 3: Set the time of the last request
            self.last_request_time = time.time()

        # If we reach here, then the thread is being stopped
        print("PaperFinderRequests.process_work_monitor(): Worker thread has completed.")



# Stand-alone examples
if __name__ == "__main__":
    # Make an instance of the class
    paperfinder_requests = PaperFinderRequests()

    # Submit three sample requests
    request_ids = []
    request_ids.append(paperfinder_requests.submit_request("Please find papers about birds"))
    request_ids.append(paperfinder_requests.submit_request("Please find papers about turtles"))
    request_ids.append(paperfinder_requests.submit_request("Please find papers about large language models"))

    time.sleep(2)   # Give a few moments to add work to the queue

    # Wait for all the work to complete
    results = {}

    while (len(results) < len(request_ids)):
        print("* main (outside thread): tick....   (queue size: " + str(paperfinder_requests.get_num_requests()) + ")")
        # Monitor all requests to see which is done

        for request_id in request_ids:
            # Check to see if their work has already been retrieved
            if (request_id not in results):
                work = paperfinder_requests.get_work(request_id)
                if (work is not None):
                    results[request_id] = work
                    print("main: Retrieved work for request ID '" + request_id + "': " + json.dumps(work, indent=2))

        # Save the results file
        with open('paperfinderrequests.json', 'w') as f:
            json.dump(results, f, indent=4)

        time.sleep(1)

    # Stop the thread
    paperfinder_requests.stop_thread()
