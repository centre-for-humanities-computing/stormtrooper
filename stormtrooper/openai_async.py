"""
Took this module from Kenneth Enevoldsen for reliable and fast async completion with OpenAI's models :)

This script is adapted from OpenAI Cookbook example:
https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional

import openai
import tiktoken  # for counting tokens
from openai.error import OpenAIError, RateLimitError
from pydantic import BaseModel


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


class APIResponse(BaseModel):
    """API responses, including errors, are stored in this format."""

    response: Optional[Any] = None
    errors: Optional[List[str]] = None
    task_id: Optional[int] = None

    def __bool__(self):
        return self.response is not None

    def to_json(self):
        """Convert to a json serializable object"""
        return self.dict()


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = (
        0  # used to cool off after hitting rate limits
    )


@dataclass
class APIRequest:
    """Stores an API request's input, outputs and other metadata along with a method for the API call"""

    task_id: int
    request: List[Dict[str, str]]
    token_consumption: int
    attempts_left: int
    api_errors = None

    async def call_api(
        self,
        responses: List[APIResponse],
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        chatgpt_kwargs: Dict[str, Any],
    ):
        logging.info(f"Starting request {self.task_id}")
        if self.api_errors is None:
            self.api_errors = []
        try:
            response = await openai.ChatCompletion.acreate(
                messages=self.request,
                **chatgpt_kwargs,
            )
            response = response.to_dict_recursive()  # type: ignore
            status_tracker.num_tasks_succeeded += 1
            status_tracker.num_tasks_in_progress -= 1
            api_resp = APIResponse(
                response=response, task_id=self.task_id, errors=self.api_errors
            )
            logging.debug(f"Finished request {self.task_id}")
            responses.append(api_resp)
            return

        except RateLimitError as e:
            logging.warning(
                f"Request {self.task_id} failed with rate limit error {e}"
            )
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
            self.api_errors.append(e.__repr__())

        except OpenAIError as e:
            logging.warning(f"Request {self.task_id} failed with error {e}")
            status_tracker.num_api_errors += 1
            self.api_errors.append(e.__repr__())

        if self.attempts_left:
            retry_queue.put_nowait(self)
            return
        status_tracker.num_tasks_in_progress -= 1
        status_tracker.num_tasks_failed += 1
        logging.warning(f"Request {self.task_id} failed permanently")
        responses.append(
            APIResponse(errors=self.api_errors, task_id=self.task_id)
        )


def num_tokens_consumed_from_chat_request(
    request: List[Dict[str, str]], model_name: str
) -> int:
    """Count the number of tokens in the request. Only supports completion.

    Numbers derived from
    https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
    """
    encoding = tiktoken.encoding_for_model(model_name)
    # if completions request, tokens = prompt + n * max_tokens
    max_tokens = 15
    n = 1
    completion_tokens = n * max_tokens

    num_tokens = 0
    for message in request:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens + completion_tokens


async def chat_parallel_request(
    requests: Iterable[List[Dict[str, str]]],
    max_requests_per_minute: int,
    max_tokens_per_minute: int,
    max_attempts_per_request: int = 5,
    **chatgpt_kwargs: Any,
):
    """Processes API requests in parallel, throttling to stay under rate limits.

    Args:
        requests: An iterable of requests, where each request is a list of messages.
        max_requests_per_minute: The maximum number of requests to send per minute.
        max_tokens_per_minute: The maximum number of tokens to send per minute.
        max_attempts_per_request: The maximum number of times to retry a request.
        chatgpt_kwargs: Keyword arguments to pass to openai.ChatCompletion.create()

    Returns:
        A coroutine that includes the APIReponse objects

    Example:
        >>> messages = [...]
        >>> output = chat_parallel_request(messages, max_requests_per_minute=30, max_tokens_per_minute=1000)
        >>> api_responses = asyncio.run(output)
    """

    logging.info(f"api arguments: {chatgpt_kwargs}")
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    model_name = chatgpt_kwargs.get("model", None)
    if model_name is None:
        model_name = "gpt-3.5-turbo"
        logging.warning(f"No model specified using default model {model_name}")
        chatgpt_kwargs["model"] = model_name

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    gen_not_finished = True  # Once generator is empty
    num_tokens_consumed = partial(
        num_tokens_consumed_from_chat_request, model_name=model_name
    )
    logging.debug(f"Initialization complete.")

    requests = iter(requests)
    responses = []

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(
                    f"Retrying request {next_request.task_id}: {next_request}"
                )
            elif gen_not_finished:
                try:
                    request = next(requests)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request=request,
                        token_consumption=num_tokens_consumed(request),
                        attempts_left=max_attempts_per_request,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(
                        f"Reading request {next_request.task_id}: {next_request}"
                    )
                except StopIteration:
                    logging.debug("Generator is empty.")
                    gen_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60,
            max_requests_per_minute,
        )

        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if there is capacity, make the next request
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # make request
                asyncio.create_task(
                    next_request.call_api(
                        responses=responses,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                        chatgpt_kwargs=chatgpt_kwargs,
                    )
                )
                next_request = None  # reset

        # if all requests are finished, return
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )

        if (
            seconds_since_rate_limit_error
            < seconds_to_pause_after_rate_limit_error
        ):
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error
                - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warning(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    logging.info("Parallel processing complete.")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )

    return responses  # type: ignore


def openai_chatcompletion(
    requests: Iterable[List[Dict[str, str]]],
    max_requests_per_minute: int = 3500,
    max_tokens_per_minute: int = 90_000,
    max_attempts_per_request: int = 5,
    chat_kwargs: Dict[str, Any] = {},
) -> List[APIResponse]:
    """
    A wrapper for the OpenAI Chat Completion API. to avoid calling asyncio.run() directly.
    """

    responses = asyncio.run(
        chat_parallel_request(
            requests=requests,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            max_attempts_per_request=max_attempts_per_request,
            **chat_kwargs,
        )
    )
    sorted_responses = sorted(responses, key=lambda r: r.task_id)
    return sorted_responses
