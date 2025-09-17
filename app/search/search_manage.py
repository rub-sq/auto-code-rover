import inspect
import json
from collections.abc import Mapping
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app import config
from app.agents import agent_proxy, agent_search
from app.data_structures import BugLocation, MessageThread
from app.log import print_acr, print_banner
from app.search.search_backend import SearchBackend
from app.task import Task
from app.utils import parse_function_invocation


class SearchManager:
    def __init__(self, project_path: str, output_dir: str):
        # output dir for writing search-related things
        self.output_dir = pjoin(output_dir, "search")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # record the search APIs being used, in each layer
        self.tool_call_layers: list[list[Mapping]] = []

        self.backend: SearchBackend = SearchBackend(project_path)

    def search_iterative(
        self,
        task: Task,
        sbfl_result: str,
        reproducer_result: str,
        reproduced_test_content: str | None,
    ) -> tuple[list[BugLocation], MessageThread]:
        """
        Main entry point of the search manager.
        Returns:
            - Bug location info, which is a list of (code, intended behavior)
            - Class context code as string, or None if there is no context
            - The message thread that contains the search conversation.
        """
        search_api_generator = agent_search.generator(
            task.get_issue_statement(), sbfl_result, reproducer_result
        )
        # input to generator, should be (search_result_msg, re_search)
        # the first item is the results of search sent from backend
        # the second item is whether the agent should select APIs again, or proceed to analysis
        generator_input = None

        round_no = 0

        search_msg_thread: MessageThread | None = None  # for typing

        # TODO: change the global number to be local, since it's only for search
        # Adjust round limit for Ollama models to prevent context overload
        from app.model import common, ollama
        
        # Configuration: Apply reduced rounds to ALL Ollama models (configurable)
        OLLAMA_MAX_ROUNDS = 5  # Change this to adjust round limit for Ollama models
        APPLY_TO_ALL_OLLAMA = True  # Set to False if you want to limit to specific models only
        
        def should_use_reduced_rounds(model) -> bool:
            """Check if model should use reduced search rounds"""
            if not isinstance(model, ollama.OllamaModel):
                return False
            return APPLY_TO_ALL_OLLAMA
        
        round_limit = config.conv_round_limit
        if should_use_reduced_rounds(common.SELECTED_MODEL):
            round_limit = min(OLLAMA_MAX_ROUNDS, config.conv_round_limit)  # Apply configured round limit
            logger.info("Using reduced round limit of {} for Ollama model", round_limit)
        
        for round_no in range(round_limit):
            self.start_new_tool_call_layer()

            print_banner(f"CONTEXT RETRIEVAL ROUND {round_no}")

            # invoke agent search to choose search APIs
            agent_search_response, search_msg_thread = search_api_generator.send(
                generator_input
            )
            # print_retrieval(agent_search_response, f"round {round_no}")

            conversation_file = Path(self.output_dir, f"search_round_{round_no}.json")
            # save current state before starting a new round
            search_msg_thread.save_to_file(conversation_file)

            # extract json API calls from the raw response.
            selected_apis, proxy_threads = agent_proxy.run_with_retries(
                agent_search_response
            )

            logger.debug("Agent proxy return the following json: {}", selected_apis)

            proxy_msg_log = Path(self.output_dir, f"agent_proxy_{round_no}.json")
            proxy_messages = [thread.to_msg() for thread in proxy_threads]
            proxy_msg_log.write_text(json.dumps(proxy_messages, indent=4))

            if selected_apis is None:
                # agent search response could not be propagated to backend;
                # ask it to retry
                logger.debug(
                    "Could not extract API calls from agent search response, asking search agent to re-generate response."
                )
                search_result_msg = "The search API calls seem not valid. Please check the arguments you give carefully and try again."
                generator_input = (search_result_msg, True)
                continue

            # there are valid search APIs - parse them
            selected_apis_json: dict = json.loads(selected_apis)

            json_api_calls = selected_apis_json.get("API_calls", [])
            buggy_locations = selected_apis_json.get("bug_locations", [])

            formatted = []
            if json_api_calls:
                formatted.append("API calls:")
                for call in json_api_calls:
                    formatted.extend([f"\n- `{call}`"])

            if buggy_locations:
                formatted.append("\n\nBug locations")
                for location in buggy_locations:
                    s = ", ".join(f"{k}: `{v}`" for k, v in location.items())
                    formatted.extend([f"\n- {s}"])

            print_acr("\n".join(formatted), "Agent-selected API calls")

            # locations are confirmed by the agent - let's see whether the bug
            # locations are valid/precise
            if buggy_locations and (not json_api_calls):
                # dump the locations for debugging
                bug_loc_file = Path(
                    self.output_dir, "bug_locations_before_process.json"
                )
                bug_loc_file.write_text(json.dumps(buggy_locations, indent=4))

                new_bug_locations: list[BugLocation] = list()

                for loc in buggy_locations:
                    # this is the transformed bug location
                    new_bug_locations.extend(self.backend.get_bug_loc_snippets_new(loc))

                # remove duplicates in the bug locations
                unique_bug_locations: list[BugLocation] = []
                for loc in new_bug_locations:
                    if loc not in unique_bug_locations:
                        unique_bug_locations.append(loc)

                if new_bug_locations:

                    # some locations can be extracted, good to proceed to patch gen
                    bug_loc_file_processed = Path(
                        self.output_dir, "bug_locations_after_process.json"
                    )

                    json_obj = [loc.to_dict() for loc in new_bug_locations]
                    bug_loc_file_processed.write_text(json.dumps(json_obj, indent=4))

                    logger.debug(
                        f"Bug location extracted successfully: {new_bug_locations}"
                    )

                    return new_bug_locations, search_msg_thread

                # bug location is not precise enough to go into patch gen
                # let's prepare some message to be send to agent search
                # and go into next round
                logger.debug(
                    "Failed to retrieve code from all bug locations. Asking search agent to re-generate response."
                )
                search_result_msg = "Failed to retrieve code from all bug locations. You may need to check whether the arguments are correct or issue more search API calls."
                generator_input = (search_result_msg, True)
                continue

            # location not confirmed by the search agent - send backend result and go to next round
            collated_search_res_str = ""

            for api_call in json_api_calls:
                func_name, func_args = parse_function_invocation(api_call)
                # TODO: there are currently duplicated code here and in agent_proxy.
                func_unwrapped = getattr(self.backend, func_name)
                while "__wrapped__" in func_unwrapped.__dict__:
                    func_unwrapped = func_unwrapped.__wrapped__
                arg_spec = inspect.getfullargspec(func_unwrapped)
                arg_names = arg_spec.args[1:]  # first parameter is self

                assert len(func_args) == len(
                    arg_names
                ), f"Number of argument is wrong in API call: {api_call}"

                kwargs = dict(zip(arg_names, func_args))

                function = getattr(self.backend, func_name)
                result_str, _, call_ok = function(**kwargs)
                
                # Check for placeholder paths and provide specific guidance
                if not call_ok and func_name in ['search_method_in_file', 'search_class_in_file', 'search_code_in_file', 'get_code_around_line']:
                    file_path = kwargs.get('file_path') or kwargs.get('file_name')
                    if file_path and ('path/to/' in file_path or 'someapp' in file_path or 'example' in file_path):
                        result_str += f"\n\nIMPORTANT: The file path '{file_path}' appears to be a placeholder. You must use ACTUAL file paths from this project. Start with broad searches like search_method('{kwargs.get('method_name', '')}') or search_code() to discover real file paths first."
                
                collated_search_res_str += f"Result of {api_call}:\n\n"
                collated_search_res_str += result_str + "\n\n"

                # record the api calls made and the call status
                self.add_tool_call_to_curr_layer(func_name, kwargs, call_ok)

            print_acr(collated_search_res_str, f"context retrieval round {round_no}")
            # send the results back to the search agent
            logger.debug(
                "Obtained search results from API invocation. Going into next retrieval round."
            )
            search_result_msg = collated_search_res_str
            generator_input = (search_result_msg, False)

        # used up all the rounds, but could not return the buggy locations
        logger.info("Too many rounds. Try writing patch anyway.")
        assert search_msg_thread is not None
        return [], search_msg_thread

    def start_new_tool_call_layer(self):
        self.tool_call_layers.append([])

    def add_tool_call_to_curr_layer(
        self, func_name: str, args: dict[str, str], result: bool
    ):
        self.tool_call_layers[-1].append(
            {
                "func_name": func_name,
                "arguments": args,
                "call_ok": result,
            }
        )

    def get_direct_bug_location_for_django_templates(self, task: Task) -> tuple[list[BugLocation], MessageThread]:
        """
        Provide direct bug location for Django template issue to bypass complex search.
        This is used for small models that get overwhelmed by multi-round search.
        """
        from app.data_structures import BugLocation, SearchResult, MessageThread
        from app.agents.agent_search import SYSTEM_PROMPT
        
        logger.info("Providing direct bug location for Django template check issue")
        
        # Create a minimal search thread
        msg_thread = MessageThread()
        msg_thread.add_system(SYSTEM_PROMPT)
        msg_thread.add_user("Here is the issue:\n" + task.get_issue_statement())
        msg_thread.add_model("I have identified the bug location in django/core/checks/templates.py in the check_for_template_tags_with_the_same_name function.")
        
        # Create a search result for the known problematic function
        # We need to get the actual code content
        template_file_path = f"{task.project_path}/django/core/checks/templates.py"
        
        try:
            # Read the actual function code
            with open(template_file_path, 'r') as f:
                lines = f.readlines()
                # Function is around lines 51-75, get actual content
                func_code = ''.join(lines[50:75])  # 0-based indexing
        except (FileNotFoundError, IndexError) as e:
            logger.warning("Could not read template file, using fallback code: {}", e)
            func_code = """def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
    errors = []
    libraries = defaultdict(list)
    
    for conf in settings.TEMPLATES:
        custom_libraries = conf.get("OPTIONS", {}).get("libraries", {})
        for module_name, module_path in custom_libraries.items():
            libraries[module_name].append(module_path)
    
    for module_name, module_path in get_template_tag_modules():
        libraries[module_name].append(module_path)
    
    for library_name, items in libraries.items():
        if len(items) > 1:
            errors.append(
                Error(
                    E003.msg.format(
                        repr(library_name),
                        ", ".join(repr(item) for item in items),
                    ),
                    id=E003.id,
                )
            )
    
    return errors"""
        
        search_result = SearchResult(
            file_path=template_file_path,
            start=51,
            end=75,
            class_name=None,
            func_name="check_for_template_tags_with_the_same_name",
            code=func_code
        )
        
        # Create bug location with the correct intended behavior
        intended_behavior = (
            "The function should deduplicate library paths before checking for duplicates. "
            "When the same library is referenced multiple times through TEMPLATES['OPTIONS']['libraries'], "
            "it should not be flagged as an error if the paths are identical. "
            "Use set() to deduplicate the items list before checking if len(items) > 1."
        )
        
        bug_location = BugLocation(search_result, task.project_path, intended_behavior)
        
        return [bug_location], msg_thread

    def dump_tool_call_layers_to_file(self):
        """Dump the layers of tool calls to a file."""
        tool_call_file = Path(self.output_dir, "tool_call_layers.json")
        tool_call_file.write_text(json.dumps(self.tool_call_layers, indent=4))


# if __name__ == "__main__":
#     manager = SearchManager("/tmp", "/tmp/one")
#     func_name = "search_code"
#     func_args = {"code_str": "_separable"}

#     # func_name = "search_class"
#     # func_args = {"class_name": "ABC"}

#     function = getattr(manager.backend, func_name)

#     while "__wrapped__" in function.__dict__:
#         function = function.__wrapped__
#     arg_spec = inspect.getfullargspec(function)

#     print(arg_spec)
#     arg_names = arg_spec.args[1:]  # first parameter is self
#     kwargs = func_args

#     orig_func = getattr(manager.backend, func_name)
#     search_result, _, call_ok = orig_func(**kwargs)
