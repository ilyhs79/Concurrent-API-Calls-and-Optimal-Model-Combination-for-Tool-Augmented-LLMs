from nodes.LLMNode import LLMNode
from nodes.Worker import WORKER_REGISTRY
from prompts.planner import *
from utils.util import LLAMA_WEIGHTS


class Planner(LLMNode):
    def __init__(self, workers, prefix=DEFAULT_PREFIX, suffix=DEFAULT_SUFFIX, fewshot=DEFAULT_FEWSHOT,
                 model_name="gpt-4", stop=None):
        super().__init__("Planner", model_name, stop, input_type=str, output_type=str)
        self.workers = workers
        self.prefix = prefix
        self.worker_prompt = self._generate_worker_prompt()
        self.suffix = suffix
        self.fewshot = fewshot

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        prompt = self.prefix + self.worker_prompt + self.fewshot + self.suffix + input + '\n'
        if self.model_name in LLAMA_WEIGHTS:
            prompt = [self.prefix + self.worker_prompt, input]
        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion

    def _get_worker(self, name):
        if name in WORKER_REGISTRY:
            return WORKER_REGISTRY[name]
        else:
            raise ValueError("Worker not found")

    def _generate_worker_prompt(self): # classify 이후 프롬프트 생성하는 것으로 추가
        # Classify prompts
        prompt = "You are a classifier of external tools, Your job is to indicate each tool belongs to which categories by adding prefix. Tools can be one of the following:\n"
        for name in self.workers:
            worker = self._get_worker(name)
            prompt += f"{worker.name}[input]: {worker.description}\n"
        prompt += '''
        There are two prefixs according to their categories of [#1. Task Type] and [#2. Domain Relevance].
        [#1. Information Retrieval] [#2. Common Domain] : Task type is to retrieve information and provide that to user. Domain is not limited.
        [#1. Task Completion] [#2. Math and Science] : Task type is to do a specific action and return the result to user. Domain is mathmatics and science.\n"
        
        Output should be like
        [#1. Information Retrieval] [#2. Common Domain] Google[input]: Worker that searches results from Google. Useful when you need to find short and succinct answers about a specific topic. Input should be a search query.

        Now, classify all the tools by adding prefix to each tools
        '''

        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
                
        prompt = '''
        For the following tasks, make plans that can solve the problem step-by-step. For each plan, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...) 
        Call external tools with same task type together and combine evidences from the tools. Use the combined evidences for the next step.

        Tools can be one of the following:\n
        '''
        prompt += completion
        
        return prompt + "\n"
