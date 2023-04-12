###################################################################
### GPT4ALL / llama.cpp / 
### reference :
###################################################################
from langchain.llms import LlamaCpp, GPT4All
from langchain import PromptTemplate, LLMChain

### compute number of tokens
import tiktoken

#### for measure run time
from datetime import datetime, date,  timedelta

#######################################################################################################
## Select model
base_path="D:\\DataWarehouse\\GPT\\LLAMACPP_Model\\model\\"
## source : https://github.com/nomic-ai/gpt4all
# GPT_PATH = base_path+"gpt4all-converted.bin"

## source : https://huggingface.co/anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g
GPT_PATH =base_path+"gpt4-x-alpaca-13b-native-ggml-q4_0.bin"
##########

enc = tiktoken.encoding_for_model("gpt-4")
################################################################
### question & prompt as context
# question = "Who is the current prime minister of Thailand and what is his impact on Thai political conflicts?"
# question = "Can you tell me about the political status in Thailand?"
# question = "To what year have you been trained with the data?"
question="Can you describe and propose solution based on conflicts which can be seen from the provided context?"

prompt="""Prayut Chan-o-cha (sometimes spelled Prayuth Chan-ocha; Thai: ประยุทธ์ จันทร์โอชา,born 21 March 1954) is a Thai politician and 
    retired army officer[1] who has served as the Prime Minister of Thailand since he seized power in a military coup in 2014.
    He is concurrently the Minister of Defence, a position he has held in his own government since 2019.[2] 
    Prayut served as Commander in Chief of Royal Thai Army from 2010 to 2014[3][4] and led the 2014 Thai coup d'état which installed the National Council for Peace and Order (NCPO), 
    the military junta which governed Thailand between 22 May 2014 and 10 July 2019.[5] The prime minister is the de facto chair of the Cabinet of Thailand. The appointment and removal of ministers can only be made with their advice. As the leader of the government the prime minister is therefore ultimately responsible for the failings and performance of their ministers and the government as a whole. The prime minister cannot hold office for a consecutive period of more than eight years. As the most visible member of the government the prime minister represents the country abroad and is the main spokesperson for the government at home. The prime minister must, under the constitution, lead the cabinet in announcing the government's policy statement in front of a joint-session of the National Assembly, within fifteen days of being sworn-in.[6]

    The prime minister is also directly responsible for many departments. These include the National Intelligence Agency, the Bureau of the Budget, the Office of the National Security Council, the Office of the Council of State, the Office of the Civil Service Commission, the Office of the National Economic and Social Development Board, the Office of Public Sector Development Commission, and the Internal Security Operations Command. Legislatively all money bills introduced in the National Assembly must require the prime minister's approval.

    The prime minister can be removed by a vote of no confidence. This process can be evoked, firstly with the vote of only one-fifth of the members of the House of Representatives for a debate on the matter. Then after the debate a vote is taken and with a simple majority the prime minister can be removed. This process cannot be repeated within one parliamentary session.

    Office and residence
    The prime minister is aided in his work by the Office of the Prime Minister (สำนักนายกรัฐมนตรี) a cabinet-level department headed usually by two ministers of state. These offices are housed in the Government House of Thailand (ทำเนียบรัฐบาล) in the Dusit area of Bangkok.

    The official residence of the prime minister is the Phitsanulok Mansion (บ้านพิษณุโลก), in the center of Bangkok. The mansion was built during the reign of King Vajiravudh. It became an official residence in 1979. The mansion is rumored to have many ghosts, therefore most prime ministers live in their private residences and only use the house for official business.[7][8]"""

# prompt=""" None """
################################################################
### Set token_limit based on the input prompt + question
required_tokens = len(enc.encode(question)) + len(enc.encode(prompt))

print(' --------------------------------------------- ')
print(' required tokens : ',required_tokens)
print(' --------------------------------------------- ')
#################
token_limit= required_tokens    #Default = 512
answer_token=512
#######################################################################################################
template = """
You are a chatbot, only answer the question with shorter than {answer_token} tokens by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'
Context: {prompt}
Question: {question}
Answer: 
"""
prompt1 = PromptTemplate(template=template, input_variables=["prompt","question","answer_token"])

#### set token limit , n_ctx - default is 512
#### ref: https://python.langchain.com/en/latest/reference/modules/llms.html

# llm = LlamaCpp(model_path=GPT_PATH)
llm = LlamaCpp(model_path=GPT_PATH,n_ctx=token_limit+answer_token)

llm_chain = LLMChain(prompt=prompt1, llm=llm)
###############################################################
#### Truncate to meet token limittation =>  llama.cpp sets default token limit to 512 tokens
prompt = prompt[:token_limit]
##############################################################
start_datetime = datetime.now()
response=llm_chain.run({'prompt':prompt,'question':question,'answer_token':answer_token})
print(len(enc.encode(question)),'   +    ',len(enc.encode(prompt)),' ===> \n ',response)
end_datetime = datetime.now()
DIFFTIME = end_datetime - start_datetime 
DIFFTIMEMIN = DIFFTIME.total_seconds()
print('1:Time_use : ',round(DIFFTIMEMIN/60,2), ' Minutes')


print(' ******************************************************************* ')
print(' *******                        D o N e                     ******** ')
print(' ******************************************************************* ')